
import argparse
import logging
import pickle
import numpy as np
import json
from pathlib import Path
# from tqdm import tqdm
from collections import defaultdict
import sys

# Add project root to path
sys.path.append(".")

from src.utils.file_utils import load_trec_run, load_qrels
# from src.retrieval.evaluator import TRECEvaluator

logger = logging.getLogger(__name__)

def evaluate_per_query(run, qrels, evaluator):
    """Evaluate run per query."""
    per_query_metrics = {}
    for qid in run.keys():
        if qid not in qrels: continue
        mini_run = {qid: run[qid]}
        m = evaluator.evaluate_run(mini_run, qrels)
        per_query_metrics[qid] = m
    return per_query_metrics

def extract_features_from_dict(stats_dict, baseline_stats=None):
    """
    Extract enhanced features from the stats dictionary.
    
    Features:
    - Query properties: clarity, entropy, q_len
    - Expansion properties: num_candidates, num_selected, total_value, total_weight, etc.
    - Directional: directional_agreement
    - Baseline features: top1_score, gap (if provided)
    """
    feats = []
    
    # 1. Query Features (Basic)
    feats.append(stats_dict.get('clarity', 0.0))
    feats.append(stats_dict.get('entropy', 0.0))
    feats.append(float(stats_dict.get('q_len', 0)))
    
    # Query Type One-Hot
    q_type = stats_dict.get('q_type', 'informational').lower()
    feats.append(1.0 if q_type == 'navigational' else 0.0)
    feats.append(1.0 if q_type == 'informational' else 0.0)
    feats.append(1.0 if q_type == 'transactional' else 0.0)
    
    # 2. Candidate/Selection Features
    num_cands = stats_dict.get('num_candidates', 0)
    num_selected = stats_dict.get('num_selected_terms', 0)
    feats.append(float(num_cands))
    feats.append(float(num_selected))
    
    # 3. Value/Weight Stats
    selected_terms = stats_dict.get('selected_terms', [])
    
    if selected_terms:
        values = [t.get('value', 0.0) for t in selected_terms]
        weights = [t.get('weight', 0.0) for t in selected_terms]
        
        max_val = max(values)
        avg_val = sum(values) / len(values)
        sum_val = sum(values)
        
        max_w = max(weights)
        avg_w = sum(weights) / len(weights)
        sum_w = sum(weights)
        
        # Avoid div by zero
        ratio_max = max_val / (max_w + 1e-6)
        ratio_avg = avg_val / (avg_w + 1e-6)
    else:
        max_val = avg_val = sum_val = 0.0
        max_w = avg_w = sum_w = 0.0
        ratio_max = ratio_avg = 0.0
        
    feats.extend([max_val, avg_val, sum_val, max_w, avg_w, sum_w, ratio_max, ratio_avg])
    
    # 4. New Advanced Features
    feats.append(stats_dict.get('directional_agreement', 1.0)) # Default to 1 (no drift) if missing
    
    if baseline_stats:
        feats.append(baseline_stats.get('top1_score', 0.0))
        feats.append(baseline_stats.get('gap', 0.0))
    else:
        feats.extend([0.0, 0.0])
        
    # 5. Theoretical Utility Features
    # Semantic Mass: sum(v * c)
    semantic_mass = 0.0
    for t in selected_terms:
        semantic_mass += t.get('value', 0.0) * t.get('count', 1.0)
    feats.append(semantic_mass)
    
    # Expansion Entropy: based on source distribution
    source_counts = stats_dict.get('selected_by_source', {})
    total_sel = sum(source_counts.values())
    entropy = 0.0
    if total_sel > 0:
        for count in source_counts.values():
            if count > 0:
                p = count / total_sel
                entropy -= p * np.log(p + 1e-9)
    feats.append(entropy)
    
    return np.array(feats, dtype=np.float32)

class SimpleMapEvaluator:
    def __init__(self, metrics=None):
        pass
        
    def evaluate_run(self, run, qrels):
        # run: {qid: [(docid, score), ...]}
        # qrels: {qid: {docid: rel}}
        # Calculate AP for the single query in run
        qid = list(run.keys())[0]
        if qid not in qrels:
            return {'map': 0.0}
            
        relevant_docs = qrels[qid]
        retrieved_docs = sorted(run[qid], key=lambda x: x[1], reverse=True)
        
        num_rel = 0
        sum_prec = 0.0
        
        # Count total relevant docs
        total_relevant = sum(1 for r in relevant_docs.values() if r > 0)
        if total_relevant == 0:
            return {'map': 0.0}
            
        for i, (docid, score) in enumerate(retrieved_docs):
            if docid in relevant_docs and relevant_docs[docid] > 0:
                num_rel += 1
                sum_prec += num_rel / (i + 1)
        
        ap = sum_prec / total_relevant
        return {'map': ap}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--msmeqe-run", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--oracle-stats", required=True)
    parser.add_argument("--labels-jsonl", default=None, help="Optional precomputed labels")
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Labels if provided
    precomputed_labels = {}
    if args.labels_jsonl:
        logger.info(f"Loading precomputed labels from {args.labels_jsonl}...")
        with open(args.labels_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                precomputed_labels[data['qid']] = data['label']

    # 2. Load Runs and Qrels (fallback for labeling)
    logger.info("Loading runs and qrels...")
    dense_run = load_trec_run(args.dense_run)
    msmeqe_run = load_trec_run(args.msmeqe_run)
    qrels = load_qrels(args.qrels)
    
    # 3. Load Oracle Stats (Features)
    logger.info("Loading oracle stats JSONL...")
    qid_to_stats = {}
    with open(args.oracle_stats, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                qid_to_stats[data['qid']] = data
            except:
                continue
                
    # 4. Labeling and Feature Extraction
    evaluator = SimpleMapEvaluator(metrics=['map'])
    logger.info("Generating labels and features...")
    
    # Coverage check
    if precomputed_labels:
        common_qids = set(precomputed_labels.keys()) & set(qid_to_stats.keys())
    else:
        common_qids = set(dense_run.keys()) & set(msmeqe_run.keys()) & set(qid_to_stats.keys()) & set(qrels.keys())
        
    logger.info(f"Overlap queries: {len(common_qids)}")
    
    if not precomputed_labels:
        dense_map = evaluate_per_query(dense_run, qrels, evaluator)
        msmeqe_map = evaluate_per_query(msmeqe_run, qrels, evaluator)
    
    # epsilon for delta-aware labeling
    epsilon = 0.0001
    
    features_list = []
    y_gain = []
    y_risk = []
    y_align = []
    qids_list = []
    
    for qid in common_qids:
        # 1. Labels
        d_score = dense_map[qid]['map']
        m_score = msmeqe_map[qid]['map']
        delta = m_score - d_score
        
        # Gain is the continuous delta
        y_gain.append(delta)
        # Risk is 1 if expansion HURT (delta < 0)
        y_risk.append(1 if delta < 0 else 0)
        
        # 2. Alignment Label (Breakthrough head)
        y_align.append(qid_to_stats[qid].get('delta_align', 0.0))
        
        # 2. Baseline Status Features
        sorted_dense = sorted(dense_run[qid], key=lambda x: x[1], reverse=True)
        top1_score = sorted_dense[0][1] if len(sorted_dense) > 0 else 0.0
        top2_score = sorted_dense[1][1] if len(sorted_dense) > 1 else 0.0
        gap = top1_score - top2_score
        
        baseline_stats = {
            'top1_score': top1_score,
            'gap': gap
        }
        
        # 3. Features
        stats = qid_to_stats[qid]
        feats = extract_features_from_dict(stats, baseline_stats=baseline_stats)
        
        features_list.append(feats)
        qids_list.append(qid)
        
    X = np.vstack(features_list)
    
    dataset = {
        'X': X,
        'y_gain': np.array(y_gain, dtype=np.float32),
        'y_risk': np.array(y_risk, dtype=np.int32),
        'y_align': np.array(y_align, dtype=np.float32),
        'qids': qids_list,
        'feature_names': [
            'clarity', 'entropy', 'q_len', 
            'nav', 'info', 'trans',
            'num_cands', 'num_sel',
            'max_val', 'avg_val', 'sum_val',
            'max_w', 'avg_w', 'sum_w',
            'val_w_max', 'val_w_avg',
            'agreement', 'base_top1', 'base_gap',
            'semantic_mass', 'exp_entropy'
        ]
    }
    
    logger.info(f"Constructed dataset: X {X.shape}, y_gain {len(y_gain)}, y_risk {sum(y_risk)}")
    
    # 5. Save
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
        
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

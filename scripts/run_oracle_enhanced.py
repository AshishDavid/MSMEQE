
import argparse
import logging
import pickle
import numpy as np
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(".")

from src.utils.file_utils import load_trec_run, save_trec_run
from src.models.budget_predictor import BudgetPredictor

logger = logging.getLogger(__name__)

def extract_features_from_dict(stats_dict, baseline_stats=None):
    """
    Extract enhanced features from the stats dictionary.
    MUST MATCH create_enhanced_oracle_data.py EXACTLY!
    """
    feats = []
    
    # 1. Query Features
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
        
        ratio_max = max_val / (max_w + 1e-6)
        ratio_avg = avg_val / (avg_w + 1e-6)
    else:
        max_val = avg_val = sum_val = 0.0
        max_w = avg_w = sum_w = 0.0
        ratio_max = ratio_avg = 0.0
        
    feats.extend([max_val, avg_val, sum_val, max_w, avg_w, sum_w, ratio_max, ratio_avg])
    
    # 4. Advanced Features
    feats.append(stats_dict.get('directional_agreement', 1.0))
    
    if baseline_stats:
        feats.append(baseline_stats.get('top1_score', 0.0))
        feats.append(baseline_stats.get('gap', 0.0))
    else:
        feats.extend([0.0, 0.0])

    # 5. Theoretical Utility Features
    semantic_mass = 0.0
    for t in selected_terms:
        semantic_mass += t.get('value', 0.0) * t.get('count', 1.0)
    feats.append(semantic_mass)
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--msmeqe-run", required=True)
    parser.add_argument("--oracle-stats", required=True)
    parser.add_argument("--model-path", help="Legacy single model path (optional)")
    parser.add_argument("--gain-model", help="Path to gain regressor pkl")
    parser.add_argument("--risk-model", help="Path to risk classifier pkl")
    parser.add_argument("--align-model", help="Path to alignment regressor pkl")
    parser.add_argument("--alpha", type=float, default=0.1, help="Risk aversion parameter")
    parser.add_argument("--beta", type=float, default=0.1, help="Alignment rescue parameter")
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Runs
    logger.info("Loading runs...")
    dense_run = load_trec_run(args.dense_run)
    msmeqe_run = load_trec_run(args.msmeqe_run)
    
    # 2. Load Models
    gain_model = None
    risk_model = None
    align_model = None
    
    if args.gain_model and args.risk_model:
        logger.info(f"Loading Gain model from {args.gain_model}...")
        with open(args.gain_model, 'rb') as f:
            gain_model = pickle.load(f)
        logger.info(f"Loading Risk model from {args.risk_model}...")
        with open(args.risk_model, 'rb') as f:
            risk_model = pickle.load(f)
        if args.align_model:
            logger.info(f"Loading Alignment model from {args.align_model}...")
            with open(args.align_model, 'rb') as f:
                align_model = pickle.load(f)
    elif args.model_path:
        logger.info(f"Loading legacy model from {args.model_path}...")
        with open(args.model_path, 'rb') as f:
            predictor = pickle.load(f)
    else:
        logger.error("Must provide either --model-path or both --gain-model and --risk-model")
        return

    # 3. Load Oracle Stats
    logger.info("Loading oracle stats...")
    qid_to_stats = {}
    with open(args.oracle_stats, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                qid_to_stats[data['qid']] = data
            except:
                continue
                
    # 4. Predict and Switch
    final_run = {}
    switch_counts = {0: 0, 50: 0} # 0=NoExp, 50=Exp
    
    all_qids = set(dense_run.keys()) | set(msmeqe_run.keys())
    
    logger.info(f"Predicting (Alpha={args.alpha})...")
    for qid in all_qids:
        # Default to dense if missing stats
        if qid not in qid_to_stats:
            final_run[qid] = dense_run.get(qid, [])
            switch_counts[0] += 1
            continue
            
        # Extract features
        stats = qid_to_stats[qid]
        
        # Baseline Status Features
        sorted_dense = sorted(dense_run[qid], key=lambda x: x[1], reverse=True)
        top1_score = sorted_dense[0][1] if len(sorted_dense) > 0 else 0.0
        top2_score = sorted_dense[1][1] if len(sorted_dense) > 1 else 0.0
        gap = top1_score - top2_score
        
        baseline_stats = {
            'top1_score': top1_score,
            'gap': gap
        }
        
        feats = extract_features_from_dict(stats, baseline_stats=baseline_stats)
        X = feats.reshape(1, -1)
        
        # Decision logic
        expand = False
        if gain_model and risk_model:
            # Expected Utility Rule: U = mu - alpha * rho
            mu = gain_model.predict(X)[0]
            # Risk rho is the probability of loss (class 1)
            rho = risk_model.predict_proba(X)[0, 1]
            
            # Alignment eta (optional)
            eta = 0.0
            if align_model:
                eta = align_model.predict(X)[0]
                
            utility = mu + args.beta * eta - args.alpha * rho
            if utility > 0:
                expand = True
        else:
            # Legacy binary logic
            expand = predictor.predict(X)[0] > 0
        
        # Determine final results
        m_results = msmeqe_run.get(qid, [])
        d_results = dense_run.get(qid, [])
        
        if expand and len(m_results) > 0:
            final_run[qid] = m_results
            switch_counts[50] += 1
        else:
            final_run[qid] = d_results
            switch_counts[0] += 1
            
    logger.info(f"Switch counts: {switch_counts}")
    
    # 5. Save
    save_trec_run(final_run, args.output, "MS-MEQE-Oracle-Enhanced")

if __name__ == "__main__":
    main()

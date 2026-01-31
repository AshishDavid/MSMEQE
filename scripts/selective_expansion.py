
import argparse
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_trec_run, load_qrels, save_trec_run

logger = logging.getLogger(__name__)

def evaluate_per_query(run, qrels, evaluator):
    """
    Evaluate a run per query.
    Returns dict: qid -> metrics_dict
    """
    # TREC Evaluator aggregates, so we need to loop manually or force it
    # Actually, TRECEvaluator.evaluate_run might not return per-query.
    # Let's implement a simple version or rely on valid subsetting.
    
    # We will verify if query exists in qrels, else skip
    per_query_metrics = {}
    
    for qid in run.keys():
        if qid not in qrels: 
            continue
            
        # Create mini-run for just this query
        mini_run = {qid: run[qid]}
        # Evaluate
        # We assume evaluate_run works for single query
        # Note: evaluate_run returns AGGREGATES (mean). 
        # For N=1, Mean = Exact value.
        m = evaluator.evaluate_run(mini_run, qrels)
        per_query_metrics[qid] = m
        
    return per_query_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--msmeqe-run", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--stats-file", required=True, help="JSONL with query stats (clarity)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    # 1. Load Data
    logger.info("Loading runs...")
    dense_run = load_trec_run(args.dense_run)
    msmeqe_run = load_trec_run(args.msmeqe_run)
    qrels = load_qrels(args.qrels)
    
    logger.info("Loading stats...")
    query_stats = {}
    with open(args.stats_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            query_stats[str(item['qid'])] = item

    # 2. Evaluate Per Query (to build Oracle)
    logger.info("Evaluating Dense per query...")
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'recall_1000'])
    dense_metrics = evaluate_per_query(dense_run, qrels, evaluator)
    
    logger.info("Evaluating MS-MEQE per query...")
    msmeqe_metrics = evaluate_per_query(msmeqe_run, qrels, evaluator)
    
    # 3. Oracle Combination
    oracle_run = {}
    oracle_source_counts = {"dense": 0, "msmeqe": 0}
    
    common_qids = set(dense_metrics.keys()) & set(msmeqe_metrics.keys())
    
    for qid in common_qids:
        d_map = dense_metrics[qid]['map']
        m_map = msmeqe_metrics[qid]['map']
        
        if d_map > m_map:
            oracle_run[qid] = dense_run[qid]
            oracle_source_counts['dense'] += 1
        else:
            oracle_run[qid] = msmeqe_run[qid]
            oracle_source_counts['msmeqe'] += 1
            
    # Evaluate Oracle
    logger.info(f"Oracle Sources: {oracle_source_counts}")
    oracle_agg = evaluator.evaluate_run(oracle_run, qrels)
    print("\n=== ORACLE (UPPER BOUND) ===")
    for k, v in oracle_agg.items():
        print(f"{k:<15}: {v:.4f}")
    print("============================\n")
    
    # 4. Selective Expansion (Clarity Threshold)
    # We want to pick MS-MEQE only if Clarity is LOW (Ambiguous)
    # High Clarity -> Low Entrop -> Simple -> Dense sufficient?
    # Actually, High Clarity means "Clear semantics". Dense usually works well.
    # Low Clarity means "Vague". Expansion helps.
    
    best_thresh_map = 0
    best_thresh = 0
    best_run = {}
    
    # Sweep thresholds
    thresholds = [0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 100.0]
    
    logger.info("Sweeping Clarity Thresholds...")
    for thresh in thresholds:
        combined_run = {}
        for qid in common_qids:
            # If stats missing, assume complex (expand) or simple? 
            # Default to MS-MEQE
            clarity = 0.0
            if qid in query_stats:
                clarity = query_stats[qid].get('clarity', 0.0)
            
            # If clarity >= Thresh -> "Clear" -> Use Dense
            # Else -> "Ambiguous" -> Use MS-MEQE
            if clarity >= thresh:
                combined_run[qid] = dense_run[qid]
            else:
                combined_run[qid] = msmeqe_run[qid]
        
        agg = evaluator.evaluate_run(combined_run, qrels)
        curr_map = agg['map']
        logger.info(f"Threshold {thresh:>4}: MAP {curr_map:.4f}")
        
        if curr_map > best_thresh_map:
            best_thresh_map = curr_map
            best_thresh = thresh
            best_run = combined_run
            
    print("\n=== SELECTIVE EXPANSION (BEST) ===")
    print(f"Best Clarity Threshold: {best_thresh}")
    best_agg = evaluator.evaluate_run(best_run, qrels)
    for k, v in best_agg.items():
        print(f"{k:<15}: {v:.4f}")
    print("==================================\n")
    
    # Save Best Run
    save_trec_run(best_run, args.output, "MS-MEQE-Selective")
    logger.info(f"Saved Selective Run to {args.output}")

if __name__ == "__main__":
    main()

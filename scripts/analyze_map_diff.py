
import argparse
import logging
from typing import Dict, List
import pandas as pd
from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_trec_run, load_qrels

def analyze_runs(dense_run_path: str, msmeqe_run_path: str, qrels_path: str):
    qrels = load_qrels(qrels_path)
    
    # Load Runs
    dense_run = load_trec_run(dense_run_path)
    msmeqe_run = load_trec_run(msmeqe_run_path)
    
    # Evaluate Per Query
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10'])
    
    # We need to evaluate each query individually to get the deltas
    # TRECEvaluator doesn't expose per-query easily in one call, so we loop or hack it.
    # Actually, evaluate_run returns global. We need per-query.
    
    # Let's verify queries intersection
    queries = set(dense_run.keys()) & set(msmeqe_run.keys())
    print(f"Analyzing {len(queries)} common queries...")
    
    results = []
    
    for qid in queries:
        # Dense Stats
        d_res = {qid: dense_run[qid]}
        d_metrics = evaluator.evaluate_run(d_res, qrels) 
        # Note: evaluate_run aggregates. But with 1 query, agg = single.
        
        # MS-MEQE Stats
        m_res = {qid: msmeqe_run[qid]}
        m_metrics = evaluator.evaluate_run(m_res, qrels)
        
        diff_map = m_metrics['map'] - d_metrics['map']
        
        results.append({
            'qid': qid,
            'dense_map': d_metrics['map'],
            'msmeqe_map': m_metrics['map'],
            'diff_map': diff_map,
            'dense_ndcg': d_metrics['ndcg_cut_10'],
            'msmeqe_ndcg': m_metrics['ndcg_cut_10'],
            'diff_ndcg': m_metrics['ndcg_cut_10'] - d_metrics['ndcg_cut_10']
        })
        
    df = pd.DataFrame(results)
    
    # Analysis 1: How many queries improved/hurt?
    improved = df[df['diff_map'] > 0.001]
    hurt = df[df['diff_map'] < -0.001]
    neutral = df[(df['diff_map'] >= -0.001) & (df['diff_map'] <= 0.001)]
    
    print("\n=== MAP Impact Analysis ===")
    print(f"Improved: {len(improved)} ({len(improved)/len(df):.1%}) - Avg Gain: {improved['diff_map'].mean():.4f}")
    print(f"Hurt:     {len(hurt)} ({len(hurt)/len(df):.1%}) - Avg Loss: {hurt['diff_map'].mean():.4f}")
    print(f"Neutral:  {len(neutral)} ({len(neutral)/len(df):.1%})")
    
    # Analysis 2: Did we hurt "Easy" queries?
    # Define "Easy" as Dense MAP > 0.5
    easy_queries = df[df['dense_map'] > 0.7]
    print(f"\n=== Impact on Easy Queries (Dense MAP > 0.7, N={len(easy_queries)}) ===")
    print(f"Avg MAP Change: {easy_queries['diff_map'].mean():.4f}")
    print(f"Queries Hurt: {len(easy_queries[easy_queries['diff_map'] < -0.001])} / {len(easy_queries)}")
    
    # Analysis 3: Did we help "Hard" queries?
    hard_queries = df[df['dense_map'] < 0.3]
    print(f"\n=== Impact on Hard Queries (Dense MAP < 0.3, N={len(hard_queries)}) ===")
    print(f"Avg MAP Change: {hard_queries['diff_map'].mean():.4f}")
    print(f"Queries Improved: {len(hard_queries[hard_queries['diff_map'] > 0.001])} / {len(hard_queries)}")

    # Analysis 4: Is there a massive disaster?
    disasters = df.sort_values('diff_map', ascending=True).head(5)
    print("\n=== Top 5 Disasters (Worst MAP Drops) ===")
    print(disasters[['qid', 'dense_map', 'msmeqe_map', 'diff_map']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', required=True)
    parser.add_argument('--msmeqe', required=True)
    parser.add_argument('--qrels', required=True)
    args = parser.parse_args()
    
    analyze_runs(args.dense, args.msmeqe, args.qrels)

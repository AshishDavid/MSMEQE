
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root
sys.path.append(".")
from src.utils.file_utils import load_trec_run, load_qrels

def calculate_ap(retrieved_docs, relevant_docs):
    if not relevant_docs: return 0.0
    num_rel = 0
    sum_prec = 0.0
    for i, (docid, score) in enumerate(retrieved_docs):
        if docid in relevant_docs:
            num_rel += 1
            sum_prec += num_rel / (i + 1)
    return sum_prec / len(relevant_docs)

def main():
    qrels_path = "data/test_qrels.txt"
    if not Path(qrels_path).exists():
        qrels_path = "msmeqe/data/test_qrels.txt" # Fallback
    
    # Best Reproduced Run
    best_run_path = "best_result_reproduction/runs/reproduced_utility_alpha_0.0.txt"
    # Oracle labels
    oracle_labels_path = "results/oracle_labels_multi.jsonl"
    
    print(f"Loading qrels from {qrels_path}...")
    qrels = load_qrels(qrels_path)
    print(f"Loading model run from {best_run_path}...")
    model_run = load_trec_run(best_run_path)
    
    print(f"Loading oracle potential from {oracle_labels_path}...")
    qid_to_oracle = {}
    with open(oracle_labels_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid_to_oracle[data['qid']] = data

    results = []
    
    for qid in qid_to_oracle:
        if qid not in model_run: continue
        
        ap_dense = qid_to_oracle[qid]['ap_dense']
        ap_max = qid_to_oracle[qid]['max_ap_cand']
        
        # Calculate AP for the current model decision
        # The model run might already be the dense run if it chose not to expand
        ap_model = calculate_ap(model_run[qid], qrels[qid])
        
        # Perfect Decision AP
        ap_perfect = max(ap_dense, ap_max)
        
        gap = ap_perfect - ap_model
        
        results.append({
            'qid': qid,
            'ap_dense': ap_dense,
            'ap_max': ap_max,
            'ap_model': ap_model,
            'ap_perfect': ap_perfect,
            'gap': gap,
            'error_type': 'missed_gain' if ap_max > ap_dense and ap_model <= ap_dense else ('harmful_expansion' if ap_model < ap_dense else 'correct')
        })

    # Stats
    gaps = [r['gap'] for r in results]
    print(f"\nOverall Mean Gap: {np.mean(gaps):.4f}")
    
    error_counts = defaultdict(int)
    for r in results:
        error_counts[r['error_type']] += 1
        
    print("\nError Distribution:")
    for et, count in error_counts.items():
        print(f"  {et}: {count} ({count/len(results)*100:.1f}%)")
        
    # Top 10 Missed Gains
    missed = sorted([r for r in results if r['error_type'] == 'missed_gain'], key=lambda x: x['gap'], reverse=True)
    print("\nTop 10 Missed Gains (False Negatives):")
    for r in missed[:10]:
        print(f"  QID {r['qid']}: Gap {r['gap']:.4f} (Max {r['ap_max']:.4f} vs Dense {r['ap_dense']:.4f}, Model chose {r['ap_model']:.4f})")

    # Top 10 Harmful Expansions
    harmful = sorted([r for r in results if r['error_type'] == 'harmful_expansion'], key=lambda x: x['ap_dense'] - x['ap_model'], reverse=True)
    print("\nTop 10 Harmful Expansions (False Positives):")
    for r in harmful[:10]:
        print(f"  QID {r['qid']}: Loss {r['ap_dense'] - r['ap_model']:.4f} (Dense {r['ap_dense']:.4f} -> Model {r['ap_model']:.4f})")

if __name__ == "__main__":
    main()


import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(".")
from src.utils.file_utils import load_trec_run, load_qrels

def calculate_ap(retrieved_docs, relevant_docs):
    """Calculate Average Precision for a single query."""
    if not relevant_docs:
        return 0.0
    
    # Sort docs by score
    sorted_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
    
    num_rel = 0
    sum_prec = 0.0
    for i, (docid, score) in enumerate(sorted_docs):
        if docid in relevant_docs:
            num_rel += 1
            sum_prec += num_rel / (i + 1)
            
    return sum_prec / len(relevant_docs)

def main():
    if len(sys.argv) < 5:
        print("Usage: python compare_runs_for_labels.py <qrels> <dense_run> <output_jsonl> <run1> <run2> ...")
        sys.exit(1)
        
    qrels_path = sys.argv[1]
    dense_run_path = sys.argv[2]
    output_path = sys.argv[3]
    candidate_runs_paths = sys.argv[4:]
    
    print("Loading qrels...")
    qrels = load_qrels(qrels_path)
    print("Loading dense baseline...")
    dense_run = load_trec_run(dense_run_path)
    
    candidate_runs = []
    for path in candidate_runs_paths:
        print(f"Loading candidate run: {path}")
        candidate_runs.append(load_trec_run(path))
        
    common_qids = set(dense_run.keys()) & set(qrels.keys())
    for run in candidate_runs:
        common_qids &= set(run.keys())
        
    print(f"Processing {len(common_qids)} queries...")
    
    labels_count = 0
    with open(output_path, 'w') as f:
        for qid in common_qids:
            # AP of dense baseline
            ap_dense = calculate_ap(dense_run[qid], qrels[qid])
            
            # Max AP among candidates
            max_ap_cand = -1.0
            best_run_idx = -1
            
            for i, run in enumerate(candidate_runs):
                ap = calculate_ap(run[qid], qrels[qid])
                if ap > max_ap_cand:
                    max_ap_cand = ap
                    best_run_idx = i
            
            # Label 1 if BEST expansion is strictly better than baseline
            # A margin can be added here
            margin = 0.001
            label = 1 if max_ap_cand > (ap_dense + margin) else 0
            
            if label == 1:
                labels_count += 1
                
            f.write(json.dumps({
                "qid": qid,
                "label": label,
                "ap_dense": ap_dense,
                "max_ap_cand": max_ap_cand,
                "best_run": candidate_runs_paths[best_run_idx] if best_run_idx >= 0 else "none"
            }) + "\n")
            
    print(f"Total labeled 1: {labels_count} / {len(common_qids)}")

if __name__ == "__main__":
    main()

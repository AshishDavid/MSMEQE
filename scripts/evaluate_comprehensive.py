
import sys
import json
import numpy as np
from collections import defaultdict

def load_qrels(file_path):
    qrels = defaultdict(dict)
    with open(file_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels[qid][docid] = int(rel)
    return qrels

def load_run(file_path):
    run = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            qid, _, docid, rank, score, _ = parts
            run[qid].append((docid, float(score)))
    
    # Sort by score
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run

def calculate_metrics(qrels, run, k_ndcg=10, k_recall=1000):
    all_ap = []
    all_ndcg = []
    all_recall = []
    all_mrr = []
    all_p1 = []
    all_p5 = []
    all_p10 = []
    all_recall_100 = []
    
    common_qids = set(qrels.keys()) & set(run.keys())
    
    for qid in common_qids:
        rel_set = {d for d, r in qrels[qid].items() if r > 0}
        if not rel_set: continue
        
        retrieved = run[qid]
        
        # 1. AP
        ap = 0.0
        num_rel = 0
        for i, (docid, _) in enumerate(retrieved):
            if docid in rel_set:
                num_rel += 1
                ap += num_rel / (i + 1)
        all_ap.append(ap / len(rel_set) if len(rel_set) > 0 else 0.0)
        
        # 2. nDCG@k
        dcg = 0.0
        for i, (docid, _) in enumerate(retrieved[:k_ndcg]):
            if docid in rel_set:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(rel_set), k_ndcg)):
            idcg += 1.0 / np.log2(i + 2)
            
        all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        
        # 3. Recall@k
        top_k_1000 = {d for d, _ in retrieved[:1000]}
        found_1000 = len(rel_set & top_k_1000)
        all_recall.append(found_1000 / len(rel_set) if len(rel_set) > 0 else 0.0)

        top_k_100 = {d for d, _ in retrieved[:100]}
        found_100 = len(rel_set & top_k_100)
        all_recall_100.append(found_100 / len(rel_set) if len(rel_set) > 0 else 0.0)

        # 4. MRR@10
        mrr = 0.0
        for i, (docid, _) in enumerate(retrieved[:10]):
            if docid in rel_set:
                mrr = 1.0 / (i + 1)
                break
        all_mrr.append(mrr)

        # 5. Precision@k
        p1 = 1.0 if len(retrieved) > 0 and retrieved[0][0] in rel_set else 0.0
        
        p5_rel = 0
        for docid, _ in retrieved[:5]:
            if docid in rel_set:
                p5_rel += 1
        p5 = p5_rel / 5.0
        
        p10_rel = 0
        for docid, _ in retrieved[:10]:
            if docid in rel_set:
                p10_rel += 1
        p10 = p10_rel / 10.0
        
        all_p1.append(p1)
        all_p5.append(p5)
        all_p10.append(p10)
        
    return {
        'MAP': float(np.mean(all_ap)),
        f'nDCG@{k_ndcg}': float(np.mean(all_ndcg)),
        'Recall@100': float(np.mean(all_recall_100)),
        'Recall@1000': float(np.mean(all_recall)),
        'MRR@10': float(np.mean(all_mrr)),
        'P@1': float(np.mean(all_p1)),
        'P@5': float(np.mean(all_p5)),
        'P@10': float(np.mean(all_p10)),
        'Queries': len(all_ap)
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_comprehensive.py <qrels> <run>")
        sys.exit(1)
        
    qrels = load_qrels(sys.argv[1])
    run = load_run(sys.argv[2])
    
    metrics = calculate_metrics(qrels, run)
    print(json.dumps(metrics, indent=2))

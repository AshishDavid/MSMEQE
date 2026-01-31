
from src.utils.file_utils import load_trec_run, load_qrels
from src.retrieval.evaluator import TRECEvaluator
import sys

run_path = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/runs/rm3_baseline.txt"
qrels_path = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/data/test_qrels.txt"

print("Loading qrels...")
qrels = load_qrels(qrels_path)
print(f"Loaded {len(qrels)} qrels")

print("Loading run (partial)...")
run = {}
count = 0
with open(run_path, 'r') as f:
    for line in f:
        parts = line.split()
        qid = parts[0]
        docid = parts[2]
        score = float(parts[4])
        if qid not in run:
            run[qid] = []
        run[qid].append((docid, score))
        if len(run) > 100 and qid not in run: # Stop after loading >100 queries completely
             break
        count += 1
        
print(f"Loaded run for {len(run)} queries")

evaluator = TRECEvaluator(metrics=['map', 'recall_1000'])
print("Evaluating...")
metrics = evaluator.evaluate_run(run, qrels)
print(metrics)

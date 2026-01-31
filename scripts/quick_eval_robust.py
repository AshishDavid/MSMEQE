
from src.utils.file_utils import load_trec_run, load_qrels
from src.retrieval.evaluator import TRECEvaluator

run_file = "runs/msmeqe.robust04.txt"
qrels_file = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_qrels.tsv"

print(f"Loading run: {run_file}")
run = load_trec_run(run_file)
print(f"Loaded {len(run)} queries in run")

print(f"Loading qrels: {qrels_file}")
qrels = load_qrels(qrels_file)

evaluator = TRECEvaluator(metrics=['map', 'mrr', 'ndcg_cut_10', 'ndcg_cut_20', 'P_20', 'recall_1000'])
metrics = evaluator.evaluate_run(run, qrels)
print("FINAL METRICS:", metrics)

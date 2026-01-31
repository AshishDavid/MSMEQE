
import sys
import pytrec_eval
import json

def calculate(qrels_path, run_path):
    with open(qrels_path, 'r') as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)
    
    with open(run_path, 'r') as f_run:
        run_dict = pytrec_eval.parse_run(f_run)
        
    metrics = {'map', 'ndcg_cut_10', 'ndcg_cut_20', 'recip_rank', 'P_20'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, metrics)
    results = evaluator.evaluate(run_dict)
    
    aggregated = {}
    for metric in metrics:
        values = [res[metric] for res in results.values()]
        aggregated[metric] = pytrec_eval.compute_aggregated_measure(metric, values)
        
    print(json.dumps(aggregated, indent=2))

if __name__ == "__main__":
    calculate(sys.argv[1], sys.argv[2])

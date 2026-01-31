import argparse
import pytrec_eval
import json
import statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-file", required=True, help="Path to TREC run file")
    parser.add_argument("--qrels", required=True, help="Path to QRELS file")
    args = parser.parse_args()

    # Load Qrels
    with open(args.qrels, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    # Load Run
    with open(args.run_file, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    # Convert keys to strings to ensure matching (though parse functions handles some)
    # pytrec_eval expects qids to ensure overlap

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg_cut_10', 'recall_1000'})
    results = evaluator.evaluate(run)

    # Aggregate
    agg_metrics = {
        'map': [],
        'ndcg_cut_10': [],
        'recall_1000': []
    }

    for qid, metrics in results.items():
        for m in agg_metrics:
            agg_metrics[m].append(metrics[m])

    print("="*40)
    print(f"FINAL RESULTS (Mean of {len(agg_metrics['map'])} queries)")
    print("="*40)
    for m in agg_metrics:
        mean_val = statistics.mean(agg_metrics[m])
        print(f"{m:<15}: {mean_val:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()

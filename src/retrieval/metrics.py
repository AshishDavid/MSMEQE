import pytrec_eval


def get_metric(qrels: str, run: str, metric: str = 'map') -> float:

    # Read the qrel file
    with open(qrels, 'r') as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)

    # Read the run file
    with open(run, 'r') as f_run:
        run_dict = pytrec_eval.parse_run(f_run)

    # Evaluate
    # We only need the specific metric requested, but pytrec_eval requires a list of measures
    # Passing [metric] to RelevanceEvaluator is more efficient if possible, but 
    # supported_measures is usually fine for the evaluate() call itself.
    # The crash happened during compute_aggregated_measure for obscure metrics.
    
    # Optimization: Only init evaluator with the metric we care about
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, {metric})
    results = evaluator.evaluate(run_dict)
    
    # Compute aggregated measure just for the requested metric
    # The results structure is {query_id: {metric: value}}
    values = [query_meas[metric] for query_meas in results.values()]
    
    return pytrec_eval.compute_aggregated_measure(metric, values)



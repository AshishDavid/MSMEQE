#!/usr/bin/env python3
# scripts/run_rq1_heterogeneous.py
# RQ1 Experiment 1.3: Heterogeneous Resource Mapping
# Force w = log(DF). Common terms are expensive. Rare terms are cheap.
# Metric: Recall@1000 (Does Knapsack pick more diversity?)

import logging
import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.retrieval.evaluator import TRECEvaluator
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    write_trec_run_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HeterogeneousWeightModel:
    def predict(self, X):
        # We don't use X. We use the candidate DFs which are passed during expand/predict.
        # But predict() signature usually just takes X.
        # The MSMEQE logic calls predict(X, candidates).
        # We need to hack the MSMEQEExpansionModel to pass candidates to a custom method or handle it there.
        # Easier: Implement a custom model that ignores X and assumes candidates are available or handled by the caller.
        # Actually, in MSMEQEExpansionModel._predict_weights:
        # w_preds = self.weight_model.predict(X_weight)
        # It doesn't pass candidates to .predict().
        # So we need to subclass MSMEQEExpansionModel and override _predict_weights.
        return np.zeros(X.shape[0])

class HeterogeneousMSMEQEModel(MSMEQEExpansionModel):
    def _predict_weights(self, X, candidates):
        # w = log(DF). Clip min to 1 to avoid zeros?
        # DF can be large. log10(1,000,000) = 6. log10(100) = 2.
        # This gives a nice range.
        weights = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            df = max(1, c.df)
            weights[i] = np.log10(df)
        return weights

class GreedyHeterogeneousModel(HeterogeneousMSMEQEModel):
    def _solve_selection(self, values, weights, budget, term_embeddings=None):
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        remaining = float(budget)
        indices = np.argsort(-values)
        for idx in indices:
            w = weights[idx]
            v = values[idx]
            if v > 0 and w <= remaining:
                counts[idx] = 1
                remaining -= w
        return counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--value-model", type=str, required=True)
    parser.add_argument("--lucene-path", type=str, required=True)
    parser.add_argument("--budget", type=float, default=20.0, help="Budget for log(DF) weights. E.g. 20 means ~3-5 common words or ~10 rare words.")
    parser.add_argument("--output-prefix", type=str, default="runs/rq1_hetero/run")
    args = parser.parse_args()

    initialize_lucene(args.lucene_path)
    
    queries = load_queries_from_file(args.topics)
    qrels = load_qrels_from_file(args.qrels)
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    value_model = joblib.load(args.value_model)
    weight_model_dummy = None # integrated in subclass
    
    candidate_extractor = MultiSourceCandidateExtractor(
        args.index, encoder, n_docs_rm3=40, n_pseudo_docs=20
    )
    dense_retriever = DenseRetriever(args.index, encoder)
    evaluator = TRECEvaluator(metrics=['recall_cut_1000', 'map']) # Focus on recall
    
    print(f"{'Method':<10} | {'Recall@1k':<12} | {'MAP':<10} | {'Avg Terms':<10}")
    print("-" * 50)
    
    # Run Greedy
    greedy = GreedyHeterogeneousModel(
        encoder, value_model, weight_model_dummy, None, 8841823, min_budget=args.budget, max_budget=args.budget, budget_step=1
    )
    greedy._predict_budget = lambda x: args.budget
    
    gp = MSMEQEEvaluationPipeline(encoder, candidate_extractor, greedy, dense_retriever, evaluator)
    res_g = gp.run_evaluation(queries, qrels, topk=1000)
    
    # Run Knapsack
    knapsack = HeterogeneousMSMEQEModel(
        encoder, value_model, weight_model_dummy, None, 8841823, min_budget=args.budget, max_budget=args.budget, budget_step=1
    )
    knapsack._predict_budget = lambda x: args.budget
    
    kp = MSMEQEEvaluationPipeline(encoder, candidate_extractor, knapsack, dense_retriever, evaluator)
    res_k = kp.run_evaluation(queries, qrels, topk=1000)
    
    # Report
    rec_g = res_g['metrics']['recall_cut_1000']
    map_g = res_g['metrics']['map']
    
    rec_k = res_k['metrics']['recall_cut_1000']
    map_k = res_k['metrics']['map']
    
    print(f"{'Greedy':<10} | {rec_g:.4f}       | {map_g:.4f}     | -")
    print(f"{'Knapsack':<10} | {rec_k:.4f}       | {map_k:.4f}     | -")
    
    write_trec_run_file(res_g['run_results'], f"{args.output_prefix}_greedy.txt", "GreedyHetero")
    write_trec_run_file(res_k['run_results'], f"{args.output_prefix}_knapsack.txt", "KnapsackHetero")

if __name__ == "__main__":
    main()

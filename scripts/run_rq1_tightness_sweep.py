#!/usr/bin/env python3
# scripts/run_rq1_tightness_sweep.py
# RQ1 Experiment 1.1: Tightness Sweep
# Compare Greedy vs Knapsack at W in {1,2,3,5,8,13,21} with normalized weights.

import logging
import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

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

class NormalizedWeightModel:
    def __init__(self, original_model):
        self.original_model = original_model
    
    def predict(self, X):
        w = self.original_model.predict(X)
        # Normalize weights to mean=1.0 per batch?
        # Or globally? The protocol says "average weight of a candidate term is exactly 1.0".
        # Doing it per-query is safer to ensure consistency across queries.
        if len(w) > 0:
            mean = np.mean(w)
            if mean > 1e-9:
                w = w / mean
            else:
                w = np.ones_like(w)
        return w

class GreedyNormalizedModel(MSMEQEExpansionModel):
    def _solve_selection(self, values, weights, budget, term_embeddings=None):
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        remaining = float(budget)
        
        # Sort by Value (ignoring weight/cost)
        indices = np.argsort(-values)
        
        for idx in indices:
            w = weights[idx]
            v = values[idx]
            if v > 0 and w <= remaining:
                counts[idx] = 1
                remaining -= w
        return counts

class KnapsackNormalizedModel(MSMEQEExpansionModel):
    # Uses standard solver, which optimizes Value subject to Budget constraint
    # But explicitly handles the float weights from the normalized model
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--value-model", type=str, required=True)
    parser.add_argument("--weight-model", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="runs/rq1_sweep/run")
    parser.add_argument("--lucene-path", type=str, required=True)
    args = parser.parse_args()

    initialize_lucene(args.lucene_path)
    
    queries = load_queries_from_file(args.topics)
    qrels = load_qrels_from_file(args.qrels)
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    value_model_base = joblib.load(args.value_model)
    weight_model_base = joblib.load(args.weight_model)
    norm_weight_model = NormalizedWeightModel(weight_model_base)
    
    # Pre-Extract Candidates (Optimization)
    candidate_extractor = MultiSourceCandidateExtractor(
        args.index, encoder, n_docs_rm3=40, n_pseudo_docs=20
    )
    # Note: Extracting all candidates upfront would be faster but memory intensive.
    # We'll rely on pipeline processing per query.
    
    dense_retriever = DenseRetriever(args.index, encoder)
    evaluator = TRECEvaluator(metrics=['ndcg_cut_10'])
    
    budgets = [1, 2, 3, 5, 8, 13, 21]
    results_table = []

    print(f"{'Budget':<6} | {'Greedy nDCG':<12} | {'Knapsack nDCG':<15} | {'Gap':<8}")
    print("-" * 50)
    
    for W in budgets:
        # 1. Run Greedy
        greedy_model = GreedyNormalizedModel(
            encoder=encoder,
            value_model=value_model_base,
            weight_model=norm_weight_model,
            budget_model=None, # Fixed W
            collection_size=8841823,
            min_budget=W, max_budget=W, budget_step=1 # Force budget
        )
        # Override budget prediction to always return W
        greedy_model._predict_budget = lambda x: W
        
        greedy_pipeline = MSMEQEEvaluationPipeline(
            encoder, candidate_extractor, greedy_model, dense_retriever, evaluator
        )
        # Suppress extensive logs
        res_g = greedy_pipeline.run_evaluation(queries, qrels, topk=1000)
        ndcg_g = res_g['metrics']['ndcg_cut_10']
        
        # 2. Run Knapsack
        knap_model = KnapsackNormalizedModel(
            encoder=encoder,
            value_model=value_model_base,
            weight_model=norm_weight_model,
            budget_model=None,
            collection_size=8841823,
            min_budget=W, max_budget=W, budget_step=1
        )
        knap_model._predict_budget = lambda x: W
        
        knap_pipeline = MSMEQEEvaluationPipeline(
            encoder, candidate_extractor, knap_model, dense_retriever, evaluator
        )
        res_k = knap_pipeline.run_evaluation(queries, qrels, topk=1000)
        ndcg_k = res_k['metrics']['ndcg_cut_10']
        
        gap = ndcg_k - ndcg_g
        print(f"{W:<6} | {ndcg_g:.4f}       | {ndcg_k:.4f}          | {gap:.4f}")
        results_table.append({'W': W, 'Greedy': ndcg_g, 'Knapsack': ndcg_k, 'Gap': gap})
        
        # Save Runs
        write_trec_run_file(res_g['run_results'], f"{args.output_prefix}_greedy_{W}.txt", f"Greedy_{W}")
        write_trec_run_file(res_k['run_results'], f"{args.output_prefix}_knapsack_{W}.txt", f"Knapsack_{W}")

if __name__ == "__main__":
    main()

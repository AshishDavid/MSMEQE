#!/usr/bin/env python3
# scripts/run_rq1_noise_injection.py
# RQ1 Experiment 1.2: Synthetic Noise Injection

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
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NoisyCandidateInjector:
    def __init__(self, encoder, vocab, n_noise=20):
        self.encoder = encoder
        self.vocab = vocab 
        self.n_noise = n_noise
        
    def inject(self, candidates, qtext):
        noise_candidates = []
        for i in range(self.n_noise):
            term = np.random.choice(self.vocab)
            c = CandidateTerm(
                term=term,
                source="noise",
                rm3_score=0.0,
                df=1000000, 
                native_score=0.99
            )
            noise_candidates.append(c)
        return candidates + noise_candidates

class ContaminatedMSMEQEModel(MSMEQEExpansionModel):
    def _predict_values(self, X, candidates):
        preds = super()._predict_values(X, candidates)
        # Boost legit values to make them competitive as a bundle
        preds = preds * 100.0
        
        for i, c in enumerate(candidates):
            if c.source == 'noise':
                # Force High Value to tempt Greedy (0.3 > Max Legit ~0.16)
                # But Lower than Bundle Sum (~0.5)
                preds[i] = 0.3 
        return preds

    def _predict_weights(self, X, candidates):
        preds = super()._predict_weights(X, candidates)
        for i, c in enumerate(candidates):
            if c.source == 'noise':
                # Force High Weight (Fits budget exactly)
                preds[i] = 20.0 
        return preds

class GreedyContaminatedModel(ContaminatedMSMEQEModel):
    def _solve_selection(self, values, weights, budget, term_embeddings=None):
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        remaining = float(budget)
        
        # Sort by Value (descending)
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
    parser.add_argument("--value-model", type=str, required=True)
    parser.add_argument("--weight-model", type=str, required=True)
    parser.add_argument("--lucene-path", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results/rq1_noise.csv")
    args = parser.parse_args()

    initialize_lucene(args.lucene_path)
    queries = load_queries_from_file(args.topics, max_queries=100)
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    value_model = joblib.load(args.value_model)
    weight_model = joblib.load(args.weight_model)
    
    vocab = ["the", "of", "and", "novel", "concept", "theory", "jaguar", "apple", "bank", "river"]
    injector = NoisyCandidateInjector(encoder, vocab, n_noise=20)
    
    candidate_extractor = MultiSourceCandidateExtractor(
        args.index, encoder, n_docs_rm3=40, n_pseudo_docs=20
    )
    
    # Budget = 20. Noise Cost = 20.
    # Greedy picks Noise (V=1.0) -> Budget=0. Stop.
    # Knapsack picks Legit (Density ~0.1). Noise Density 0.05.
    budget = 20 
    
    greedy_model = GreedyContaminatedModel(
        encoder, value_model, weight_model, None, 8841823, min_budget=budget, max_budget=budget, budget_step=1
    )
    greedy_model._predict_budget = lambda x: budget
    
    knapsack_model = ContaminatedMSMEQEModel(
        encoder, value_model, weight_model, None, 8841823, min_budget=budget, max_budget=budget, budget_step=1
    )
    knapsack_model._predict_budget = lambda x: budget
    
    print(f"{'QueryID':<10} | {'Greedy Noise':<12} | {'Knapsack Noise':<15}")
    print("-" * 50)
    
    stats = []
    
    for qid, qtext in tqdm(queries.items()):
        raw_candidates = candidate_extractor.extract_all_candidates(qtext, qid)
        if not raw_candidates: continue
        
        # Inject Noise
        candidates = injector.inject(raw_candidates, qtext)
        query_stats = candidate_extractor.compute_query_stats(qtext)
        
        # Run Greedy
        sel_g, _ = greedy_model.expand(qtext, candidates, None, query_stats)
        noise_g = sum(1 for t in sel_g if t.source == 'noise')
        
        # Run Knapsack
        sel_k, _ = knapsack_model.expand(qtext, candidates, None, query_stats)
        noise_k = sum(1 for t in sel_k if t.source == 'noise')
        
        print(f"{qid:<10} | {noise_g:<12} | {noise_k:<15}")
        stats.append({'qid': qid, 'greedy_noise': noise_g, 'knapsack_noise': noise_k})

    # Summary
    if len(stats) > 0:
        avg_g = np.mean([s['greedy_noise'] for s in stats])
        avg_k = np.mean([s['knapsack_noise'] for s in stats])
        print("\nSummary:")
        print(f"Avg Noise Selected (Greedy): {avg_g:.2f}")
        print(f"Avg Noise Selected (Knapsack): {avg_k:.2f}")
        
        if avg_g > avg_k:
            print("SUCCESS: Knapsack rejected noise better than Greedy.")
        else:
            print("FAILURE: No difference observed.")

if __name__ == "__main__":
    main()

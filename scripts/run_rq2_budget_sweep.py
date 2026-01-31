#!/usr/bin/env python3
# scripts/run_rq2_budget_sweep.py
# RQ2: Safety of Aggression - Budget/Depth Sweep
# Compares RM3 (blind expansion) vs MS-MEQE (risk-aware) at depths k=[10, 30, 50, 70]

import logging
import argparse
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.retrieval.evaluator import TRECEvaluator

# Import helpers from main eval script
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    load_kb_candidates_from_file,
    write_trec_run_file,
    DenseRetriever
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models for Ablation
# ---------------------------------------------------------------------------

class FixedBudgetModel:
    def __init__(self, budget):
        self.budget = float(budget)
    def predict(self, X):
        return np.array([self.budget] * X.shape[0])

class RM3BaselineModel(MSMEQEExpansionModel):
    """
    Simulates standard RM3 (or any top-k) behavior within the MS-MEQE pipeline
    by setting all Values=1, Weights=1, and selecting exactly Budget terms.
    """
    def _predict_values(self, X, candidates):
        # All terms have equal value (just rely on initial candidate ranking/score)
        # But wait, MS-MEQE candidates are multi-source. 
        # For a fair "RM3 Baseline", we should only select from 'docs' source 
        # and use their RM3 score as value.
        m = len(candidates)
        vals = np.zeros(m)
        for i, c in enumerate(candidates):
            if c.source == 'docs':
                vals[i] = c.rm3_score # Use standard RM3 score
            else:
                vals[i] = -1.0 # Ignore non-RM3 terms
        return vals

    def _predict_weights(self, X, candidates):
        # Unit cost for all terms
        return np.ones(len(candidates))

    def _solve_knapsack(self, values, weights, budget, term_embeddings=None):
        # Implementation of Standard Top-K
        # Sort by Value (RM3 score) desc
        # Select Top K (where K = budget)
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        
        # Filter negative values (non-RM3 terms)
        valid_indices = [i for i in range(m) if values[i] > 0]
        # Sort valid by RM3 score
        sorted_valid = sorted(valid_indices, key=lambda i: values[i], reverse=True)
        
        # Take Top K
        k = int(budget)
        for idx in sorted_valid[:k]:
            counts[idx] = 1
            
        return counts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ2: Budget Sweep")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=['msmeqe', 'rm3'], required=True)
    parser.add_argument("--budget", type=int, required=True, help="Fixed budget/depth K")
    parser.add_argument("--value-model", type=str, default=None)
    parser.add_argument("--weight-model", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lucene-path", type=str, required=True)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    initialize_lucene(args.lucene_path)
    
    queries = load_queries_from_file(args.topics)
    qrels = load_qrels_from_file(args.qrels)
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    # Setup Models
    value_model = None
    weight_model = None
    budget_model = FixedBudgetModel(args.budget)
    
    if args.model_type == 'msmeqe':
        value_model = joblib.load(args.value_model)
        weight_model = joblib.load(args.weight_model)
        ModelClass = MSMEQEExpansionModel
    else:
        # RM3 Baseline: dummy models, logic overrides them
        value_model = "dummy" 
        weight_model = "dummy"
        ModelClass = RM3BaselineModel
        
    msmeqe_model = ModelClass(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823,
        lambda_interp=0.3 # Keep interpolation constant
    )
    
    candidate_extractor = MultiSourceCandidateExtractor(
        args.index, encoder, n_docs_rm3=100, n_pseudo_docs=20 # Fetch enough for depth 70
    )
    
    dense_retriever = DenseRetriever(args.index, encoder)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10'])
    
    # Simplified pipeline run
    run_results = {}
    for qid, qtext in tqdm(queries.items(), desc=f"{args.model_type.upper()}@{args.budget}"):
        try:
           candidates = candidate_extractor.extract_all_candidates(qtext, qid)
           query_stats = candidate_extractor.compute_query_stats(qtext)
           # Force budget
           # Note: MSMEQE internal logic might clip or step budget. 
           # We rely on FixedBudgetModel passing through the exact value/weights relative.
           selected_terms, q_star = msmeqe_model.expand(
               qtext, candidates, None, query_stats
           )
           
           results = dense_retriever.retrieve(q_star, k=1000)
           run_results[qid] = results
        except Exception as e:
            logger.error(f"Error {qid}: {e}")
            
    metrics = evaluator.evaluate_run(run_results, qrels)
    
    logger.info(f"RESULTS {args.model_type} @ {args.budget}")
    for k,v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    write_trec_run_file(run_results, args.output, f"{args.model_type}_{args.budget}")

if __name__ == "__main__":
    main()

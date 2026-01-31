#!/usr/bin/env python3
# scripts/run_rq1_ablation.py
# RQ1: Optimization (Knapsack) vs Heuristics (Greedy)
# Based on scripts/run_msmeqe_evaluation.py

import logging
import argparse
import sys
import json
import time
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.retrieval.evaluator import TRECEvaluator
from src.models.budget_predictor import BudgetPredictor

# Import existing pipeline components
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    load_kb_candidates_from_file,
    write_trec_run_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Greedy Model Variant
# ---------------------------------------------------------------------------

class GreedyMSMEQEExpansionModel(MSMEQEExpansionModel):
    """
    Greedy Baseline for RQ1.
    
    Selection Logic:
      - Rank candidates purely by predicted Value (v).
      - Ignore Risk (w) for ranking.
      - Select top terms that fit within the budget (using weights).
      - No repetition (0/1).
    """
    
    def _solve_knapsack(
            self,
            values: np.ndarray,
            weights: np.ndarray,
            budget: float,
            term_embeddings: np.ndarray = None,
    ) -> np.ndarray:
        m = len(values)
        if m == 0 or budget <= 0:
            return np.zeros((m,), dtype=np.int32)
            
        counts = np.zeros(m, dtype=np.int32)
        remaining_budget = float(budget)
        
        # Greedy Sort: Descending order of Value
        sorted_indices = np.argsort(-values)
        
        for idx in sorted_indices:
            w = weights[idx]
            v = values[idx]
            
            # Simple rule: if it fits, take it.
            # Only if Value is positive
            if v > 0 and w <= remaining_budget:
                counts[idx] = 1
                remaining_budget -= w
        
        # Debug Log for first few queries
        if np.sum(counts) > 0 and np.random.rand() < 0.05:
            logger.info(f"GREEDY SELECTED {np.sum(counts)} terms. Budget Left: {remaining_budget}")
            
        return counts

class FixedBudgetModel70:
    """Forces budget to 70 for controlled comparison."""
    def predict(self, X):
        # Return 70 for all rows
        return np.array([70] * X.shape[0])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ1 Ablation: Knapsack vs Greedy")
    
    # Re-use standard arguments 
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--selection-method", type=str, choices=['knapsack', 'greedy'], required=True,
                        help="Selection logic to use")
    
    # Models
    parser.add_argument("--value-model", type=str, required=True)
    parser.add_argument("--weight-model", type=str, required=True)
    parser.add_argument("--budget-model", type=str, required=True) # Will be ignored/overridden
    
    # Components
    parser.add_argument("--sbert-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--kb-candidates", type=str, default=None)
    parser.add_argument("--lucene-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    logger.info(f"RQ1 ABLATION: {args.selection_method.upper()}")
    
    # Initialize
    initialize_lucene(args.lucene_path)
    queries = load_queries_from_file(args.topics)
    qrels = load_qrels_from_file(args.qrels)
    
    encoder = SemanticEncoder(model_name=args.sbert_model)
    
    # Load Models
    value_model = joblib.load(args.value_model)
    weight_model = joblib.load(args.weight_model)
    # FORCE FIXED BUDGET 70
    budget_model = FixedBudgetModel70()
    
    # Initialize Candidate Extractor (Standard)
    # Using defaults for RM3/KB/Emb params
    kb_candidates_map = None
    if args.kb_candidates:
        kb_candidates_map = load_kb_candidates_from_file(args.kb_candidates)
        
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index,
        encoder=encoder,
        n_docs_rm3=40,
        n_pseudo_docs=20,
    )
    
    # Initialize MS-MEQE Model
    model_class = MSMEQEExpansionModel
    if args.selection_method == 'greedy':
        model_class = GreedyMSMEQEExpansionModel
        
    msmeqe_model = model_class(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model, # Fixed 70
        collection_size=8841823,
        lambda_interp=0.3
    )
    
    dense_retriever = DenseRetriever(args.index, encoder)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'P_20'])
    
    pipeline = MSMEQEEvaluationPipeline(
        encoder=encoder,
        candidate_extractor=candidate_extractor,
        msmeqe_model=msmeqe_model,
        dense_retriever=dense_retriever,
        evaluator=evaluator,
        kb_candidates_map=kb_candidates_map
    )
    
    results = pipeline.run_evaluation(queries, qrels, topk=args.topk)
    
    write_trec_run_file(results['run_results'], args.output, f"RQ1_{args.selection_method}")
    
    logger.info("Metrics:")
    for k, v in results['metrics'].items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

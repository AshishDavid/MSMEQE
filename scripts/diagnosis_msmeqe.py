
import sys
import os
import argparse
import logging
import json
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm
from src.utils.lucene_utils import initialize_lucene



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_query(query_text, query_id, index_path, value_model_path, weight_model_path, budget_model_path, encoder_name):

    # Initialize components
    initialize_lucene(lucene_path="./lucene_jars")
    encoder = SemanticEncoder(model_name=encoder_name)
    feature_extractor = FeatureExtractor(collection_size=8841823)
    
    logger.info("Loading models...")
    value_model = joblib.load(value_model_path)
    weight_model = joblib.load(weight_model_path)
    try:
        budget_model = joblib.load(budget_model_path)
    except:
        from src.models.budget_predictor import ConstantModel
        budget_model = ConstantModel(50)
        logger.warning("Could not load budget model, using ConstantModel(50)")

    # Initialize Extractors
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=index_path,
        encoder=encoder,
        n_docs_rm3=20, 
        n_pseudo_docs=10
    )
    
    msmeqe_model = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823,
        lambda_interp=0.3
    )

    logger.info(f"--- DIAGNOSING QUERY: '{query_text}' (ID: {query_id}) ---")

    # 1. Extract Candidates
    candidates = candidate_extractor.extract_all_candidates(query_text, query_id=query_id)
    logger.info(f"Extracted {len(candidates)} candidates")
    
    # 2. Trace MS-MEQE Expansion
    q_emb = encoder.encode([query_text])[0]
    term_strings = [c.term for c in candidates]
    term_embs = encoder.encode(term_strings)
    
    pseudo_centroid = candidate_extractor.compute_pseudo_centroid(query_text)
    
    # Value Features & Preds
    X_val = msmeqe_model._build_value_features(query_text, q_emb, pseudo_centroid, candidates, term_embs)
    v_pred = msmeqe_model._predict_values(X_val, candidates)
    
    # Weight Features & Preds
    X_wt = msmeqe_model._build_weight_features(query_text, q_emb, candidates, term_embs)
    w_pred = msmeqe_model._predict_weights(X_wt, candidates)
    
    # Budget
    query_stats = candidate_extractor.compute_query_stats(query_text)
    budget = msmeqe_model._predict_budget(query_stats)
    logger.info(f"Predicted Budget: {budget}")
    logger.info(f"Query Stats: {json.dumps(query_stats, indent=2)}")

    # Knapsack
    counts = msmeqe_model._solve_unbounded_knapsack(v_pred, w_pred, budget)
    
    # Analysis
    logger.info("\n--- CANDIDATE ANALYSIS (Top 20 by RM3 Score) ---")
    
    # Sort by RM3 score to see if good RM3 terms are selected
    sorted_by_rm3 = sorted(zip(candidates, v_pred, w_pred, counts), key=lambda x: x[0].rm3_score, reverse=True)
    
    print(f"{'Term':<20} {'Source':<6} {'RM3':<6} {'Val (Pred)':<10} {'Wgt (Pred)':<10} {'Count':<5} {'Selected?'}")
    print("-" * 80)
    for c, v, w, cnt in sorted_by_rm3[:20]:
        sel = "YES" if cnt > 0 else "NO"
        print(f"{c.term[:18]:<20} {c.source:<6} {c.rm3_score:.4f} {v:.4f}     {w:.4f}     {cnt:<5} {sel}")

    # Inspect Selected Terms
    selected = [c for c, cnt in zip(candidates, counts) if cnt > 0]
    logger.info(f"\nTotal Selected Terms: {len(selected)}")
    
    # Check Lambda
    clarity = query_stats.get('clarity', 12.0)
    dyn_lambda = min(0.6, max(0.05, 0.3 * (clarity / 12.0)))
    logger.info(f"Dynamic Lambda: {dyn_lambda:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="what is the function of the ribosome")
    parser.add_argument("--qid", type=str, default="sample_q")
    args = parser.parse_args()
    
    diagnose_query(
        args.query, args.qid,
        "data/msmarco_index",
        "models/value_model.pkl",
        "models/weight_model.pkl",
        "models/budget_model.pkl",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

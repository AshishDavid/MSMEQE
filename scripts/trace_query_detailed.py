import argparse
import sys
import os
import json
import numpy as np
import logging
from pathlib import Path
import joblib
import torch

# Add src to path
sys.path.append(os.getcwd())

from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor, create_candidate_stats_dict
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.models.budget_predictor import BudgetPredictor
from src.utils.lucene_utils import initialize_lucene
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Query text to trace")
    parser.add_argument("--index", type=str, default="data/msmarco_index")
    parser.add_argument("--models-dir", type=str, default="models_llm")
    parser.add_argument("--llm-candidates", type=str, default="data/llm_candidates_test.jsonl")
    parser.add_argument("--lucene-path", type=str, default="lucene_jars/")
    parser.add_argument("--qid", type=str, default=None, help="Query ID for LLM lookup")
    args = parser.parse_args()

    # 1. Setup
    print("\n" + "="*80)
    print(f"TRACING PIPELINE FOR QUERY: '{args.query}' (ID: {args.qid})")
    print("="*80)
    
    initialize_lucene(args.lucene_path)
    
    print("\n[1] LOADING MODELS...")
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    feature_extractor = FeatureExtractor(
        collection_size=8841823,
    )
    
    cand_extractor = MultiSourceCandidateExtractor(
        index_path=args.index,
        encoder=encoder,
        llm_candidates_path=args.llm_candidates
    )
    
    # Load Models
    value_model = joblib.load(f"{args.models_dir}/value_model.pkl")
    weight_model = joblib.load(f"{args.models_dir}/weight_model.pkl")
    budget_model = joblib.load(f"{args.models_dir}/budget_model.pkl") 
    # Use predictor wrapper if needed, but direct load is fine if we use predict
    
    # 2. Candidate Extraction
    print("\n" + "="*80)
    print("[2] CANDIDATE EXTRACTION")
    print("="*80)
    candidates = cand_extractor.extract_all_candidates(args.query, query_id=args.qid)
    
    print(f"Found {len(candidates)} candidates from sources: {set(c.source for c in candidates)}")
    print("-" * 40)
    print(f"{'Term':<20} {'Source':<10} {'Native Score':<10}")
    print("-" * 40)
    for c in candidates[:10]: # Show top 10 raw
        print(f"{c.term:<20} {c.source:<10} {c.native_score:.4f}")
    if len(candidates) > 10: print("...")

    # 3. Feature Extraction & Scoring
    print("\n" + "="*80)
    print("[3] SCORING (Value vs. Weight)")
    print("="*80)
    
    query_emb = encoder.encode(args.query)
    
    # Budget Prediction
    # Need stats for budget
    # Construct a dummy stats dict or use a helper
    
    scored_candidates = []
    
    print(f"{'Term':<20} {'Value (Gain)':<15} {'Weight (Risk)':<15} {'Ratio (V/W)':<15}")
    print("-" * 70)
    
    features_list = []
    cands_for_pred = []
    
    for cand in candidates:
        # Hack: we don't have cached stats for single query easily without running the full pipeline logic
        # So we'll rely on FeatureExtractor doing the heavy lifting if we pass it the right args
        # But FeatureExtractor needs `candidate_stats`.
        # MultiSourceCandidateExtractor doesn't return global stats. 
        # We will approximate stats or simulate them.
        # Actually, let's just assume we can get them.
        
        # Simpler: use the actual MSMEQEExpander class which encapsulates this logic?
        # No, that might hide the details. I want to show the details.
        
        # Let's extract features manually
        # 1. Term embedding
        term_emb = encoder.encode(cand.term)
        
        # 2. Stats
        stats_dict = create_candidate_stats_dict(
            rm3_score=cand.rm3_score,
            tf_pseudo=cand.tf_pseudo,
            coverage_pseudo=cand.coverage_pseudo,
            df=cand.df,
            cf=cand.cf,
            native_rank=cand.native_rank,
            native_score=cand.native_score
        )
        
        # Value Features
        v_feats = feature_extractor.extract_value_features(
            candidate_term=cand.term,
            candidate_source=cand.source,
            candidate_stats=stats_dict,
            query_text=args.query,
            query_embedding=query_emb,
            term_embedding=term_emb
        )
        
        # Weight Features
        w_feats = feature_extractor.extract_weight_features(
            candidate_term=cand.term,
            candidate_source=cand.source,
            candidate_stats=stats_dict,
            query_text=args.query,
            query_embedding=query_emb,
            term_embedding=term_emb
        )
        
        # Predict immediately (single item prediction)
        v = value_model.predict(np.array([v_feats]))[0]
        w = weight_model.predict(np.array([w_feats]))[0]
        w = max(0.001, w)
        ratio = v / w
        
        scored_candidates.append({
            'term': cand.term,
            'source': cand.source,
            'value': v,
            'weight': w,
            'ratio': ratio,
            'native_score': cand.native_score
        })

    if not scored_candidates:
        print("No candidates processed.")
        return

    # Sort by Ratio
    scored_candidates.sort(key=lambda x: x['ratio'], reverse=True)
    
    for item in scored_candidates[:10]:
         print(f"{item['term']:<20} {item['value']:<15.4f} {item['weight']:<15.4f} {item['ratio']:<15.4f}")

    # 4. Budget & Selection
    print("\n" + "="*80)
    print("[4] SELECTION (Knapsack Optimization)")
    print("="*80)
    
    # Predict Budget
    # Compute query stats
    query_stats = cand_extractor.compute_query_stats(args.query)
    
    # Extract query features
    q_feats = feature_extractor.extract_query_features(args.query, query_stats)
    
    # Predict
    budget_pred = budget_model.predict([q_feats])
    budget = budget_pred[0]
    
    print(f"Predicted Risk Budget: {budget:.4f}")
    
    # Knapsack Greedy
    selected = []
    current_weight = 0.0
    total_value = 0.0
    
    print("-" * 60)
    print(f"{'Action':<10} {'Term':<15} {'Weight':<10} {'Budget Left':<10}")
    print("-" * 60)
    
    for item in scored_candidates:
        if item['term'] in [s['term'] for s in selected]:
            continue
            
        if item['value'] <= 0:
             continue # Don't select negative value terms
             
        if current_weight + item['weight'] <= budget:
            selected.append(item)
            current_weight += item['weight']
            total_value += item['value']
            print(f"{'TAKE':<10} {item['term']:<15} {item['weight']:<10.4f} {budget-current_weight:<10.4f}")
        else:
            print(f"{'SKIP':<10} {item['term']:<15} {item['weight']:<10.4f} (Over budget)")

    # 5. Final Query
    print("\n" + "="*80)
    print("[5] FINAL QUERY EXPANSION")
    print("="*80)
    print(f"Original: {args.query}")
    expansion_str = " ".join([f"{s['term']}^{s['value']:.2f}" for s in selected])
    print(f"Expansion: {expansion_str}")
    
    # Interpolation
    lambda_val = 0.1 # Using our tuned value
    print(f"\nConstructing Hybrid Embedding (Lambda={lambda_val})...")
    # q_final = (1-L)*q + L*centroid(expansion)
    
    print("\nDone.")

if __name__ == "__main__":
    main()

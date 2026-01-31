
import logging
import argparse
import sys
import numpy as np
import joblib
from collections import defaultdict
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from scripts.run_msmeqe_evaluation import load_queries_from_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load Models
    value_model = joblib.load("models_10k/value_model.pkl")
    weight_model = joblib.load("models_10k/weight_model.pkl")
    # Budget constant 70
    
    # Load 1 query
    queries = load_queries_from_file("data/trec_dl_2019/queries.tsv", max_queries=1)
    qid, qtext = list(queries.items())[0]
    
    from src.utils.lucene_utils import initialize_lucene
    initialize_lucene("./lucene_jars")
    
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index",
        encoder=encoder,
        n_docs_rm3=40,
        n_pseudo_docs=20,
    )
    
    # Extract candidates
    candidates = candidate_extractor.extract_all_candidates(qtext, qid)
    query_stats = candidate_extractor.compute_query_stats(qtext)
    
    # Run MS-MEQE Logic to get features/predictions
    msmeqe = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=None, # Not used for this diagnosis
        collection_size=8841823
    )
    
    # Manual feature extraction (mirroring expand method)
    q_emb = encoder.encode([qtext])[0]
    term_strings = [c.term for c in candidates]
    term_emb = encoder.encode(term_strings)
    
    X_val = msmeqe._build_value_features(
        query_text=qtext,
        query_embedding=q_emb,
        pseudo_centroid=None, # Simplifying
        candidates=candidates,
        term_embeddings=term_emb,
    )
    
    X_weight = msmeqe._build_weight_features(
        query_text=qtext,
        query_embedding=q_emb,
        candidates=candidates,
        term_embeddings=term_emb,
    )
    
    v_preds = msmeqe._predict_values(X_val, candidates)
    w_preds = msmeqe._predict_weights(X_weight, candidates)
    
    budget = 70.0
    
    print(f"\nQuery: {qtext} (ID: {qid})")
    print(f"Candidates: {len(candidates)}")
    print(f"Index | Term | Value (v) | Weight (w) | Ratio (v/w) | Greedy Rank | Knapsack Rank")
    print("-" * 80)
    
    # Simulate Greedy Selection
    greedy_indices = np.argsort(-v_preds)
    greedy_selected = []
    rem_budget = budget
    for idx in greedy_indices:
        if rem_budget <= 0: break
        if v_preds[idx] > 0 and w_preds[idx] <= rem_budget:
            greedy_selected.append(candidates[idx].term)
            rem_budget -= w_preds[idx]
            
    # Simulate Knapsack Selection (approx ratio sort for visualization)
    # Note: Real knapsack does dynamic programming or ratio-greedy.
    # Yield = v / w
    yields = v_preds / (w_preds + 1e-6)
    knapsack_indices = np.argsort(-yields)
    knapsack_selected = []
    rem_budget = budget
    for idx in knapsack_indices:
         if rem_budget <= 0: break
         if v_preds[idx] > 0 and w_preds[idx] <= rem_budget:
             knapsack_selected.append(candidates[idx].term)
             rem_budget -= w_preds[idx]

    # Display Top 20 Candidates by Value
    for i in range(min(20, len(candidates))):
        idx = greedy_indices[i]
        term = candidates[idx].term
        v = v_preds[idx]
        w = w_preds[idx]
        ratio = v / (w + 1e-6)
        print(f"{idx:3d} | {term[:15]:15s} | {v:6.3f} | {w:6.3f} | {ratio:6.3f}")

    print("\nOverlap Analysis (Top 70 Budget):")
    greedy_set = set(greedy_selected)
    knapsack_set = set(knapsack_selected)
    print(f"Greedy Selected Count: {len(greedy_set)}")
    print(f"Knapsack Selected Count: {len(knapsack_set)}")
    print(f"Jaccard Overlap: {len(greedy_set & knapsack_set) / len(greedy_set | knapsack_set):.2f}")
    
    diff = greedy_set - knapsack_set
    if diff:
        print(f"Terms in Greedy NOT in Knapsack: {diff}")
    diff2 = knapsack_set - greedy_set
    if diff2:
        print(f"Terms in Knapsack NOT in Greedy: {diff2}")

if __name__ == "__main__":
    main()

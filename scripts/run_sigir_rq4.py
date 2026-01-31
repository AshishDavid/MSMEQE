#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm
from src.retrieval.evaluator import TRECEvaluator
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucene-jars", type=str, required=True)
    args = parser.parse_args()
    
    initialize_lucene(args.lucene_jars)
    value_model = joblib.load("models/value_model.pkl")
    # Weight model for completeness
    weight_model = joblib.load("models/weight_model.pkl")
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    # 1. Feature Importance (XGBoost Gain)
    print("\nRQ4: Extracting Feature Importance...")
    try:
        # Assuming value_model is an XGBoost model or similar
        booster = value_model.get_booster()
        importance = booster.get_score(importance_type='gain')
        # Map indices to names if possible. For now, just save raw.
        import_df = pd.DataFrame(list(importance.items()), columns=['feature', 'gain'])
        import_df = import_df.sort_values(by='gain', ascending=False)
        import_df.to_csv("results/sigir_final_rq4_importance.csv", index=False)
        print("Feature Importance (Top 10):")
        print(import_df.head(10).to_string())
    except Exception as e:
        logger.warning(f"Could not extract XGBoost importance: {e}")

    # 2. Semantic Gap (Correlation Analysis)
    print("\nRQ4: Analyzing Semantic Gap (Correlation)...")
    config = {
        'index': 'data/msmarco_index',
        'topics': 'data/trec_dl_2019/queries.tsv',
        'qrels': 'data/trec_dl_2019/qrels.txt',
        'max_queries': 43
    }
    
    queries = load_queries_from_file(config['topics'], max_queries=config['max_queries'])
    
    # Initialize explicit extractors
    kb_ext = KBCandidateExtractor("data/wat_output_combined.jsonl")
    emb_ext = EmbeddingCandidateExtractor(encoder, "data/vocab_embeddings.pkl")
    
    # Initialize MultiSource with objects
    extractor = MultiSourceCandidateExtractor(
        config['index'], encoder,
        kb_extractor=kb_ext,
        emb_extractor=emb_ext,
        n_docs_rm3=200, n_pseudo_docs=50,
        n_kb=50, n_emb=50
    )
    model = MSMEQEExpansionModel(encoder, value_model, weight_model, None, 8841823)
    
    all_data = [] # List of (cos_sim, idp, rm3, predicted_v)
    
    for q_id, q_text in tqdm(queries.items(), desc="Collecting correlation data"):
        q_emb = encoder.encode_single(q_text)
        # q_emb is not needed for extract_all_candidates but needed for features
        # Pass query_id for KB lookup
        candidates = extractor.extract_all_candidates(q_text, query_id=str(q_id))
        if not candidates: continue
        
        query_stats = extractor.compute_query_stats(q_text)
        pseudo_centroid = extractor.compute_pseudo_centroid(q_text)
        
        # Build features manually or wrap builder
        term_strings = [c.term for c in candidates]
        term_embs = encoder.encode(term_strings)
        X_val = model._build_value_features(q_text, q_emb, pseudo_centroid, candidates, term_embs)
        v_preds = model._predict_values(X_val, candidates)
        
        # Feature indices based on src/features/feature_extraction.py
        # 0: cos_sim_q
        # 3: idf
        # 5: rm3_score
        for i in range(len(candidates)):
            all_data.append({
                'cos_sim': X_val[i, 0],
                'idf': X_val[i, 3],
                'rm3': X_val[i, 5],
                'v_pred': v_preds[i]
            })

    corr_df = pd.DataFrame(all_data)
    correlations = {
        'cos_sim': pearsonr(corr_df['cos_sim'], corr_df['v_pred'])[0],
        'idf': pearsonr(corr_df['idf'], corr_df['v_pred'])[0],
        'rm3': pearsonr(corr_df['rm3'], corr_df['v_pred'])[0]
    }
    print("Pearson Correlations with Predicted Value (v):")
    for feat, val in correlations.items():
        print(f"  {feat}: {val:.4f}")
    pd.DataFrame([correlations]).to_csv("results/sigir_final_rq4_correlations.csv", index=False)

    # 3. Case Studies
    print("\nRQ4: Extracting Case Studies...")
    case_studies = []
    # Pick a few specific queries to show Win/Loss
    target_qids = list(queries.keys())[:5]
    for q_id in target_qids:
        q_text = queries[q_id]
        q_emb = encoder.encode_single(q_text)
        # q_emb not needed for extraction
        candidates = extractor.extract_all_candidates(q_text, query_id=str(q_id))
        if not candidates:
            selected_terms = []
        else:
            query_stats = extractor.compute_query_stats(q_text)
            pseudo_centroid = extractor.compute_pseudo_centroid(q_text)
            selected_terms, _ = model.expand(q_text, candidates, pseudo_centroid, query_stats)
        
        # Just log terms for report
        terms_str = ", ".join([f"{t.term} ({t.source})" for t in selected_terms[:5]])
        case_studies.append({'query': q_text, 'selected_terms': terms_str})
    
    pd.DataFrame(case_studies).to_csv("results/sigir_final_rq4_cases.csv", index=False)

if __name__ == "__main__":
    main()

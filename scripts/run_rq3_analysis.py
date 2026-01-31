#!/usr/bin/env python3
# scripts/run_rq3_analysis.py
# RQ3: Interpretability & Feature Analysis
# Computes correlations between Features and Predicted Value/Weight

import logging
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from collections import defaultdict

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from scripts.run_msmeqe_evaluation import load_queries_from_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--value-model", type=str, required=True)
    parser.add_argument("--weight-model", type=str, required=True)
    parser.add_argument("--lucene-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output-csv", type=str, default="results/rq3_feature_correlations.csv")
    
    args = parser.parse_args()
    
    initialize_lucene(args.lucene_path)
    
    # Load Models
    logger.info("Loading models...")
    value_model = joblib.load(args.value_model)
    weight_model = joblib.load(args.weight_model)
    budget_model = None # Not needed for per-term analysis
    
    # Load Queries
    queries = load_queries_from_file(args.topics, max_queries=args.sample_size)
    
    # Components
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    candidate_extractor = MultiSourceCandidateExtractor(
        args.index, encoder, n_docs_rm3=40, n_pseudo_docs=20
    )
    
    # MS-MEQE Helper (to access private methods)
    msmeqe = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823,
        lambda_interp=0.3
    )
    
    data_records = []
    
    logger.info("Extracting features and predictions...")
    for qid, qtext in tqdm(queries.items()):
        try:
            candidates = candidate_extractor.extract_all_candidates(qtext, qid)
            if not candidates: continue
            
            # Manual Pipeline Execution to capture intermediate data
            q_emb = encoder.encode([qtext])[0]
            term_strings = [c.term for c in candidates]
            term_embs = encoder.encode(term_strings)
            
            # 1. Feature Matrices
            X_val = msmeqe._build_value_features(
                query_text=qtext,
                query_embedding=q_emb,
                pseudo_centroid=None,
                candidates=candidates,
                term_embeddings=term_embs
            )
            
            X_wt = msmeqe._build_weight_features(
                query_text=qtext,
                query_embedding=q_emb,
                candidates=candidates,
                term_embeddings=term_embs
            )
            
            # 2. Predictions
            v_preds = msmeqe._predict_values(X_val, candidates)
            w_preds = msmeqe._predict_weights(X_wt, candidates)
            
            # 3. Collect Data
            for i, c in enumerate(candidates):
                # Basic Features
                rec = {
                    'qid': qid,
                    'term': c.term,
                    'source': c.source,
                    'pred_value': v_preds[i],
                    'pred_weight': w_preds[i],
                    'rm3_score': c.rm3_score,
                    'tf_pseudo': c.tf_pseudo,
                    'df': c.df,
                    'native_score': c.native_score,
                }
                
                # Check semantic similarity (cosine)
                # q_emb and term_embs[i]
                sim = np.dot(q_emb, term_embs[i]) / (np.linalg.norm(q_emb) * np.linalg.norm(term_embs[i]) + 1e-9)
                rec['emb_sim'] = float(sim)
                
                # Heuristics
                # Value/Weight Ratio
                rec['yield'] = v_preds[i] / (w_preds[i] + 1e-6)
                
                data_records.append(rec)
                
        except Exception as e:
            logger.error(f"Error {qid}: {e}")
            
    df = pd.DataFrame(data_records)
    logger.info(f"Collected {len(df)} term samples.")
    
    # Analysis: Correlations
    logger.info("\n--- CORRELATION ANALYSIS (Pearson r) ---")
    
    features = ['rm3_score', 'tf_pseudo', 'df', 'native_score', 'emb_sim']
    targets = ['pred_value', 'pred_weight', 'yield']
    
    results = []
    
    print(f"{'Feature':<15} | {'Value (r)':<10} | {'Weight (r)':<10} | {'Yield (r)':<10}")
    print("-" * 55)
    
    for feat in features:
        row = {'Feature': feat}
        line = f"{feat:<15} |"
        for target in targets:
            # Drop NaNs
            valid = df[[feat, target]].dropna()
            if len(valid) > 10:
                r, _ = pearsonr(valid[feat], valid[target])
                row[f'{target}_r'] = r
                line += f" {r:8.4f}  |"
            else:
                line += "   NaN     |"
        print(line)
        results.append(row)
        
    # Save raw data
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved raw data to {args.output_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import joblib
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, SelectedTerm
from scripts.run_msmeqe_evaluation import load_queries_from_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucene-jars", type=str, required=True)
    args = parser.parse_args()
    
    initialize_lucene(args.lucene_jars)
    value_model = joblib.load("models/value_model.pkl")
    weight_model = joblib.load("models/weight_model.pkl")
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    configs = {
        'DL19': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2019/queries.tsv',
            'max_queries': 43
        },
        'DL20': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2020/queries.tsv',
            'max_queries': 54
        },
        'NQ': {
            'index': 'data/nq_index',
            'topics': 'data/nq/queries.tsv',
            'max_queries': 100
        }
    }
    
    all_stats = []
    
    for ds_name, config in configs.items():
        if not os.path.exists(config['index']):
            continue
            
        logger.info(f"RQ3: Analyzing Source Attribution for {ds_name}")
        queries = load_queries_from_file(config['topics'], max_queries=config['max_queries'])
        # Initialize with FULL source configuration like RQ2
        # Explicitly initialize component extractors
        kb_ext = KBCandidateExtractor("data/wat_output_combined.jsonl")
        emb_ext = EmbeddingCandidateExtractor(encoder, "data/vocab_embeddings.pkl")
        
        # Initialize MultiSource with objects
        extractor = MultiSourceCandidateExtractor(
            config['index'], encoder,
            kb_extractor=kb_ext,
            emb_extractor=emb_ext,
            n_docs_rm3=200, n_pseudo_docs=50,
            n_kb=50, n_emb=50
            # kb_path/vocab_path removed as we pass objects
        )
        
        # Use a model that predicts its own budget (Natural MS-MEQE)
        model = MSMEQEExpansionModel(encoder, value_model, weight_model, None, 8841823)
        # Note: In the codebase, budget_model might be found in models/budget_model.pkl
        if os.path.exists("models/budget_model.pkl"):
            model.budget_model = joblib.load("models/budget_model.pkl")

        dataset_source_magnitude = defaultdict(float)
        total_magnitude = 0.0
        
        for q_id, q_text in tqdm(queries.items(), desc=f"Processing {ds_name}"):
            # q_emb is not needed for extract_all_candidates
            # Pass query_id to enable KB lookup
            candidates = extractor.extract_all_candidates(q_text, query_id=str(q_id))
            if not candidates: continue
            
            query_stats = extractor.compute_query_stats(q_text)
            pseudo_centroid = extractor.compute_pseudo_centroid(q_text)
            
            selected_terms, _ = model.expand(
                query_text=q_text,
                candidates=candidates,
                pseudo_doc_centroid=pseudo_centroid,
                query_stats=query_stats
            )
            
            for term in selected_terms:
                magnitude = term.count * term.value
                # Map internal source names to clean categories
                src = term.source
                if 'rm3' in src.lower(): category = 'RM3'
                elif 'kb' in src.lower(): category = 'KB'
                elif 'emb' in src.lower(): category = 'Embedding'
                else: 
                    category = 'Other'
                    if len(all_stats) == 0 and magnitude > 0: # Print once per dataset
                         logger.info(f"DEBUG: Unclassified source: '{src}'")
                
                dataset_source_magnitude[category] += max(0, magnitude) # only positive contribution
                total_magnitude += max(0, magnitude)

        if total_magnitude > 0:
            for cat, mag in dataset_source_magnitude.items():
                all_stats.append({
                    'dataset': ds_name,
                    'source': cat,
                    'percentage': (mag / total_magnitude) * 100.0
                })

    df = pd.DataFrame(all_stats)
    df.to_csv("results/sigir_final_rq3.csv", index=False)
    print("\nRQ3 FINAL RESULTS SUMMARY:")
    print(df.to_string())

if __name__ == "__main__":
    main()


import argparse
import logging
import json
import sys
from pathlib import Path
import numpy as np
# from tqdm import tqdm

# Add project root to path
sys.path.append(".")

from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor # Need to init this explicitly to pass path
from src.reranking.semantic_encoder import SemanticEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True) # TSV or JSON
    parser.add_argument("--wat-output", required=True)
    parser.add_argument("--llm-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-path", default="data/msmarco_index")
    
    args = parser.parse_args()
    
    # 1. Load Queries
    logger.info("Loading queries...")
    queries = {}
    with open(args.queries, 'r') as f:
        # Check first line for format
        first = f.readline()
        f.seek(0)
        if first.strip().startswith("{"):
             # JSONL likely? Or just JSON?
             # Assuming standard TSV for now based on file usage
             pass
             
    # Read TSV
    with open(args.queries, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
                
    # 2. Init Extractor
    logger.info("Initializing components...")
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    kb_extractor = KBCandidateExtractor(
        wat_output_path=args.wat_output,
        max_candidates_per_query=30,
        min_entity_confidence=0.1
    )
    
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        llm_candidates_path=args.llm_candidates, # Pass file path, it loads it internally
        n_docs_rm3=10,
        n_kb=10,
        n_emb=10,
        n_pseudo_docs=10
    )
    
    # Ensure Lucene initialized (Extractor usually does it but explicit is safer)
    if not hasattr(extractor, 'lucene_searcher'):
        extractor._init_lucene_stats()
        
    # 3. Load Models for MS-MEQE
    logger.info("Loading MS-MEQE models...")
    try:
        import joblib
        value_model = joblib.load("models/value_model.pkl")
        weight_model = joblib.load("models/weight_model.pkl")
        budget_model = joblib.load("models/budget_model.pkl")
    except ImportError:
        import pickle
        with open("models/value_model.pkl", 'rb') as f: value_model = pickle.load(f)
        with open("models/weight_model.pkl", 'rb') as f: weight_model = pickle.load(f)
        with open("models/budget_model.pkl", 'rb') as f: budget_model = pickle.load(f)

    from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
    msmeqe = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823, # standard
        lambda_interp=0.3
    )

    # 4. Generate Stats
    logger.info(f"Generating stats for {len(queries)} queries...")
    
    with open(args.output, 'w') as f:
        for qid, text in queries.items():
            try:
                # 1. Extract Candidates
                candidates = extractor.extract_all_candidates(
                    query_text=text,
                    query_id=str(qid)
                )
                
                # 2. Compute basic query stats
                q_stats = extractor.compute_query_stats(text)
                
                # 3. Run Expansion Model to get Selected Terms and Values
                # needed for oracle features
                pseudo_centroid = extractor.compute_pseudo_centroid(text)
                
                selected_terms, q_star = msmeqe.expand(
                    query_text=text,
                    candidates=candidates,
                    pseudo_doc_centroid=pseudo_centroid,
                    query_stats=q_stats
                )
                
                # 4. Construct Output Dict
                # Use q_stats as base
                output_data = q_stats.copy()
                output_data['qid'] = str(qid)
                output_data['query_text'] = text
                output_data['num_candidates'] = len(candidates)
                
                # Breakdown by source
                source_counts = {'docs': 0, 'kb': 0, 'emb': 0, 'llm': 0}
                for c in candidates:
                    if c.source in source_counts:
                         source_counts[c.source] += 1
                output_data['num_candidates_by_source'] = source_counts
                
                # Selected terms info
                output_data['num_selected_terms'] = len(selected_terms)
                
                # Directional Agreement: cosine(q, q*)
                q_emb = msmeqe.encoder.encode([text])[0]
                # normalize for cosine
                # normalize for cosine
                q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                qs_norm = q_star / (np.linalg.norm(q_star) + 1e-9)
                directional_agreement = float(np.dot(q_norm, qs_norm))
                output_data['directional_agreement'] = directional_agreement
                
                sel_objs = []
                sel_source_counts = {'docs': 0, 'kb': 0, 'emb': 0, 'llm': 0}
                total_value = 0.0
                total_weight = 0.0
                
                for t in selected_terms:
                    sel_objs.append({
                        'term': t.term,
                        'source': t.source,
                        'value': t.value,
                        'weight': t.weight,
                        'count': t.count
                    })
                    if t.source in sel_source_counts:
                        sel_source_counts[t.source] += 1
                    total_value += t.value
                    total_weight += t.weight

                output_data['selected_terms'] = sel_objs
                output_data['selected_by_source'] = sel_source_counts
                output_data['total_value'] = total_value
                output_data['total_weight'] = total_weight
                
                # Add ratios for Oracle
                output_data['avg_val'] = total_value / len(selected_terms) if selected_terms else 0.0
                output_data['avg_w'] = total_weight / len(selected_terms) if selected_terms else 0.0
                
                f.write(json.dumps(output_data) + "\n")
                
            except Exception as e:
                logger.error(f"Error for {qid}: {e}")
                # Write minimal record to keep alignment? Or skip.
                # Skip is better to avoid noise
                continue
                
    logger.info(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    from src.utils.lucene_utils import initialize_lucene
    try:
        initialize_lucene(lucene_path="./lucene_jars")
    except:
        pass
    main()


import logging
import argparse
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Imports from existing project
from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.retrieval.evaluator import TRECEvaluator
from src.expansion.nucleus_sampling import apply_nucleus_sampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes to patch valid environment
class MockBudgetModel:
    def predict(self, X):
        return np.zeros(X.shape[0]) # Ignored

def run_nucleus_evaluation(args):
    """
    Run evaluation using Nucleus Sampling instead of fixed budget.
    """
    # 1. Init
    initialize_lucene(args.lucene_path)
    encoder = SemanticEncoder(model_name=args.sbert_model)
    
    # 2. Load Models
    value_model = joblib.load(args.value_model)
    weight_model = joblib.load(args.weight_model)
    budget_model = MockBudgetModel() # Not used
    
    # 3. Setup Extractors
    # A. Embedding Extractor (Global)
    emb_extractor = None
    if args.vocab_embeddings:
        logger.info(f"Initializing Embedding Extractor with {args.vocab_embeddings}")
        emb_extractor = EmbeddingCandidateExtractor(encoder=encoder, vocab_path=args.vocab_embeddings)
        
    # B. KB/LLM Data (Per Query Override)
    # We load these into memory to pass as overrides/cache
    # Note: MultiSourceCandidateExtractor handles LLM loading if path provided
    # For KB, we prefer passing the override list if we have precomputed files
    
    kb_override_map = {}
    if args.kb_data:
        logger.info(f"Loading KB candidates from {args.kb_data}")
        try:
            with open(args.kb_data, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = str(data.get('qid', data.get('id', '')))
                    if qid:
                        kb_override_map[qid] = data.get('candidates', [])
        except Exception as e:
            logger.warning(f"Failed to load KB data: {e}")

    # 4. Setup Multi-Source Extractor
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index,
        encoder=encoder,
        kb_extractor=None, # We use override map for speed/reproducibility
        emb_extractor=emb_extractor, 
        n_docs_rm3=40,
        n_pseudo_docs=20,
        n_emb=20,
        n_kb=20,
        llm_candidates_path=args.llm_data # Passes path directly to extractor
    )
    
    # 5. Dense Retriever
    from scripts.run_msmeqe_evaluation import DenseRetriever
    dense_retriever = DenseRetriever(args.index, encoder)
    
    # 6. Load Data
    from scripts.run_msmeqe_evaluation import load_queries_from_file, load_qrels_from_file
    queries = load_queries_from_file(args.topics)
    qrels = load_qrels_from_file(args.qrels)
    
    # Instantiate MSMEQE Model ONCE
    msmeqe = MSMEQEExpansionModel(
        encoder, value_model, weight_model, budget_model, 
        args.collection_size, lambda_interp=args.lambda_val
    )
    
    run_results = {}
    
    # 7. Loop
    logger.info(f"Running Nucleus Evaluation (All Sources) with p={args.p}")
    
    for qid, qtext in tqdm(queries.items()):
        try:
            qid_str = str(qid)
            
            # Prepare KB override for this query
            kb_cands = kb_override_map.get(qid_str, None)
            
            # A. Extract Candidates
            candidates = candidate_extractor.extract_all_candidates(
                query_text=qtext, 
                query_id=qid_str,
                kb_override=kb_cands  # Pass precomputed KB terms if available
            )
            
            if not candidates:
                 q_emb = encoder.encode([qtext])[0]
                 run_results[qid] = dense_retriever.retrieve(q_emb, k=1000)
                 continue
                 
            # B. Predict Values
            
            # Encode candidates
            term_strings = [c.term for c in candidates]
            if not term_strings:
                 continue
            term_embs = encoder.encode(term_strings)

            # Generate features utilizing MSMEQE internal helpers
            query_stats = candidate_extractor.compute_query_stats(qtext)
            pseudo_centroid = candidate_extractor.compute_pseudo_centroid(qtext)
            q_emb = encoder.encode([qtext])[0]

            X_val = msmeqe._build_value_features(
                query_text=qtext,
                query_embedding=q_emb,
                pseudo_centroid=pseudo_centroid,
                candidates=candidates,
                term_embeddings=term_embs
            )
            
            X_wt = msmeqe._build_weight_features(
                query_text=qtext,
                query_embedding=q_emb,
                candidates=candidates,
                term_embeddings=term_embs
            )
            
            # Predict
            values = msmeqe._predict_values(X_val, candidates)
            weights = msmeqe._predict_weights(X_wt, candidates)
            
            # Attach predictions to candidate objects
            scored_candidates = []
            for i, cand in enumerate(candidates):
                cand.value = float(values[i]) 
                cand.weight = float(weights[i])
                # Store embedding for expansion vector construction
                cand.embedding = term_embs[i]
                scored_candidates.append(cand)
                
            # C. Apply Nucleus Sampling
            selected = apply_nucleus_sampling(scored_candidates, p=args.p, max_terms=200)
            
            # D. Construct Query
            q_emb = encoder.encode([qtext])[0]
            
            expansion_vec = np.zeros_like(q_emb)
            for cand in selected:
                if hasattr(cand, 'embedding') and cand.embedding is not None:
                     expansion_vec += cand.weight * cand.embedding
            
            if len(selected) > 0:
                 expansion_vec /= len(selected)
            
            # Final Q*
            q_star = (1 - args.lambda_val) * q_emb + args.lambda_val * expansion_vec
            
            # E. Retrieve
            run_results[qid] = dense_retriever.retrieve(q_star, k=1000)
            
        except Exception as e:
            logger.error(f"Error {qid}: {e}")
            continue

    # 7. Evaluate
    evaluator = TRECEvaluator(metrics=['map', 'recip_rank', 'ndcg_cut_10', 'ndcg_cut_20', 'P_20', 'recall_1000'])
    metrics = evaluator.evaluate_run(run_results, qrels)
    print("NUCLEUS RESULTS:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--weight-model", required=True)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--lucene-path", required=True)
    parser.add_argument("--collection-size", type=int, default=528024)
    parser.add_argument("--p", type=float, default=0.9, help="Nucleus P")
    parser.add_argument("--lambda-val", type=float, default=0.4, help="Interpolation weight")
    
    # New Arguments for Data Sources
    parser.add_argument("--vocab-embeddings", default=None, help="Path to vocab_embeddings.pkl")
    parser.add_argument("--kb-data", default=None, help="Path to KB candidates JSONL")
    parser.add_argument("--llm-data", default=None, help="Path to LLM candidates JSONL")
    
    args = parser.parse_args()
    run_nucleus_evaluation(args)

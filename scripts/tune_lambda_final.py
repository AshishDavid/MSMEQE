#!/usr/bin/env python3
# scripts/tune_lambda_final.py

import argparse
import logging
import sys
import time
import joblib
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pytrec_eval

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, SelectedTerm
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
# Removed broken import

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DenseRetriever:
    """Dense retrieval using pre-computed document embeddings."""

    def __init__(self, index_path: str, encoder):
        self.encoder = encoder
        self.index_path = index_path

        # Load pre-computed document embeddings
        # Handle cases where index_path is file or directory
        base_path = Path(index_path)
        if base_path.is_file():
            base_path = base_path.parent

        doc_emb_path = base_path / "doc_embeddings.npy"
        doc_ids_path = base_path / "doc_ids.json"

        # Try parent if not found
        if not doc_emb_path.exists():
            doc_emb_path = base_path.parent / "doc_embeddings.npy"
            doc_ids_path = base_path.parent / "doc_ids.json"

        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Document embeddings not found. "
                f"Run scripts/precompute_doc_embeddings.py first."
            )

        logger.info(f"Loading document embeddings from {doc_emb_path}")
        self.doc_embeddings = np.load(str(doc_emb_path))

        with open(doc_ids_path, 'r') as f:
            self.doc_ids = json.load(f)

        logger.info(
            f"Loaded {len(self.doc_ids)} document embeddings, "
            f"dim={self.doc_embeddings.shape[1]}"
        )

    def retrieve(
            self,
            query_embedding: np.ndarray,
            k: int = 1000,
    ) -> list:
        """Retrieve documents using dense similarity."""
        # Normalize query
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        # Compute cosine similarity
        similarities = self.doc_embeddings @ q_norm

        # Get top-k
        top_k = min(k, len(similarities))
        top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Build results
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

def load_queries(topics_file):
    queries = {}
    with open(topics_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_qrels(qrels_file):
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) >= 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--weight-model", required=True)
    parser.add_argument("--budget-model", required=True)
    parser.add_argument("--lucene-path", required=True)
    parser.add_argument("--llm-candidates", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--lambdas", type=str, default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    args = parser.parse_args()

    lambda_values = [float(x) for x in args.lambdas.split(',')]
    logger.info(f"Tuning lambdas: {lambda_values}")

    initialize_lucene(args.lucene_path)

    # Components
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    value_model = joblib.load(args.value_model)
    weight_model = joblib.load(args.weight_model)
    budget_model = joblib.load(args.budget_model)
    
    dense_retriever = DenseRetriever(args.index, encoder)
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index,
        encoder=encoder,
        n_docs_rm3=40,
        n_pseudo_docs=20,
        llm_candidates_path=args.llm_candidates
    )

    msmeqe_model = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823,
        lambda_interp=0.3 # Default, will be overridden manually
    )

    queries = load_queries(args.topics)
    if args.max_queries:
        queries = dict(list(queries.items())[:args.max_queries])
    
    qrels = load_qrels(args.qrels)

    # Storage for results per lambda
    # lambda -> {qid: {docid: score}}
    results_by_lambda = {l: {} for l in lambda_values}

    logger.info(f"Processing {len(queries)} queries...")
    
    for qid, qtext in tqdm(queries.items()):
        try:
            # 1. Base Query Embedding (Always needed)
            q_emb = encoder.encode([qtext])[0]
            
            # 2. Extract and Select Terms
            candidates = candidate_extractor.extract_all_candidates(qtext, query_id=qid)
            
            if not candidates:
                # Fallback: retrieval with plain query for all lambdas
                res = dense_retriever.retrieve(q_emb, k=1000)
                for l in lambda_values:
                    results_by_lambda[l][qid] = {d: s for d, s in res}
                continue

            query_stats = candidate_extractor.compute_query_stats(qtext)
            pseudo_centroid = candidate_extractor.compute_pseudo_centroid(qtext)
            
            selected_terms, _ = msmeqe_model.expand(
                query_text=qtext,
                candidates=candidates,
                pseudo_doc_centroid=pseudo_centroid,
                query_stats=query_stats
            )
            
            # 3. Calculate Expansion Vector (Weighted Average)
            if not selected_terms:
                 expansion_vec = q_emb 
            else:
                sel_term_strings = [t.term for t in selected_terms]
                if not sel_term_strings:
                    expansion_vec = q_emb
                else:
                    sel_term_embs = encoder.encode(sel_term_strings)
                    
                    scaled_vectors = []
                    scaled_weights = []
                    
                    for i, t in enumerate(selected_terms):
                        weight = float(t.count) * float(t.value)
                        if weight <= 0: continue
                        scaled_vectors.append(weight * sel_term_embs[i])
                        scaled_weights.append(weight)
                    
                    if not scaled_vectors:
                        expansion_vec = q_emb
                    else:
                        scaled_vectors = np.stack(scaled_vectors, axis=0)
                        total_weight = sum(scaled_weights) + 1e-12
                        expansion_vec = np.sum(scaled_vectors, axis=0) / total_weight

            # 4. Sweep Lambdas
            for l in lambda_values:
                # q* = (1 - λ) e(q) + λ * exp_vec
                q_star = (1.0 - l) * q_emb + l * expansion_vec
                
                # Retrieve
                res_list = dense_retriever.retrieve(q_star, k=1000)
                results_by_lambda[l][qid] = {d: s for d, s in res_list}

        except Exception as e:
            logger.error(f"Error query {qid}: {e}")
            # Fallback
            q_emb = encoder.encode([qtext])[0]
            res = dense_retriever.retrieve(q_emb, k=1000)
            res_dict = {d: s for d, s in res}
            for l in lambda_values:
                results_by_lambda[l][qid] = res_dict
    
    # Evaluate
    logger.info("Evaluation results:")
    print(f"{'Lambda':<10} {'MAP':<10} {'nDCG@10':<10} {'Recall@1k':<10}")
    print("-" * 45)
    
    best_map = 0
    best_lambda = 0
    
    for l in lambda_values:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recall_1000'})
        eval_res = evaluator.evaluate(results_by_lambda[l])
        
        # Aggregate
        maps = [v['map'] for v in eval_res.values()]
        ndcgs = [v['ndcg_cut_10'] for v in eval_res.values()]
        recalls = [v['recall_1000'] for v in eval_res.values()]
        
        mean_map = statistics.mean(maps) if maps else 0
        mean_ndcg = statistics.mean(ndcgs) if ndcgs else 0
        mean_recall = statistics.mean(recalls) if recalls else 0
        
        print(f"{l:<10.2f} {mean_map:<10.4f} {mean_ndcg:<10.4f} {mean_recall:<10.4f}")
        
        if mean_map > best_map:
            best_map = mean_map
            best_lambda = l

    print("-" * 45)
    print(f"Best Lambda (MAP): {best_lambda} (MAP: {best_map:.4f})")

if __name__ == "__main__":
    import statistics
    main()

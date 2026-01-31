
"""
Tune MS-MEQE interpolation weight (lambda).
"""

import logging
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm
import joblib
from pathlib import Path

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.retrieval.evaluator import TRECEvaluator

logger = logging.getLogger(__name__)

class Tuner:
    def __init__(self, args):
        self.args = args
        self.encoder = SemanticEncoder(model_name=args.sbert_model)
        
        # Load models
        self.value_model = joblib.load(args.value_model)
        self.weight_model = joblib.load(args.weight_model)
        self.budget_model = joblib.load(args.budget_model)
        
        # Initialize Extractors
        kb_extractor = None
        if args.kb_wat_output:
            kb_extractor = KBCandidateExtractor(wat_output_path=args.kb_wat_output)
            
        emb_extractor = None
        if args.emb_vocab:
            emb_extractor = EmbeddingCandidateExtractor(encoder=self.encoder, vocab_path=args.emb_vocab)

        self.candidate_extractor = MultiSourceCandidateExtractor(
            index_path=args.index,
            encoder=self.encoder,
            kb_extractor=kb_extractor,
            emb_extractor=emb_extractor,
            llm_candidates_path=args.llm_candidates
        )
        
        # Dense Retrieval Pre-load
        base_path = Path(args.index)
        if base_path.is_file(): base_path = base_path.parent
        self.doc_embeddings = np.load(str(base_path / "doc_embeddings.npy"), mmap_mode='r')
        with open(base_path / "doc_ids.json", 'r') as f:
            self.doc_ids = json.load(f)
            
        self.evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'P_10'])

    def run_tuning(self, queries, qrels, lambdas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        best_map = -1
        best_lambda = -1
        results_per_lambda = {}
        
        # Pre-compute query stats and candidates ONCE
        logger.info("Pre-computing candidates...")
        query_data = {}
        for qid, qtext in tqdm(queries.items(), desc="Pre-processing"):
            try:
                candidates = self.candidate_extractor.extract_all_candidates(qtext, qid)
                stats = self.candidate_extractor.compute_query_stats(qtext)
                centroid = self.candidate_extractor.compute_pseudo_centroid(qtext)
                query_data[qid] = (qtext, candidates, stats, centroid)
            except Exception as e:
                logger.warning(f"Failed {qid}: {e}")

        logger.info(f"Starting Grid Search on {lambdas}")
        
        for lam in lambdas:
            logger.info(f"Evaluating lambda={lam}")
            
            # Re-initialize MS-MEQE with specific lambda
            model = MSMEQEExpansionModel(
                encoder=self.encoder,
                value_model=self.value_model,
                weight_model=self.weight_model,
                budget_model=self.budget_model,
                collection_size=self.args.collection_size,
                lambda_interp=lam
            )
            
            run_results = {}
            
            for qid, (qtext, candidates, stats, centroid) in query_data.items():
                try:
                    selected, q_star = model.expand(qtext, candidates, centroid, stats)
                    
                    # Retrieval
                    q_norm = q_star / (np.linalg.norm(q_star) + 1e-12)
                    sims = self.doc_embeddings @ q_norm
                    
                    # Top-1000
                    k = 1000
                    top_idx = np.argpartition(-sims, k-1)[:k]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                    
                    run_results[qid] = [(self.doc_ids[i], float(sims[i])) for i in top_idx]
                except:
                    continue
            
            metrics = self.evaluator.evaluate_run(run_results, qrels)
            current_map = metrics.get('map', 0)
            
            logger.info(f"Lambda {lam}: MAP={current_map:.4f}, nDCG={metrics.get('ndcg_cut_10', 0):.4f}")
            results_per_lambda[lam] = metrics
            
            if current_map > best_map:
                best_map = current_map
                best_lambda = lam
                # Save best run
                self.best_run_results = run_results

        logger.info(f"=== TUNING COMPLETE ===")
        logger.info(f"Best Lambda: {best_lambda} (MAP={best_map:.4f})")
        
        # Save best run file
        output_path = Path(self.args.output_dir) / f"tuned_run_lambda_{best_lambda}.txt"
        with open(output_path, 'w') as f:
            for qid, res in self.best_run_results.items():
                for rank, (docid, score) in enumerate(res, start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} MS-MEQE-Tuned\n")
        logger.info(f"Saved best run to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--lucene-path", required=True)
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--weight-model", required=True)
    parser.add_argument("--budget-model", required=True)
    parser.add_argument("--llm-candidates", default=None)
    parser.add_argument("--kb-wat-output", default=None)
    parser.add_argument("--emb-vocab", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection-size", type=int, default=8841823)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    initialize_lucene(args.lucene_path)
    
    # Load data
    queries = {}
    with open(args.topics, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2: queries[parts[0]] = parts[1]
            
    # Limit queries
    queries = dict(list(queries.items())[:args.max_queries])
    
    qrels = {} # Load qrels... simplified
    with open(args.qrels, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                docid = parts[2]
                rel = int(parts[3])
                if qid not in qrels: qrels[qid] = {}
                qrels[qid][docid] = rel
                
    tuner = Tuner(args)
    tuner.run_tuning(queries, qrels)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# scripts/rerank_bm25_robust.py

import argparse
import logging
import sys
import json
import time
import joblib
import numpy as np
import pytrec_eval
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.utils.file_utils import load_trec_run, load_qrels

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Robust04Reranker:
    def __init__(self, args):
        self.args = args
        initialize_lucene(args.lucene_path)
        
        self.encoder = SemanticEncoder(model_name=args.sbert_model)
        
        # Load models
        logger.info("Loading MS-MEQE models...")
        self.value_model = joblib.load(args.value_model)
        self.weight_model = joblib.load(args.weight_model)
        self.budget_model = joblib.load(args.budget_model)
        
        # Load KB candidates into map
        self.kb_override_map = {}
        if args.kb_data:
            logger.info(f"Loading KB candidates from {args.kb_data}")
            with open(args.kb_data, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    qid = data.get('qid', data.get('id', ''))
                    if qid:
                        self.kb_override_map[str(qid)] = data.get('candidates', [])
            
        emb_extractor = None
        if args.emb_vocab:
            logger.info("Initializing embedding extractor...")
            emb_extractor = EmbeddingCandidateExtractor(encoder=self.encoder, vocab_path=args.emb_vocab)

        self.candidate_extractor = MultiSourceCandidateExtractor(
            index_path=args.index,
            encoder=self.encoder,
            kb_extractor=None, # We use overrides instead
            emb_extractor=emb_extractor
        )
        
        # Initialize MS-MEQE model skeleton
        self.msmeqe_model = MSMEQEExpansionModel(
            encoder=self.encoder,
            value_model=self.value_model,
            weight_model=self.weight_model,
            budget_model=self.budget_model,
            collection_size=args.collection_size,
            lambda_interp=0.4 # Default
        )

        # Load document embeddings for re-ranking
        logger.info("Loading document embeddings for re-ranking...")
        base_path = Path(args.index)
        self.doc_embeddings = np.load(str(base_path / "doc_embeddings.npy"), mmap_mode='r')
        with open(base_path / "doc_ids.json", 'r') as f:
            doc_id_list = json.load(f)
            self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_id_list)}

    def get_folds(self, qids):
        """Split queries into 5 sequential folds."""
        sorted_qids = sorted([str(q) for q in qids])
        n = len(sorted_qids)
        fold_size = (n + 4) // 5
        folds = []
        for i in range(0, n, fold_size):
            folds.append(sorted_qids[i:i + fold_size])
        return folds

    def compute_expanded_embeddings(self, queries):
        """Pre-compute q_emb and expansion_vec for each query."""
        results = {}
        for qid, qtext in tqdm(queries.items(), desc="Pre-computing expansions"):
            try:
                # 1. Base Query Embedding
                q_emb = self.encoder.encode([qtext])[0]
                
                # 2. Extract and Select Terms
                candidates = self.candidate_extractor.extract_all_candidates(
                    qtext, 
                    query_id=qid,
                    kb_override=self.kb_override_map.get(str(qid))
                )
                stats = self.candidate_extractor.compute_query_stats(qtext)
                pseudo_centroid = self.candidate_extractor.compute_pseudo_centroid(qtext)
                
                selected_terms, _ = self.msmeqe_model.expand(
                    query_text=qtext,
                    candidates=candidates,
                    pseudo_doc_centroid=pseudo_centroid,
                    query_stats=stats
                )
                
                # 3. Calculate Expansion Vector
                if not selected_terms:
                    expansion_vec = q_emb
                else:
                    sel_term_strings = [t.term for t in selected_terms]
                    sel_term_embs = self.encoder.encode(sel_term_strings)
                    
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
                        expansion_vec = np.sum(scaled_vectors, axis=0) / (sum(scaled_weights) + 1e-12)
                
                results[qid] = {
                    'q_emb': q_emb,
                    'expansion_vec': expansion_vec
                }
            except Exception as e:
                logger.error(f"Error processing query {qid}: {e}")
                results[qid] = None
        return results

    def rerank_docs(self, qid, q_star, baseline_docs):
        """Rerank docs for one query."""
        q_norm = q_star / (np.linalg.norm(q_star) + 1e-12)
        
        doc_idxs = []
        valid_docids = []
        for docid in baseline_docs:
            if docid in self.doc_id_to_idx:
                doc_idxs.append(self.doc_id_to_idx[docid])
                valid_docids.append(docid)
        
        if not doc_idxs:
            return []
            
        # Efficiently pull only the docs we need (if using mmap)
        subset_embs = self.doc_embeddings[doc_idxs]
        
        # doc_embeddings are usually pre-normalized, but let's be safe
        norms = np.linalg.norm(subset_embs, axis=1, keepdims=True) + 1e-12
        subset_embs_norm = subset_embs / norms
        
        sims = subset_embs_norm @ q_norm
        
        results = sorted([(d, float(s)) for d, s in zip(valid_docids, sims)], key=lambda x: x[1], reverse=True)
        return results

    def run_cv(self):
        # Load data
        queries = {}
        with open(self.args.topics, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2: queries[parts[0]] = parts[1]
        
        qrels = load_qrels(self.args.qrels)
        baseline_run = load_trec_run(self.args.bm25_run)
        
        # Filter queries present in both run and topics
        qids = sorted(list(set(queries.keys()) & set(baseline_run.keys())))
        queries = {qid: queries[qid] for qid in qids}
        
        logger.info(f"Running CV on {len(qids)} queries")
        
        # Pre-compute
        expansions = self.compute_expanded_embeddings(queries)
        
        # Split folds
        folds = self.get_folds(qids)
        
        final_run_results = {}
        fold_lambdas = []
        
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for i in range(5):
            test_fold = folds[i]
            train_folds = [qid for f_idx, fold in enumerate(folds) if f_idx != i for qid in fold]
            
            logger.info(f"--- Fold {i+1}/5 ---")
            logger.info(f"Train on {len(train_folds)} queries, Test on {len(test_fold)} queries")
            
            # 1. Tune on Training Folds
            best_map = -1
            best_lam = 0.4
            
            for lam in lambdas:
                train_run = {}
                for qid in train_folds:
                    if expansions[qid] is None: continue
                    q_star = (1.0 - lam) * expansions[qid]['q_emb'] + lam * expansions[qid]['expansion_vec']
                    train_run[qid] = {d: s for d, s in self.rerank_docs(qid, q_star, [d for d, s in baseline_run[qid]])}
                
                evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map'})
                metrics = evaluator.evaluate(train_run)
                mean_map = statistics.mean([v['map'] for v in metrics.values()])
                
                logger.info(f"  Lam {lam:.1f}: Train MAP = {mean_map:.4f}")
                if mean_map > best_map:
                    best_map = mean_map
                    best_lam = lam
            
            logger.info(f"  Optimal Lambda for Fold {i+1}: {best_lam}")
            fold_lambdas.append(best_lam)
            
            # 2. Evaluate on Test Fold
            for qid in test_fold:
                if expansions[qid] is None:
                    final_run_results[qid] = baseline_run[qid]
                    continue
                q_star = (1.0 - best_lam) * expansions[qid]['q_emb'] + best_lam * expansions[qid]['expansion_vec']
                final_run_results[qid] = self.rerank_docs(qid, q_star, [d for d, s in baseline_run[qid]])

        # Final Evaluation
        logger.info("========================================================")
        logger.info("FINAL CROSS-VALIDATED RESULTS")
        logger.info("========================================================")
        
        eval_dict = {qid: {d: s for d, s in res} for qid, res in final_run_results.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recip_rank'})
        final_metrics = evaluator.evaluate(eval_dict)
        
        avg_map = statistics.mean([v['map'] for v in final_metrics.values()])
        avg_ndcg = statistics.mean([v['ndcg_cut_10'] for v in final_metrics.values()])
        avg_mrr = statistics.mean([v['recip_rank'] for v in final_metrics.values()])
        
        logger.info(f"Average MAP:      {avg_map:.4f}")
        logger.info(f"Average nDCG@10:  {avg_ndcg:.4f}")
        logger.info(f"Average MRR:      {avg_mrr:.4f}")
        logger.info(f"Mean Opt Lambda:  {statistics.mean(fold_lambdas):.4f}")
        
        # Save run file
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for qid in sorted(final_run_results.keys()):
                for rank, (docid, score) in enumerate(final_run_results[qid], start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} MS-MEQE-Rerank-CV\n")
        logger.info(f"Cross-validated run saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--bm25-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--value-model", required=True)
    parser.add_argument("--weight-model", required=True)
    parser.add_argument("--budget-model", required=True)
    parser.add_argument("--emb-vocab", required=True)
    parser.add_argument("--kb-data", required=True)
    parser.add_argument("--lucene-path", default="lucene_jars")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection-size", type=int, default=528155)
    
    args = parser.parse_args()
    
    reranker = Robust04Reranker(args)
    reranker.run_cv()

if __name__ == "__main__":
    main()

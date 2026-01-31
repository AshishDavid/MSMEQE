#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.retrieval.evaluator import TRECEvaluator
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ConstantBudgetModel(MSMEQEExpansionModel):
    def __init__(self, *args, target_w=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_w = target_w
    
    def _predict_budget(self, query_stats):
        return self.target_w

class RM3BaselineModel(MSMEQEExpansionModel):
    """
    Acts like Top-K RM3 by taking top K terms by value (ignoring weights and risk).
    """
    def __init__(self, *args, target_k=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_k = target_k

    def _predict_budget(self, query_stats):
        return float(self.target_k)

    def _predict_values(self, X, candidates=None):
        return np.array([c.rm3_score for c in candidates])

    def _solve_selection(self, values, weights, budget, term_embeddings=None):
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        indices = np.argsort(-values)
        k = int(budget)
        for i in range(min(k, m)):
            counts[indices[i]] = 1
        return counts

    def _build_enhanced_query_embedding(self, query_embedding, term_embeddings, values, counts, clarity):
        # Disable v > 0 filter for true aggressive baseline
        scaled_vectors = []
        scaled_weights = []
        for emb, v, c in zip(term_embeddings, values, counts):
            if c <= 0: continue
            weight = float(c) * max(1e-6, float(v))
            scaled_vectors.append(weight * emb)
            scaled_weights.append(weight)
        if not scaled_vectors: return query_embedding
        dyn_lambda = 0.3 # Fixed lambda for baseline
        weighted_avg = np.sum(scaled_vectors, axis=0) / (np.sum(scaled_weights) + 1e-12)
        return (1.0 - dyn_lambda) * query_embedding + dyn_lambda * weighted_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucene-jars", type=str, required=True)
    args = parser.parse_args()
    
    initialize_lucene(args.lucene_jars)
    value_model = joblib.load("models/value_model.pkl")
    weight_model = joblib.load("models/weight_model.pkl")
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize Multi-Source Components
    kb_extractor = None
    if os.path.exists("data/wat_output_test.jsonl"):
        kb_extractor = KBCandidateExtractor(wat_output_path="data/wat_output_test.jsonl")
    
    emb_extractor = None
    if os.path.exists("data/vocab_embeddings.pkl"):
        emb_extractor = EmbeddingCandidateExtractor(encoder=encoder, vocab_path="data/vocab_embeddings.pkl")
    
    configs = {
        'DL19': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2019/queries.tsv',
            'qrels': 'data/trec_dl_2019/qrels.txt',
            'max_queries': 43
        },
        'DL20': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2020/queries.tsv',
            'qrels': 'data/trec_dl_2020/qrels.txt',
            'max_queries': 54
        },
        'NQ': {
            'index': 'data/nq_index',
            'topics': 'data/nq/queries.tsv',
            'qrels': 'data/nq/qrels.txt',
            'max_queries': 100
        }
    }
    
    budgets = [10, 30, 50, 70, 100]
    all_results = []
    
    for ds_name, config in configs.items():
        if not os.path.exists(config['index']):
            continue
        
        queries = load_queries_from_file(config['topics'], max_queries=config['max_queries'])
        qrels = load_qrels_from_file(config['qrels'])
        # Initialize extractor with explicit KB/Emb components
        kb_path = "data/wat_output_combined.jsonl"
        vocab_path = "data/vocab_embeddings.pkl"
        
        kb_ext = KBCandidateExtractor(wat_output_path=kb_path)
        emb_ext = EmbeddingCandidateExtractor(encoder=encoder, vocab_path=vocab_path)

        extractor = MultiSourceCandidateExtractor(
            config['index'], encoder,
            n_docs_rm3=200, n_pseudo_docs=50,
            n_kb=50, n_emb=50,
            kb_extractor=kb_ext,
            emb_extractor=emb_ext
        )
        dense_retriever = DenseRetriever(config['index'], encoder)
        evaluator = TRECEvaluator(metrics=['ndcg_cut_10', 'map'])
        
        for k_val in budgets:
            logger.info(f"RQ2: {ds_name} | Budget k={k_val}")
            
            # RM3 Baseline (Top-K Value)
            rm3_model = RM3BaselineModel(encoder, value_model, weight_model, None, 8841823, target_k=k_val)
            rm3_pipe = MSMEQEEvaluationPipeline(encoder, extractor, rm3_model, dense_retriever, evaluator)
            res_rm3 = rm3_pipe.run_evaluation(queries, qrels, topk=1000)
            
            # MS-MEQE (Knapsack with Risk)
            # For MS-MEQE to show "Safe Aggression", it needs to use its predicted weights.
            # Here W=k_val is the CAP, but it might take fewer terms if density is low.
            mseqe_model = ConstantBudgetModel(encoder, value_model, weight_model, None, 8841823, target_w=float(k_val))
            mseqe_pipe = MSMEQEEvaluationPipeline(encoder, extractor, mseqe_model, dense_retriever, evaluator)
            res_ms = mseqe_pipe.run_evaluation(queries, qrels, topk=1000)
            
            all_results.append({
                'dataset': ds_name,
                'budget': k_val,
                'method': 'RM3_ValueOnly',
                'ndcg@10': res_rm3['metrics']['ndcg_cut_10'],
                'map': res_rm3['metrics']['map']
            })
            all_results.append({
                'dataset': ds_name,
                'budget': k_val,
                'method': 'MS-MEQE',
                'ndcg@10': res_ms['metrics']['ndcg_cut_10'],
                'map': res_ms['metrics']['map']
            })

    df = pd.DataFrame(all_results)
    df.to_csv("results/sigir_final_rq2.csv", index=False)
    print("\nRQ2 FINAL RESULTS SUMMARY:")
    print(df.to_string())

if __name__ == "__main__":
    main()

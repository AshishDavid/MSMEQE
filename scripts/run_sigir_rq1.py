#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, SelectedTerm
from src.retrieval.evaluator import TRECEvaluator
from scripts.run_msmeqe_evaluation import (
    load_queries_from_file,
    load_qrels_from_file,
    DenseRetriever,
    MSMEQEEvaluationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SIGIR_RQ1_Model(MSMEQEExpansionModel):
    """
    Model for RQ1 that implements Heterogeneous Weighting and Noise Injection.
    """
    def __init__(self, *args, boost_legit=100.0, noise_v=0.3, noise_w=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.boost_legit = boost_legit
        self.noise_v = noise_v
        self.noise_w = noise_w

    def _predict_values(self, X, candidates):
        preds = super()._predict_values(X, candidates)
        # Boost legit values to make them competitive as a bundle
        preds = preds * self.boost_legit
        
        for i, c in enumerate(candidates):
            if hasattr(c, 'source') and c.source == 'noise':
                preds[i] = self.noise_v
        return preds

    def _predict_weights(self, X, candidates):
        # Implementation of Heterogeneous Weighting: w = log10(DF)
        # We clip min to 1 Doc to avoid log(0)
        weights = np.zeros(len(candidates))
        for i, c in enumerate(candidates):
            if hasattr(c, 'source') and c.source == 'noise':
                weights[i] = self.noise_w
            else:
                df = max(1, c.df)
                weights[i] = np.log10(df)
        return weights

class Greedy_SIGIR_RQ1_Model(SIGIR_RQ1_Model):
    def _solve_selection(self, values, weights, budget, term_embeddings=None):
        m = len(values)
        counts = np.zeros(m, dtype=np.int32)
        remaining = float(budget)
        indices = np.argsort(-values)
        for idx in indices:
            w = weights[idx]
            v = values[idx]
            if v > 0 and w <= remaining:
                counts[idx] = 1
                remaining -= w
        return counts

def inject_noise(candidates, n=20):
    from src.expansion.candidate_extraction_pipeline import CandidateTerm
    noise_candidates = []
    for i in range(n):
        c = CandidateTerm(
            term=f"noise_term_{i}",
            df=1000000, # Not used for weight because we override _predict_weights
            source='noise',
            native_score=0.0
        )
        noise_candidates.append(c)
    return candidates + noise_candidates

class NoiseInjectingCandidateExtractor:
    def __init__(self, base_extractor, n_noise=20):
        self.base_extractor = base_extractor
        self.n_noise = n_noise
    
    def extract_all_candidates(self, query_text, query_id=None, kb_override=None):
        candidates = self.base_extractor.extract_all_candidates(query_text, query_id, kb_override)
        return inject_noise(candidates, self.n_noise)

    def compute_query_stats(self, query_text):
        return self.base_extractor.compute_query_stats(query_text)

    def compute_pseudo_centroid(self, query_text):
        return self.base_extractor.compute_pseudo_centroid(query_text)

def run_experiment(dataset_name, dataset_config, value_model, budget_sweep, n_noise=20):
    initialize_lucene(dataset_config['lucene_path'])
    queries = load_queries_from_file(dataset_config['topics'], max_queries=dataset_config.get('max_queries', 100))
    qrels = load_qrels_from_file(dataset_config['qrels'])
    encoder = SemanticEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    
    kb_path = "data/wat_output_combined.jsonl"
    vocab_path = "data/vocab_embeddings.pkl"
    
    kb_ext = KBCandidateExtractor(wat_output_path=kb_path)
    emb_ext = EmbeddingCandidateExtractor(encoder=encoder, vocab_path=vocab_path)
    
    base_extractor = MultiSourceCandidateExtractor(
        dataset_config['index'], encoder, 
        n_docs_rm3=40, n_pseudo_docs=20,
        n_kb=30, n_emb=30,
        kb_extractor=kb_ext,
        emb_extractor=emb_ext
    )
    extractor = NoiseInjectingCandidateExtractor(base_extractor, n_noise=n_noise)
    dense_retriever = DenseRetriever(dataset_config['index'], encoder)
    evaluator = TRECEvaluator(metrics=['ndcg_cut_10', 'map'])
    
    results = []
    
    for W in budget_sweep:
        logger.info(f"Running {dataset_name} Sweep Budget W={W}")
        
        # 1. Greedy
        greedy_model = Greedy_SIGIR_RQ1_Model(
            encoder, value_model, None, None, 8841823, min_budget=W, max_budget=W, budget_step=1
        )
        greedy_model._predict_budget = lambda x: W
        
        # 2. Knapsack
        knapsack_model = SIGIR_RQ1_Model(
            encoder, value_model, None, None, 8841823, min_budget=W, max_budget=W, budget_step=1
        )
        knapsack_model._predict_budget = lambda x: W
        
        # We need to collect noise stats. The easy way is to run query by query or use a wrapper.
        # Let's modify the pipeline to return selected terms or wrap it.
        
        for name, model in [("Greedy", greedy_model), ("Knapsack", knapsack_model)]:
            pipeline = MSMEQEEvaluationPipeline(encoder, extractor, model, dense_retriever, evaluator)
            
            noise_counts = []
            def track_noise(selected_terms, q_star):
                n = sum(1 for t in selected_terms if t.source == 'noise')
                noise_counts.append(n)
            
            original_expand = model.expand
            model.expand = lambda *args, **kwargs: (res := original_expand(*args, **kwargs), track_noise(*res), res)[2]
            
            res = pipeline.run_evaluation(queries, qrels, topk=1000)
            avg_noise = np.mean(noise_counts) if noise_counts else 0.0
            
            results.append({
                'dataset': dataset_name,
                'W': W,
                'method': name,
                'ndcg@10': res['metrics']['ndcg_cut_10'],
                'map': res['metrics']['map'],
                'avg_noise': avg_noise
            })
            
            logger.info(f"DONE: {dataset_name} | W={W} | {name} | nDCG@10: {res['metrics']['ndcg_cut_10']:.4f} | Noise: {avg_noise:.2f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lucene-jars", type=str, required=True)
    args = parser.parse_args()
    
    value_model = joblib.load("models/value_model.pkl")
    
    configs = {
        'DL19': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2019/queries.tsv',
            'qrels': 'data/trec_dl_2019/qrels.txt',
            'lucene_path': args.lucene_jars,
            'max_queries': 43
        },
        'DL20': {
            'index': 'data/msmarco_index',
            'topics': 'data/trec_dl_2020/queries.tsv',
            'qrels': 'data/trec_dl_2020/qrels.txt',
            'lucene_path': args.lucene_jars,
            'max_queries': 54
        },
        'NQ': {
            'index': 'data/nq_index',
            'topics': 'data/nq/queries.tsv',
            'qrels': 'data/nq/qrels.txt',
            'lucene_path': args.lucene_jars,
            'max_queries': 100
        }
    }
    
    budget_sweep = [1, 3, 5, 10]
    all_results = []
    
    for ds_name, config in configs.items():
        if not os.path.exists(config['index']):
            logger.warning(f"Index not found for {ds_name}, skipping.")
            continue
        res = run_experiment(ds_name, config, value_model, budget_sweep)
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    df.to_csv("results/sigir_final_rq1.csv", index=False)
    print("\nRQ1 FINAL RESULTS SUMMARY:")
    print(df[df['W'] == 5].to_string())

if __name__ == "__main__":
    main()

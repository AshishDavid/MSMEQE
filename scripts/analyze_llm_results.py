
"""
Analyze MS-MEQE results: Baseline vs Tuned.
"""
import sys
import argparse
import logging
import json
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def compute_ap(run_list, qrels_dict):
    """
    Compute Average Precision for a single query.
    run_list: [(docid, score), ...]
    qrels_dict: {docid: relevance}
    """
    if not qrels_dict: return 0.0
    
    score = 0.0
    num_rel = 0
    total_rel = sum([1 for r in qrels_dict.values() if r > 0])
    if total_rel == 0: return 0.0
    
    for i, (docid, _) in enumerate(run_list, start=1):
        if docid in qrels_dict and qrels_dict[docid] > 0:
            num_rel += 1
            score += num_rel / i
            
    return score / total_rel

class Analyzer:
    def __init__(self, args):
        self.args = args
        self.encoder = SemanticEncoder(model_name=args.sbert_model)
        
        self.value_model = joblib.load(args.value_model)
        self.weight_model = joblib.load(args.weight_model)
        self.budget_model = joblib.load(args.budget_model)
        
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
        
        base_path = Path(args.index)
        if base_path.is_file(): base_path = base_path.parent
        self.doc_embeddings = np.load(str(base_path / "doc_embeddings.npy"), mmap_mode='r')
        with open(base_path / "doc_ids.json", 'r') as f:
            self.doc_ids = json.load(f)

    def analyze(self, queries, qrels, best_lambda):
        # 1. Baseline (Lambda=0)
        logger.info("Computing Baseline (Lambda=0)...")
        base_results = self._run_model(queries, lam=0.0)
        
        # 2. Tuned (Lambda=BEST)
        logger.info(f"Computing Tuned (Lambda={best_lambda})...")
        tuned_results = self._run_model(queries, lam=best_lambda)
        
        # 3. Compare aggregate metrics
        import pytrec_eval
        
        # Convert to pytrec_eval format
        qrels_eval = {str(qid): {str(did): int(rel) for did, rel in docs.items()} for qid, docs in qrels.items()}
        
        run_base = {str(qid): {str(did): float(score) for did, score in base_results.get(qid, [])} for qid in queries}
        run_tuned = {str(qid): {str(did): float(score) for did, score in tuned_results.get(qid, [])} for qid in queries}
        
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_eval, {'map', 'ndcg_cut_10', 'recall_1000'})
        
        res_base = evaluator.evaluate(run_base)
        res_tuned = evaluator.evaluate(run_tuned)
        
        def avg_metric(res, metric):
            if not res: return 0.0
            return sum(r[metric] for r in res.values()) / len(res)
            
        print("\n" + "="*60)
        print("AGGREGATE METRICS (Subset N={})".format(len(queries)))
        print("="*60)
        print(f"{'Metric':<15} {'Baseline (Lam=0)':<20} {'Tuned (Lam={best_lambda})':<20} {'Diff %':<10}")
        print("-" * 65)
        
        for m in ['map', 'ndcg_cut_10', 'recall_1000']:
            b = avg_metric(res_base, m)
            t = avg_metric(res_tuned, m)
            pct = ((t - b) / b * 100) if b > 0 else 0.0
            print(f"{m:<15} {b:.4f}{' '*14} {t:.4f}{' '*14} {pct:+.2f}%")
        print("="*60 + "\n")
        
        # 4. Compare Per-Query (AP)
        diffs = []
        for qid in queries:
            if qid not in qrels: continue
            
            ap_base = compute_ap(base_results.get(qid, []), qrels.get(qid, {}))
            ap_tuned = compute_ap(tuned_results.get(qid, []), qrels.get(qid, {}))
            
            diff = ap_tuned - ap_base
            diffs.append({
                'qid': qid,
                'qtext': queries[qid],
                'ap_base': ap_base,
                'ap_tuned': ap_tuned,
                'diff': diff
            })
            
        # Sort
        diffs.sort(key=lambda x: x['diff'], reverse=True)
        
        print("\n" + "="*60)
        print(f"ANALYSIS REPORT (Lambda {best_lambda} vs Baseline)")
        print("="*60)
        
        print("\nTOP 3 IMPROVED:")
        for item in diffs[:3]:
            self._print_item(item)
            
        print("\nTOP 3 HURT:")
        for item in diffs[-3:]:
            self._print_item(item)

    def _run_model(self, queries, lam):
        model = MSMEQEExpansionModel(
            encoder=self.encoder,
            value_model=self.value_model,
            weight_model=self.weight_model,
            budget_model=self.budget_model,
            collection_size=self.args.collection_size,
            lambda_interp=lam
        )
        
        results = {}
        for qid, qtext in tqdm(queries.items()):
            try:
                candidates = self.candidate_extractor.extract_all_candidates(qtext, qid)
                stats = self.candidate_extractor.compute_query_stats(qtext)
                centroid = self.candidate_extractor.compute_pseudo_centroid(qtext)
                
                selected, q_star = model.expand(qtext, candidates, centroid, stats)
                
                q_norm = q_star / (np.linalg.norm(q_star) + 1e-12)
                sims = self.doc_embeddings @ q_norm
                
                k = 1000
                top_idx = np.argpartition(-sims, k-1)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                
                results[qid] = [(self.doc_ids[i], float(sims[i])) for i in top_idx]
                
                # Hack: Store candidates for printing later?
                # For now just re-extract in print_item if needed, or rely on fast extraction
            except:
                continue
        return results

    def _print_item(self, item):
        print(f"\nQuery [{item['qid']}]: {item['qtext']}")
        print(f"AP: {item['ap_base']:.4f} -> {item['ap_tuned']:.4f} (Diff: {item['diff']:+.4f})")
        
        # Show LLM candidates
        candidates = self.candidate_extractor.extract_all_candidates(item['qtext'], item['qid'])
        
        # DEBUG: Print all sources found
        sources = set(c.source for c in candidates)
        print(f"DEBUG Sources found: {sources}")
        
        llm_cands = [c.term for c in candidates if c.source.lower() == 'llm']
        print(f"LLM Candidates ({len(llm_cands)}): {', '.join(llm_cands[:10])}...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-lambda", type=float, required=True)
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
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection-size", type=int, default=8841823)
    
    args = parser.parse_args()
    initialize_lucene(args.lucene_path)
    
    queries = {}
    with open(args.topics, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2: queries[parts[0]] = parts[1]
    queries = dict(list(queries.items())[:args.max_queries])
    
    qrels = {}
    with open(args.qrels, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                docid = parts[2]
                rel = int(parts[3])
                if qid not in qrels: qrels[qid] = {}
                qrels[qid][docid] = rel
                
    analyzer = Analyzer(args)
    analyzer.analyze(queries, qrels, args.best_lambda)

if __name__ == "__main__":
    main()

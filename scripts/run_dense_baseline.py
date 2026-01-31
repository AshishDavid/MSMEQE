
import argparse
import logging
import json
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.reranking.semantic_encoder import SemanticEncoder
from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_qrels, save_trec_run

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Fast Dense Baseline")
    parser.add_argument("--index", type=str, required=True, help="Path to document embeddings directory")
    parser.add_argument("--topics", type=str, required=True, help="Path to queries TSV")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--output", type=str, required=True, help="Output run file")
    parser.add_argument("--doc-ids", type=str, default="doc_ids.json", help="Doc IDs filename")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # 1. Load Resources
    logger.info(f"Loading encoder: {args.model}")
    encoder = SemanticEncoder(model_name=args.model)

    logger.info(f"Loading doc embeddings from {args.index}")
    doc_emb_path = Path(args.index) / "doc_embeddings.npy"
    doc_ids_path = Path(args.index) / args.doc_ids
    
    # Use mmap for speed/memory
    doc_embeddings = np.load(str(doc_emb_path), mmap_mode='r')
    with open(doc_ids_path, 'r') as f:
        doc_ids = json.load(f)
    
    logger.info(f"Loaded {len(doc_ids)} docs. Shape: {doc_embeddings.shape}")

    # 2. Load Queries
    queries = {}
    with open(args.topics, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    
    logger.info(f"Loaded {len(queries)} queries")

    # 3. Dense Retrieval Loop
    logger.info("Starting retrieval...")
    run_results = {}
    
    batch_size = 32
    query_ids = list(queries.keys())
    
    total_time = 0
    
    for i in tqdm(range(0, len(query_ids), batch_size)):
        batch_qids = query_ids[i:i+batch_size]
        batch_texts = [queries[qid] for qid in batch_qids]
        
        # Encode queries
        q_embs = encoder.encode(batch_texts) # (B, D)
        
        # Normalize
        q_norms = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12)
        
        # Dot product
        # (B, D) @ (N, D).T = (B, N)
        # Note: dot with mmap might be slow if memory is tight, but usually fine for batch reading
        # Optimized: perform batch dot product
        start = time.time()
        scores = np.dot(q_norms, doc_embeddings.T)
        total_time += (time.time() - start)
        
        # Top K
        k = 1000
        # Iterate over batch to extract top k
        for j, qid in enumerate(batch_qids):
            q_scores = scores[j]
            # Fast top-k using argpartition
            top_k_idx = np.argpartition(-q_scores, k)[:k]
            # sort exact top k
            top_k_sorted = top_k_idx[np.argsort(-q_scores[top_k_idx])]
            
            run_results[qid] = [
                (doc_ids[idx], float(q_scores[idx]))
                for idx in top_k_sorted
            ]

    logger.info(f"Retrieval finished. Batch Dot Product Time: {total_time:.2f}s")

    # 4. Save Run
    save_trec_run(run_results, args.output, "DenseBaseline")

    # 5. Evaluate
    qrels_data = load_qrels(args.qrels)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'recall_100', 'recall_1000', 'P_10'])
    metrics = evaluator.evaluate_run(run_results, qrels_data)
    
    print("\n=== DENSE BASELINE METRICS ===")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    main()

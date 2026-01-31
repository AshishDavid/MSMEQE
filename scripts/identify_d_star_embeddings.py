import os
import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(".")

from src.utils.lucene_utils import initialize_lucene
from src.retrieval.bm25_scorer import TokenBM25Scorer
from src.reranking.semantic_encoder import SemanticEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 0:
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(docid)
    return qrels

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--index-path", default="data/msmarco_index")
    parser.add_argument("--lucene-path", default="lucene_jars")
    parser.add_argument("--output", default="data/d_star_embeddings.pkl")
    args = parser.parse_args()

    # 1. Initialize Lucene
    if not initialize_lucene(args.lucene_path):
        logger.error("Failed to initialize Lucene")
        sys.exit(1)
        
    scorer = TokenBM25Scorer(args.index_path)
    encoder = SemanticEncoder()
    
    # 2. Load Qrels
    logger.info(f"Loading qrels from {args.qrels}...")
    qrels = load_qrels(args.qrels)
    
    # 3. Extract and Encode d*
    d_star_embeddings = {}
    missing_docs = 0
    
    # Use index searcher directly to get stored fields
    searcher = scorer.searcher
    
    logger.info("Extracting and encoding d* documents...")
    for qid, docids in tqdm(qrels.items()):
        # Use the first relevant document as d*
        d_star_id = docids[0]
        
        # Retrieve d* text from Lucene
        id_term = scorer.Term("id", str(d_star_id))
        id_query = scorer.TermQuery(id_term)
        hits = searcher.search(id_query, 1)
        
        if hits.totalHits.value() > 0:
            doc = searcher.storedFields().document(hits.scoreDocs[0].doc)
            doc_text = doc.get("contents")
            
            if doc_text:
                # Encode text
                emb = encoder.encode([doc_text])[0]
                d_star_embeddings[qid] = emb
            else:
                missing_docs += 1
        else:
            missing_docs += 1
            
    logger.info(f"Successfully encoded {len(d_star_embeddings)} d* embeddings.")
    if missing_docs > 0:
        logger.warning(f"Missing text for {missing_docs} documents.")
        
    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(d_star_embeddings, f)
    logger.info(f"Saved d* embeddings to {args.output}")

if __name__ == "__main__":
    main()

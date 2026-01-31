
import argparse
import logging
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json

# Add project root to path
sys.path.append(".")

from src.utils.file_utils import load_trec_run, save_trec_run
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-path", default="data/msmarco_index")
    
    args = parser.parse_args()
    
    logger.info("Initializing components...")
    
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        n_docs_rm3=30, # Enable RM3
        n_kb=0,
        n_emb=0,
        n_pseudo_docs=10 # Needed for RM3
    )
    
    if not hasattr(extractor, 'lucene_searcher'):
        extractor._init_lucene_stats()
        
    searcher = extractor.lucene_searcher
    
    run_results = defaultdict(list)
    
    logger.info("Starting RM3-Only retrieval...")
    
    with open(args.test_queries, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        qid = parts[0]
        query_text = parts[1]
        
        try:
            # 1. Extract Candidates (RM3 Only)
            candidates = extractor.extract_all_candidates(
                query_text=query_text,
                query_id=qid
            )
            
            # 2. Construct Expanded Query
            from jnius import autoclass
            Builder = autoclass('org.apache.lucene.search.BooleanQuery$Builder')
            TermQuery = autoclass('org.apache.lucene.search.TermQuery')
            Term = autoclass('org.apache.lucene.index.Term')
            Occur = autoclass('org.apache.lucene.search.BooleanClause$Occur')
            BoostQuery = autoclass('org.apache.lucene.search.BoostQuery')
            
            builder = Builder()
            
            # Add Original Query (Weight 1.0)
            orig_tokens = query_text.lower().split()
            for t in orig_tokens:
                    tq = TermQuery(Term("contents", t))
                    builder.add(BoostQuery(tq, 1.0), Occur.SHOULD)
                    
            # Add RM3 Candidates (Weight derived from RM3 score or fixed 0.5?)
            # RM3 candidates have `native_score` (prob).
            # Usually we use that.
            # Let's scale it.
            for cand in candidates:
                # cand.rm3_score is the probability.
                # In robust RM3, we usually interpolate.
                # Let's just use a simplified weight: 0.5 * score
                w = 0.5 * cand.rm3_score * 10 # Scale up a bit so it's visible? 
                # Or just fixed weight 0.2 like others to be comparable?
                # Better to use the score if available.
                # But to be fair with LLM/KB where we used fixed weight?
                # Actually LLM/KB confidence is also available.
                # Let's use fixed weight 0.2 for consistency with "Ablation".
                # Or stick to standard RM3 implementation?
                # The user wants to see which SOURCE is better.
                # If RM3 is tuned and others are not, it's unfair.
                # But RM3 comes with weights.
                # I'll use a fixed weight 0.2 for all candidates to measure "Candidate Quality" primarily.
                
                tq = TermQuery(Term("contents", cand.term))
                builder.add(BoostQuery(tq, 0.2), Occur.SHOULD)
                
            final_query = builder.build()
            
            # 3. Search
            docs = searcher.search(final_query, 1000)
            
            # 4. Store Results
            stored_fields = searcher.storedFields()
            
            for i, score_doc in enumerate(docs.scoreDocs):
                safe_doc = stored_fields.document(score_doc.doc)
                doc_id = safe_doc.get("id")
                run_results[qid].append((doc_id, score_doc.score))
                
        except Exception as e:
            logger.error(f"Error processing {qid}: {e}")
            continue

    save_trec_run(run_results, args.output, "MS-MEQE-RM3-Only")
    logger.info(f"Saved run to {args.output}")

if __name__ == "__main__":
    from src.utils.lucene_utils import initialize_lucene
    try:
        initialize_lucene(lucene_path="./lucene_jars")
    except:
        pass
    main()

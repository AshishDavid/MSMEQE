
import argparse
import logging
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(".")

from src.utils.file_utils import load_trec_run, save_trec_run, load_qrels
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder
from src.retrieval.evaluator import TRECEvaluator # Use correct import if available, else standard

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-path", default="data/msmarco_index")
    parser.add_argument("--test-queries", default="data/test_queries.tsv") # Fallback if we need text
    
    # We need a base run to get the list of queries (or qrels)
    # But usually we re-rank or re-retrieve.
    # For "Expansion", we usually take a query, expand it, and run a NEW retrieval.
    # However, MultiSourceCandidateExtractor just *extracts* terms.
    # We need to actually *execute* the retrieval with those terms.
    # Since we don't have a full retrieval loop script handy, let's verify if we can leverage
    # an existing retrieval script or build a simple one here.
    
    # Actually, `MultiSourceCandidateExtractor` assumes we have the query text.
    # We should iterate over the queries in `llm_candidates` (since it covers the test set).
    
    args = parser.parse_args()
    
    # 1. Load Data
    logger.info("Initializing components...")
    
    # We need Encoder for initialization but won't use it for LLM-only if we just trust the file
    # But the class requires it.
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        llm_candidates_path=args.llm_candidates,
        n_docs_rm3=0, # Disable RM3
        n_kb=0,       # Disable KB
        n_emb=0,      # Disable Emb
        n_pseudo_docs=1 # Must be > 0 for Lucene search call check 
                        # Wait, extract_all_candidates might need it for stats calculation.
                        # Let's keep it minimal.
    )
    
    # Load test queries to get text
    queries = {}
    with open(args.llm_candidates, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['qid'])
            # We assume the file has 'query_text' or similar?
            # Let's check the file format in a separate step if unsure, but typically yes.
            # If not, we might need a separate query file.
            pass

    # Let's assume we iterate the JSONL
    
    # We need to PERFORM RETRIEVAL with these terms.
    # The `extractor` just gives us `CandidateTerm` objects.
    # We need a `Retriever` or `Searcher`.
    # `MultiSourceCandidateExtractor` has `_perform_lucene_search` but that's for pseudo docs.
    # We need a proper retrieval function.
    
    # Let's look for `src/retrieval/retriever.py` or similar.
    # If not found, we use `extractor.lucene_searcher` directly.

    if not hasattr(extractor, 'lucene_searcher'):
        extractor._init_lucene_stats()
        
    searcher = extractor.lucene_searcher
    parser_cls = extractor.QueryParser
    analyzer = extractor.analyzer
    
    run_results = defaultdict(list)
    
    logger.info("Starting LLM-Only retrieval...")
    if len(extractor.llm_cache) > 0:
        first_key = list(extractor.llm_cache.keys())[0]
        logger.info(f"LLM Cache Sample Key: '{first_key}' (Type: {type(first_key)})")
    else:
        logger.warning("LLM Cache is EMPTY!")
    
    with open(args.llm_candidates, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            qid = str(data['qid'])
            query_text = data.get('query', '') or data.get('query_text', '')
            
            if not query_text:
                continue
                
            # 1. Extract Candidates (LLM Only)
            candidates = extractor.extract_all_candidates(
                query_text=query_text,
                query_id=qid
            )
            
            # 2. Construct Expanded Query
            from jnius import autoclass
            BooleanQuery = autoclass('org.apache.lucene.search.BooleanQuery')
            Builder = autoclass('org.apache.lucene.search.BooleanQuery$Builder')
            TermQuery = autoclass('org.apache.lucene.search.TermQuery')
            Term = autoclass('org.apache.lucene.index.Term')
            Occur = autoclass('org.apache.lucene.search.BooleanClause$Occur')
            BoostQuery = autoclass('org.apache.lucene.search.BoostQuery')
            QueryParser = autoclass('org.apache.lucene.queryparser.classic.QueryParser')
            
            # Use the analyzer from extractor (which is EnglishAnalyzer)
            analyzer = extractor.analyzer
            parser = QueryParser("contents", analyzer)
            
            builder = Builder()
            
            # Add Original Query (Weight 1.0)
            # Parse the whole query text to handle stemming/analysis correctly
            try:
                orig_query = parser.parse(query_text)
                builder.add(BoostQuery(orig_query, 1.0), Occur.SHOULD)
            except Exception as e:
                logger.warning(f"Failed to parse query '{query_text}': {e}")
                continue

            # Add LLM Candidates (Weight 0.2)
            for cand in candidates:
                # Analyze candidate term too!
                try:
                    # Candidates might be multi-word, so parse them
                    cand_query = parser.parse(cand.term)
                    builder.add(BoostQuery(cand_query, 0.2), Occur.SHOULD)
                except:
                    # Fallback for special chars
                    pass
                
            final_query = builder.build()
            
            # 3. Search
            docs = searcher.search(final_query, 1000)
            
            # 4. Store Results
            stored_fields = searcher.storedFields() # Get StoredFields instance outside loop if possible, or inside
            # In Lucene 10, searcher.storedFields() returns StoredFields which has .document(docId)
            
            for i, score_doc in enumerate(docs.scoreDocs):
                try:
                    safe_doc = searcher.storedFields().document(score_doc.doc)
                    
                    # Try 'docid' (binary)
                    f = safe_doc.getField("docid")
                    if f:
                        doc_id = f.binaryValue().utf8ToString()
                    else:
                        # Try 'id'
                        f = safe_doc.getField("id")
                        if f:
                            doc_id = f.stringValue()
                        else:
                            logger.warning(f"Doc {score_doc.doc} has no 'docid' or 'id' field")
                            doc_id = "UNKNOWN"
                            
                    run_results[qid].append((doc_id, score_doc.score))
                except Exception as e:
                    logger.error(f"Error extracting docid for doc {score_doc.doc}: {e}")
                
    # Save Run
    save_trec_run(run_results, args.output, "MS-MEQE-LLM-Only")
    logger.info(f"Saved run to {args.output}")

if __name__ == "__main__":
    from collections import defaultdict
    import json
    
    # Initialize lucene (needed for Extractor)
    from src.utils.lucene_utils import initialize_lucene
    try:
        initialize_lucene(lucene_path="./lucene_jars")
    except:
        pass # Might be already initialized
        
    main()

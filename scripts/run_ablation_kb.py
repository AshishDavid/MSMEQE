
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
from src.expansion.kb_expansion import KBCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wat-output", required=True)
    parser.add_argument("--test-queries", required=True) # Needs to be something iterable with qid
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-path", default="data/msmarco_index")
    
    args = parser.parse_args()
    
    logger.info("Initializing components...")
    
    # Encoder needed for initialization
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize KB Extractor
    kb_extractor = KBCandidateExtractor(
        wat_output_path=args.wat_output,
        max_candidates_per_query=30,
        min_entity_confidence=0.1
    )
    
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        n_docs_rm3=0, # Disable RM3
        n_kb=30,      # Enable KB
        n_emb=0,      # Disable Emb
        n_pseudo_docs=1  # Must be > 0 
    )
    
    # Ensure Lucene initialized
    if not hasattr(extractor, 'lucene_searcher'):
        extractor._init_lucene_stats()
        
    searcher = extractor.lucene_searcher
    
    run_results = defaultdict(list)
    
    logger.info("Starting KB-Only retrieval...")
    
    # Load Queries
    queries_map = {}
    with open(args.test_queries, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries_map[parts[0]] = parts[1]
                
    with open(args.wat_output, 'r') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                qid = str(data.get('doc_id'))
                
                # Get text from map
                query_text = queries_map.get(qid, "")
                
                if not query_text:
                    # logger.warning(f"Query text not found for {qid}")
                    continue
                
                # 1. Extract Candidates (KB Only via pipeline)
                # Pipeline uses kb_extractor.extract_candidates_with_metadata(query_text, qid)
                # KBCandidateExtractor uses self.wat_entities[qid] (preloaded)
                
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
                QueryParser = autoclass('org.apache.lucene.queryparser.classic.QueryParser')
                
                analyzer = extractor.analyzer
                parser = QueryParser("contents", analyzer)
                
                builder = Builder()
                
                # Add Original Query (Weight 1.0)
                try:
                    orig_query = parser.parse(query_text)
                    builder.add(BoostQuery(orig_query, 1.0), Occur.SHOULD)
                except:
                    continue
                     
                # Add KB Candidates (Weight 0.5)
                for cand in candidates:
                    try:
                        cand_query = parser.parse(cand.term)
                        builder.add(BoostQuery(cand_query, 0.5), Occur.SHOULD)
                    except:
                        pass
                    
                final_query = builder.build()
                
                # 3. Search
                docs = searcher.search(final_query, 1000)
                
                # 4. Store Results
                stored_fields = searcher.storedFields() # Get StoredFields
                
                for i, score_doc in enumerate(docs.scoreDocs):
                    try:
                        safe_doc = stored_fields.document(score_doc.doc)
                        f = safe_doc.getField("docid")
                        if f:
                            doc_id = f.binaryValue().utf8ToString()
                        else:
                            f = safe_doc.getField("id")
                            if f:
                                doc_id = f.stringValue()
                            else:
                                logger.warning(f"Doc {score_doc.doc} has no 'docid' or 'id'")
                                doc_id = "UNKNOWN"
                        run_results[qid].append((doc_id, score_doc.score))
                    except Exception as e:
                        logger.error(f"Error extracting docid: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing {qid}: {e}")
                continue
                
    # Save Run
    save_trec_run(run_results, args.output, "MS-MEQE-KB-Only")
    logger.info(f"Saved run to {args.output}")

if __name__ == "__main__":
    from src.utils.lucene_utils import initialize_lucene
    try:
        initialize_lucene(lucene_path="./lucene_jars")
    except:
        pass
    main()

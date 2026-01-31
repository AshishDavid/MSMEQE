
import argparse
import logging
import json
from tqdm import tqdm
from pathlib import Path
from src.expansion.rm_expansion import LuceneRM3Scorer
from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_qrels, save_trec_run
import jnius_config
import os

# Ensure JVM config
if not jnius_config.vm_running:
    try:
        jnius_config.add_options('-Djava.awt.headless=true')
        jnius_config.add_options(f"-Djava.library.path={os.environ.get('LD_LIBRARY_PATH')}")
        lucene_jars = list(Path("lucene_jars").glob("*.jar"))
        classpath = [str(p.absolute()) for p in lucene_jars]
        jnius_config.set_classpath('.', *classpath)
    except Exception as e:
        print(f"Warning on JVM init: {e}")

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Sparse RM3 Baseline")
    parser.add_argument("--index", type=str, required=True, help="Path to Lucene index")
    parser.add_argument("--topics", type=str, required=True, help="Path to queries TSV")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--output", type=str, required=True, help="Output run file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    logger.info(f"Loading RM3 Scorer with index: {args.index}")
    rm3_scorer = LuceneRM3Scorer(index_dir=args.index)
    
    # Load Doc IDs mapping
    logger.info("Loading doc_ids.json mapping...")
    with open(os.path.join(args.index, "doc_ids.json"), "r") as f:
        doc_ids_map = json.load(f)

    # 2. Load Queries
    queries = {}
    with open(args.topics, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    
    logger.info(f"Loaded {len(queries)} queries")

    # 3. Retrieval Loop
    logger.info("Starting sparse RM3 retrieval...")
    run_results = {}
    
    for qid, qtext in tqdm(queries.items()):
        # LuceneRM3Scorer typically returns expansion terms. 
        # We need to hack it or use a proper searcher to get ranked documents.
        # Actually, let's use the internal searcher if possible, or Pyserini if installed.
        # Looking at src/expansion/rm_expansion.py, it uses a generic 'search' method.
        # We'll use get_expansion_terms, which internally runs a search, but we want the search RESULTS.
        
        # Checking LuceneRM3Scorer implementation details...
        # It seems designed for extracting terms, not returning a full run.
        # But we can access self.searcher.search(query, k).
        pass
        # WAIT: LuceneRM3Scorer wraps a Lucene index. We can write a direct search loop here.
    
    # REWRITE: We need a direct Lucene searcher, not just the RM3 expansion logic.
    # The project seems to rely on `jnius` interfacing with custom Java classes or standard Lucene.
    # Let's use the SimpleSearcher approach if available, or build valid Lucene query.
    
    # Access the searcher from the initialized rm3_scorer to reuse JVM
    searcher = rm3_scorer.searcher
    analyzer = rm3_scorer.analyzer
    
    # Use classes already loaded by the scorer helper
    QueryParser = rm3_scorer.JQueryParser
    
    # We need a parser instance
    parser = QueryParser("contents", analyzer)

    for qid, qtext in tqdm(queries.items()):
        try:
            lucene_query = parser.parse(qtext)
            
            # 1. Standard Search (BM25)
            # We perform a standard search using the existing Searcher
            top_docs = searcher.search(lucene_query, 1000)
            hits = top_docs.scoreDocs
            
            run_results[qid] = []
            for i, hit in enumerate(hits):
                # Retrieve doc ID from binary stored field
                try:
                    doc_fields = searcher.storedFields().document(hit.doc)
                    bin_ref = doc_fields.getField("docid").binaryValue()
                    doc_id = bin_ref.utf8ToString()
                except Exception as e:
                    logger.error(f"Error retrieving ID for hit {hit.doc}: {e}")
                    continue
                    
                score = hit.score
                run_results[qid].append((doc_id, score))
                
        except Exception as e:
            logger.error(f"Error processing query {qid}: {e}")
            run_results[qid] = []

    # 4. Save Run
    save_trec_run(run_results, args.output, "BM25Baseline")

    # 5. Evaluate
    qrels_data = load_qrels(args.qrels)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'recall_100', 'recall_1000', 'P_10'])
    metrics = evaluator.evaluate_run(run_results, qrels_data)
    
    print("\n=== SPARSE BM25 METRICS ===")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    main()


import os
import sys
import logging
from tqdm import tqdm
from src.utils.lucene_utils import initialize_lucene, get_lucene_classes
from src.retrieval.evaluator import TRECEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_queries(topics_file):
    queries = {}
    with open(topics_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Assuming TSV: qid \t query
                queries[parts[0]] = parts[1]
    return queries

def main():
    index_path = "data/robust04_index"
    topics_file = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_queries.tsv"
    qrels_file = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_qrels.tsv"
    lucene_path = "lucene_jars"
    
    # Init Lucene
    if not initialize_lucene(lucene_path):
        sys.exit(1)
        
    classes = get_lucene_classes()
    DirectoryReader = classes['IndexReader'] # Mapping generic name, but assumes IndexReader-like
    FSDirectory = classes['FSDirectory']
    JavaPaths = classes['JavaPaths']
    IndexSearcher = classes['IndexSearcher']
    QueryParser = classes['QueryParser']
    EnglishAnalyzer = classes['EnglishAnalyzer']
    BM25Similarity = classes['BM25Similarity']
    MultiFieldQueryParser = classes.get('MultiFieldQueryParser') # Optional

    logger.info(f"Opening index: {index_path}")
    directory = FSDirectory.open(JavaPaths.get(index_path))
    reader = DirectoryReader.open(directory)
    
    # INSPECT INDEX STATS
    num_docs = reader.numDocs()
    logger.info(f"Index contains {num_docs} documents.")
    
    # Check a random document content
    if num_docs > 0:
        sample_doc = reader.storedFields().document(0)
        logger.info(f"Sample Doc Fields: {[f.name() for f in sample_doc.getFields()]}")
        logger.info(f"Sample Doc Content (first 100 chars): {sample_doc.get('contents')[:100] if sample_doc.get('contents') else 'NONE'}")
        
    # SEARCH
    # Try passing context explicitly to avoid ambiguity
    searcher = IndexSearcher(reader.getContext())
    searcher.setSimilarity(BM25Similarity(1.2, 0.75))
    analyzer = EnglishAnalyzer()
    parser = QueryParser("contents", analyzer)
    
    queries = load_queries(topics_file)
    logger.info(f"Loaded {len(queries)} queries")
    
    run_data = {}
    
    for qid, qtext in tqdm(list(queries.items())[:50]): # Limit to 50 for speed debugging
        try:
            qtext_safe = QueryParser.escape(qtext)
            query = parser.parse(qtext_safe)
            top_docs = searcher.search(query, 1000)
            
            run_data[qid] = []
            for score_doc in top_docs.scoreDocs:
                # Retrieve doc ID
                stored_doc = searcher.storedFields().document(score_doc.doc)
                doc_id = stored_doc.get("id") or stored_doc.get("docid")
                run_data[qid].append((doc_id, score_doc.score))
                
        except Exception as e:
            logger.error(f"Error querying {qid}: {e}")
            
    # Save partial run
    output_run = "runs/debug_bm25_robust_fixed.txt"
    with open(output_run, 'w') as f:
        for qid, docs in run_data.items():
            for rank, (docid, score) in enumerate(docs, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} DebugBM25\n")
    
    # Evaluate if we have Qrels
    from src.utils.file_utils import load_qrels
    qrels = load_qrels(qrels_file)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'P_10'])
    # Filter qrels to just those queries we ran? No, evaluator handles missing fine usually
    metrics = evaluator.evaluate_run(run_data, qrels)
    print("Baseline Fixed Metrics (First 50):", metrics)

if __name__ == "__main__":
    main()

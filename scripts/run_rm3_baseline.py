
import argparse
import logging
import json
import os
import sys
from tqdm import tqdm
from pathlib import Path
from src.expansion.rm_expansion import LuceneRM3Scorer
from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_qrels, save_trec_run
import jnius_config

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
    parser = argparse.ArgumentParser(description="Run RM3 Baseline (BM25 + Expansion)")
    parser.add_argument("--index", type=str, required=True, help="Path to Lucene index directory")
    parser.add_argument("--topics", type=str, required=True, help="Path to queries TSV")
    parser.add_argument("--qrels", type=str, required=True, help="Path to qrels file")
    parser.add_argument("--output", type=str, required=True, help="Output run file")
    parser.add_argument("--fb-terms", type=int, default=10, help="Number of feedback terms")
    parser.add_argument("--fb-docs", type=int, default=10, help="Number of feedback docs")
    parser.add_argument("--original-query-weight", type=float, default=0.5, help="Weight of original query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # 1. Initialize RM3 Scorer
    logger.info(f"Loading RM3 Scorer with index: {args.index}")
    rm3_scorer = LuceneRM3Scorer(
        index_dir=args.index,
        orig_query_weight=args.original_query_weight
    )

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

    # 3. RM3 Retrieval Loop
    logger.info("Starting RM3 retrieval...")
    run_results = {}
    
    # Access internal searcher and query builder helpers
    searcher = rm3_scorer.searcher
    analyzer = rm3_scorer.analyzer
    BooleanQueryBuilder = rm3_scorer.JBooleanQueryBuilder
    TermQuery = rm3_scorer.JTermQuery
    Term = rm3_scorer.JTerm
    BooleanClauseOccur = rm3_scorer.JBooleanClauseOccur
    BoostQuery = getattr(rm3_scorer, 'JBoostQuery', None) # Might need to load this class

    # We need BoostQuery to apply weights to terms in Lucene
    from jnius import autoclass
    try:
        JBoostQuery = autoclass("org.apache.lucene.search.BoostQuery")
    except:
        JBoostQuery = None
        logger.warning("Could not load BoostQuery, weights might be ignored if not handled differently.")

    for qid, qtext in tqdm(queries.items()):
        try:
            # Step A: Get Expansion Terms
            # This internally runs BM25 first, then computes RM3 terms
            expansion_terms = rm3_scorer.expand(
                query_str=qtext,
                n_docs=args.fb_docs,
                n_terms=args.fb_terms,
                use_rm3=True 
            )
            
            # Step B: Build Expanded Query
            # We will build a BooleanQuery with SHOULD clauses for all terms
            builder = BooleanQueryBuilder()
            
            # Add original query terms (simplified: just parse again or rely on RM3 returning them?)
            # RM3 returns interpolated terms, so it includes original terms if they are high probability.
            # But standard practice is to query with the top K terms from the RM3 distribution.
            
            if not expansion_terms:
                # Fallback to simple parse if no expansion
                # But we need to use the analyzer-aware parser
                pass # Logic below handles empty list essentially
            
            # To be safe and robust:
            # If RM3 returned terms, use them as the query.
            # They are already weighted by (1-alpha)*RM1 + alpha*QM
            
            for term_text, weight in expansion_terms:
                tq = TermQuery(Term("contents", term_text))
                if JBoostQuery:
                    bq = JBoostQuery(tq, float(weight))
                    builder.add(bq, BooleanClauseOccur.SHOULD)
                else:
                    # Fallback if no boost query (unlikely in modern Lucene)
                    builder.add(tq, BooleanClauseOccur.SHOULD)
            
            final_query = builder.build()
            
            # Step C: Execute Final Search
            # If final query is empty (e.g. stop words only), fallback to original
            # But expand() usually handles empty checks.
            
            # If expanded query has no clauses, use original text
            if len(expansion_terms) == 0:
                # Use standard parser from scorer
                # Fallback to BM25 basically
                parser = rm3_scorer.JQueryParser("contents", analyzer)
                final_query = parser.parse(qtext)

            top_docs = searcher.search(final_query, 1000)
            hits = top_docs.scoreDocs
            
            run_results[qid] = []
            for i, hit in enumerate(hits):
                # Retrieve doc ID from binary stored field
                try:
                    doc_fields = searcher.storedFields().document(hit.doc)
                    bin_ref = doc_fields.getField("docid").binaryValue()
                    doc_id = bin_ref.utf8ToString()
                except Exception as e:
                    # Try fallback to map if binary fails? No, stick to binary as proven.
                    continue
                
                score = hit.score
                run_results[qid].append((doc_id, score))

        except Exception as e:
            logger.error(f"Error processing query {qid}: {e}")
            run_results[qid] = []

    # 4. Save Run
    save_trec_run(run_results, args.output, "RM3Baseline")

    # 5. Evaluate
    qrels_data = load_qrels(args.qrels)
    evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'recall_100', 'recall_1000'])
    metrics = evaluator.evaluate_run(run_results, qrels_data)
    
    print("\n=== RM3 BASELINE METRICS ===")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("==============================\n")

if __name__ == "__main__":
    main()

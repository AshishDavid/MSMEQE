
import ir_datasets
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_queries(queries, output_tsv):
    logger.info(f"Saving queries to {output_tsv}...")
    with open(output_tsv, 'w') as f_tsv:
        for query in queries:
            text = query.text.replace('\t', ' ').replace('\n', ' ').strip()
            f_tsv.write(f"{query.query_id}\t{text}\n")

def save_qrels(qrels, output_file):
    logger.info(f"Saving qrels to {output_file}...")
    with open(output_file, 'w') as f:
        for qrel in qrels:
            # TREC format: qid Q0 docid rel
            # BEIR qrels usually imply test/test set?
            # actually beir/nq has train/dev/test splits in qrels?
            # checking qrels properties might be needed.
            # but ir_datasets usually iterates all qrels.
            f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")

def main():
    dataset = ir_datasets.load("beir/nq")
    data_dir = Path("data/nq")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    save_queries(dataset.queries_iter(), data_dir / "queries.tsv")
    save_qrels(dataset.qrels_iter(), data_dir / "qrels.txt")
    print("Done")

if __name__ == "__main__":
    main()

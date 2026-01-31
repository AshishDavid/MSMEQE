import ir_datasets
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_queries(queries, output_tsv, output_json=None):
    logger.info(f"Saving queries to {output_tsv}...")
    with open(output_tsv, 'w') as f_tsv:
        if output_json:
            f_json = open(output_json, 'w')
            queries_dict = {}
        
        for query in queries:
            # Clean query text just in case (remove tabs/newlines)
            text = query.text.replace('\t', ' ').replace('\n', ' ').strip()
            # TSV: qid \t text
            f_tsv.write(f"{query.query_id}\t{text}\n")
            
            if output_json:
                queries_dict[query.query_id] = text
        
        if output_json:
            json.dump(queries_dict, f_json)
            f_json.close()
            logger.info(f"Saved JSON queries to {output_json}")

def save_qrels(qrels, output_file):
    logger.info(f"Saving qrels to {output_file}...")
    with open(output_file, 'w') as f:
        for qrel in qrels:
            # TREC format: qid Q0 docid rel
            f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")

def download_dataset(dataset_id, name):
    data_dir = Path("data") / name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {name} ({dataset_id})...")
    
    try:
        dataset = ir_datasets.load(dataset_id)
        
        # Save Queries
        save_queries(
            dataset.queries_iter(), 
            data_dir / "queries.tsv",
            data_dir / "queries.json"
        )
        
        # Save Qrels
        save_qrels(
            dataset.qrels_iter(), 
            data_dir / "qrels.txt"
        )
        logger.info(f"Successfully processed {name}")
        
    except Exception as e:
        logger.error(f"Failed to process {name}: {e}")
        if "trec-robust04" in dataset_id:
             logger.warning("Robust04 failed likely due to missing local data. Please ensure the data is available locally as per ir_datasets requirements.")

def main():
    # TREC DL 2019
    download_dataset("msmarco-passage/trec-dl-2019/judged", "trec_dl_2019")
    
    # TREC DL 2020
    download_dataset("msmarco-passage/trec-dl-2020/judged", "trec_dl_2020")
    
    # Natural Questions (NQ)
    # Using beir/nq as it is a common standard, but verifying if another ID is better
    # ir_datasets has 'beir/nq' which links to BEIR benchmark version of NQ
    download_dataset("beir/nq", "nq")
    
    # TREC Robust04
    # Note: this usually requires local configuration
    download_dataset("trec-robust04", "robust04")
    
    logger.info("All datasets processed.")

if __name__ == "__main__":
    main()

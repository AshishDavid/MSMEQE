import ir_datasets
import json
import os
from pathlib import Path

def save_queries(queries, output_tsv, output_json=None):
    print(f"Saving queries to {output_tsv}...")
    with open(output_tsv, 'w') as f_tsv:
        if output_json:
            f_json = open(output_json, 'w')
            queries_dict = {}
        
        for query in queries:
            # TSV: qid \t text
            f_tsv.write(f"{query.query_id}\t{query.text}\n")
            
            if output_json:
                queries_dict[query.query_id] = query.text
        
        if output_json:
            json.dump(queries_dict, f_json)
            f_json.close()
            print(f"Saved JSON queries to {output_json}")

def save_qrels(qrels, output_file):
    print(f"Saving qrels to {output_file}...")
    with open(output_file, 'w') as f:
        for qrel in qrels:
            # TREC format: qid Q0 docid rel
            f.write(f"{qrel.query_id} Q0 {qrel.doc_id} {qrel.relevance}\n")

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Train Data
    print("Loading msmarco-passage/train...")
    dataset = ir_datasets.load("msmarco-passage/train")
    
    save_queries(dataset.queries_iter(), 
                 data_dir / "train_queries.tsv", 
                 data_dir / "train_queries.json")
    
    save_qrels(dataset.qrels_iter(), data_dir / "train_qrels.txt")

    # Dev Data (using as test)
    print("Loading msmarco-passage/dev/small...")
    dataset_dev = ir_datasets.load("msmarco-passage/dev/small")
    
    save_queries(dataset_dev.queries_iter(), 
                 data_dir / "test_queries.tsv",
                 data_dir / "test_queries.json")
                 
    save_qrels(dataset_dev.qrels_iter(), data_dir / "test_qrels.txt")
    
    print("Done!")

if __name__ == "__main__":
    main()

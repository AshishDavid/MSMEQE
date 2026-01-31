
import json
import csv
import sys
from pathlib import Path

def inspect_jsonl(filepath):
    print(f"--- Inspecting {filepath} ---")
    ids = set()
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            try:
                doc = json.loads(line)
                # Try standard Pyserini fields
                did = doc.get('id') or doc.get('docid')
                if did:
                    ids.add(did)
                count += 1
            except:
                pass
            if count > 0 and count % 100000 == 0:
                print(f"Read {count} docs...")
    print(f"Total Docs: {count}")
    print(f"Unique IDs: {len(ids)}")
    return ids

def inspect_qrels(filepath):
    print(f"\n--- Inspecting Qrels {filepath} ---")
    doc_ids = set()
    rows = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split() # Splits on any whitespace
            if len(parts) >= 3:
                # TREC format: qid 0 docid rel
                doc_ids.add(parts[2])
                rows += 1
    print(f"Total Qrels Rows: {rows}")
    print(f"Unique Referenced Docs: {len(doc_ids)}")
    return doc_ids

def main():
    qrels_path = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_qrels.tsv"
    jsonl_pyserini = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/robust04_pyserini.jsonl"
    jsonl_plain = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/robust04.jsonl"
    
    qrel_docs = inspect_qrels(qrels_path)
    
    print("\n>>> Checking robust04_pyserini.jsonl (USED IN RUN)")
    pyserini_docs = inspect_jsonl(jsonl_pyserini)
    
    overlap_pyserini = qrel_docs.intersection(pyserini_docs)
    print(f"Overlap Qrels vs Pyserini: {len(overlap_pyserini)} / {len(qrel_docs)} ({len(overlap_pyserini)/len(qrel_docs):.2%})")
    
    missing = list(qrel_docs - pyserini_docs)[:5]
    print(f"Sample Missing in Pyserini: {missing}")

    print("\n>>> Checking robust04.jsonl (ALTERNATIVE)")
    plain_docs = inspect_jsonl(jsonl_plain)
    
    overlap_plain = qrel_docs.intersection(plain_docs)
    print(f"Overlap Qrels vs Plain: {len(overlap_plain)} / {len(qrel_docs)} ({len(overlap_plain)/len(qrel_docs):.2%})")

if __name__ == "__main__":
    main()

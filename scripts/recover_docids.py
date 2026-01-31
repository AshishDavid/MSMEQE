
import json
import logging
from tqdm import tqdm
import sys

def main():
    input_file = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/robust04_pyserini.jsonl"
    output_file = "data/robust04_index/doc_ids.json"
    
    doc_ids = []
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            try:
                doc = json.loads(line)
                # Logic from precompute matching
                doc_id = doc.get('id') or doc.get('docid') or doc.get('_id')
                doc_text = doc.get('contents') or doc.get('text') or doc.get('body')
                
                if doc_id and doc_text:
                    doc_ids.append(doc_id)
            except Exception:
                continue
                
    print(f"Found {len(doc_ids)} docs.")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(doc_ids, f)
        
    print("Done recovery.")

if __name__ == "__main__":
    main()

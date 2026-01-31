
import json
from pathlib import Path

def create_full_wat_input(topics_file, output_file):
    print(f"Loading queries from {topics_file}...")
    with open(topics_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid = parts[0]
                title = parts[1]
                desc = parts[2]
                # Combine Title and Description for enrichment
                full_text = f"{title}. {desc}"
                entry = {
                    "entity_id": qid,
                    "wikipedia_lead_text": full_text
                }
                f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(lines)} entries to {output_file}")

if __name__ == "__main__":
    create_full_wat_input("/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_queries.tsv", "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/data/robust04_wat_input_full.jsonl")


import json
import argparse
from pathlib import Path

def merge_kb_candidates(input_files, output_file):
    merged_data = {}
    
    for file_path in input_files:
        print(f"Loading {file_path}...")
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = str(item['qid'])
                # If QID already exists, we prefer the one from a newer file in the list
                merged_data[qid] = item
                
    print(f"Total unique queries in merged KB: {len(merged_data)}")
    
    with open(output_file, 'w') as f:
        # Sort by QID to keep it clean
        for qid in sorted(merged_data.keys()):
            f.write(json.dumps(merged_data[qid]) + "\n")
            
    print(f"Saved merged KB candidates to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="List of JSONL files to merge")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()
    merge_kb_candidates(args.inputs, args.output)

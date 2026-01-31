
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(".")
from src.utils.file_utils import load_trec_run, save_trec_run

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_best_run.py <labels_jsonl> <output_run>")
        sys.exit(1)
        
    labels_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 1. Load labels and best run info
    print("Loading best run info...")
    qid_to_best_run = {}
    with open(labels_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            qid_to_best_run[data['qid']] = data['best_run']
            
    # 2. Get unique run paths to load
    run_paths = set(qid_to_best_run.values())
    loaded_runs = {}
    for path in run_paths:
        if path == "none": continue
        print(f"Loading {path}...")
        loaded_runs[path] = load_trec_run(path)
        
    # 3. Assemble best run
    print("Assembling best run...")
    final_run = {}
    for qid, best_path in qid_to_best_run.items():
        if best_path == "none" or best_path not in loaded_runs:
            # This shouldn't happen for the queries we processed
            continue
            
        final_run[qid] = loaded_runs[best_path][qid]
        
    # 4. Save
    save_trec_run(final_run, output_path, "MS-MEQE-Best-Source")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

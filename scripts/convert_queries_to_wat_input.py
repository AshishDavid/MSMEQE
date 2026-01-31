
import json
import argparse
from pathlib import Path

def convert_to_wat_input(input_file, output_file):
    print(f"Converting {input_file} to {output_file}...")
    
    queries = {}
    path = Path(input_file)
    
    # Load queries
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                queries = data
            elif isinstance(data, list):
                queries = {str(i): q for i, q in enumerate(data)}
    elif path.suffix == '.tsv':
         with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
    
    # Write in WAT input format
    # {"entity_id": "qid", "wikipedia_lead_text": "query text"}
    with open(output_file, 'w') as f:
        for qid, text in queries.items():
            entry = {
                "entity_id": str(qid),
                "wikipedia_lead_text": text
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"Converted {len(queries)} queries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_to_wat_input(args.input, args.output)

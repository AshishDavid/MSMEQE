
#!/usr/bin/env python3
# scripts/precompute_llm_candidates.py

"""
Precompute LLM-MEQE candidates for Train and Test sets.
Uses Llama-3.1-8B-Instruct to generate entity-rich expansions.
"""

import argparse
import logging
import json
import os
from tqdm import tqdm
from src.expansion.llm_generation import LLMCandidateGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def load_queries(path):
    queries = {}
    with open(path, 'r') as f:
        # Check file extension
        if path.endswith('.json'):
            data = json.load(f)
            # data could be dict or list
            if isinstance(data, dict):
                queries = data
            elif isinstance(data, list):
                # assume list of objects? usually dict qid->text
                pass 
        else:
            # Assume TSV
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
    return queries

def process_file(generator, input_path, output_path, limit=None):
    if os.path.exists(output_path):
        logger.info(f"Output {output_path} already exists. Skipping.")
        return

    logger.info(f"Loading queries from {input_path}")
    queries = load_queries(input_path)
    logger.info(f"Found {len(queries)} queries.")

    logger.info(f"Generating candidates to {output_path}")
    
    query_items = list(queries.items())
    if limit:
        logger.info(f"Limiting to first {limit} queries.")
        query_items = query_items[:limit]

    with open(output_path, 'w') as f_out:
        for qid, text in tqdm(query_items):
            try:
                rich_text, raw_data = generator.generate_candidates(text)
                
                record = {
                    "qid": qid,
                    "query": text,
                    "llm_text": rich_text,
                    "llm_data": raw_data
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush() # flush for safety
            except Exception as e:
                logger.error(f"Error processing qid {qid}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-queries", default="data/train_queries.json") # or tsv
    parser.add_argument("--test-queries", default="data/test_queries.tsv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--limit-train", type=int, default=None, help="Limit number of training queries")
    parser.add_argument("--limit-test", type=int, default=None, help="Limit number of test queries")
    args = parser.parse_args()

    # Initialize generator once (heavy load)
    # Note: user might want to run this in screen if it takes long.
    # We will assume standard run for now.
    generator = LLMCandidateGenerator()

    # Train
    train_out = os.path.join(args.output_dir, "llm_candidates_train.jsonl")
    # Handle train queries which might be TSV or JSON in this repo
    # Based on README: "data/train_queries.json"
    process_file(generator, args.train_queries, train_out, limit=args.limit_train)

    # Test
    test_out = os.path.join(args.output_dir, "llm_candidates_test.jsonl")
    process_file(generator, args.test_queries, test_out, limit=args.limit_test)

if __name__ == "__main__":
    main()

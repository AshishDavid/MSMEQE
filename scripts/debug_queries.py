
from pathlib import Path

def load_queries_from_file(topics_file):
    queries = {}
    with open(topics_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, qtext = parts[0], parts[1]
                queries[qid] = qtext
    return queries

if __name__ == "__main__":
    f = "/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_queries.tsv"
    qs = load_queries_from_file(f)
    print(f"Loaded {len(qs)} queries.")
    for qid, qtext in list(qs.items())[:5]:
        print(f"[{qid}] '{qtext}'")


import requests
import os
import json

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Done.")

def convert_beir_qrels_to_msmarco(input_path, output_path):
    print(f"Converting {input_path} to MS MARCO format...")
    # This is tricky because BEIR might not share the same doc IDs.
    # However, 'nq-test' in BEIR uses Wikipedia titles/text.
    # MS MARCO uses integer IDs.
    # We really need the NQ-to-MSMARCO mapping if we want to use the MS MARCO index.
    # Fortunately, the 'rocketqa' dataset or similar usually provides this.
    pass

def main():
    # The BEIR qrels (doc0, doc1) are internal to BEIR's subset.
    # We need the Qrels that link NQ queries to MS MARCO Passage IDs (PID).
    # This dataset is commonly used in 'Dense Passage Retrieval' papers on MS MARCO.
    
    # URL for NQ qrels mapped to MS MARCO PIDs
    # This is often provided by the RocketQA or similar repos.
    # Alternative: Use the 'msmarco-passage/test/nq' from ir_datasets if it exists?
    # Checked ir_datasets: 'beir/nq' is the only NQ.
    
    # Let's try downloading from a known reliable source for NQ-MSMARCO qrels.
    # E.g. https://huggingface.co/datasets/Tevatron/msmarco-passage-nq/resolve/main/qrels.test.tsv
    
    url = "https://huggingface.co/datasets/Tevatron/msmarco-passage-nq/resolve/main/qrels.test.tsv?download=true"
    output_dir = "data/nq"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/qrels_msmarco.txt"
    
    print(f"Downloading NQ-MSMARCO qrels from {url}...")
    response = requests.get(url)
    
    with open(output_path, 'w') as f:
        # Convert TSV (qid, pid, score) to TREC qrels format (qid 0 pid 1)
        # Tevatron format: qid \t docid \t score
        count = 0
        for line in response.text.splitlines():
            if not line.strip(): continue
            parts = line.split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                docid = parts[1]
                # Write in TREC format: qid 0 docid 1
                f.write(f"{qid} 0 {docid} 1\n")
                count += 1
    
    print(f"Saved {count} qrels to {output_path}")

if __name__ == "__main__":
    main()

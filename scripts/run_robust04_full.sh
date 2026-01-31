#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
set -e

ROBUST_DOCS="/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/robust04_pyserini.jsonl"
ROBUST_QUERIES="/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_queries.tsv"
ROBUST_QRELS="/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/trec_robust04_qrels.tsv"
INDEX_DIR="data/robust04_index"

echo "========================================================"
echo "Robust04 Evaluation Pipeline"
echo "========================================================"

# 1. Build Lucene Index
# Check if valid Lucene index exists (look for segments file)
if [ -z "$(ls -A $INDEX_DIR/segments* 2>/dev/null)" ]; then
    echo "[1/3] Indexing Robust04..."
    # using --docid-field id --text-field contents
    python scripts/index_collection.py \
        --input "$ROBUST_DOCS" \
        --format jsonl \
        --docid-field id \
        --text-field contents \
        --output "$INDEX_DIR" \
        --lucene-path lucene_jars \
        --analyzer english \
        --ram-buffer 4096 \
        --commit-every 100000
else
    echo "[1/3] Index already exists at $INDEX_DIR"
fi

# 2. Precompute Embeddings
if [ ! -f "$INDEX_DIR/doc_embeddings.npy" ]; then
    echo "[2/3] Precomputing Document Embeddings..."
    python scripts/precompute_doc_embeddings.py \
        --input-file "$ROBUST_DOCS" \
        --output-dir "$INDEX_DIR" \
        --batch-size 1024
else
    echo "[2/3] Embeddings already exist at $INDEX_DIR"
fi

# 3. Run Evaluation
echo "[3/3] Running MS-MEQE Evaluation..."
python scripts/run_msmeqe_evaluation.py \
    --index "$INDEX_DIR" \
    --topics "$ROBUST_QUERIES" \
    --qrels "$ROBUST_QRELS" \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --budget-model models/budget_model.pkl \
    --emb-vocab data/vocab_embeddings.pkl \
    --output runs/msmeqe.robust04.txt \
    --run-name MS-MEQE-Robust04 \
    --lucene-path lucene_jars \
    --collection-size 528155

echo "Done!"

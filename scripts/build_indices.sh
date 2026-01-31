#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
set -e

# Ensure data directory exists
mkdir -p data

echo "========================================================"
echo "Building Indices for NQ and Robust04"
echo "========================================================"

# 1. Natural Questions (BEIR)
if [ ! -d "data/nq_index" ]; then
    echo "[1/2] Indexing Natural Questions (beir/nq)..."
    # Using beir/nq corpus
    python scripts/index_collection.py \
        --collection beir/nq \
        --output data/nq_index \
        --lucene-path lucene_jars \
        --analyzer english \
        --ram-buffer 4096 || echo "NQ Indexing failed (possibly missing data)"
else
    echo "[1/2] NQ Index already exists."
fi

# 2. TREC Robust04
if [ ! -d "data/robust04_index" ]; then
    echo "[2/2] Indexing TREC Robust04..."
    # Using trec-robust04 corpus
    python scripts/index_collection.py \
        --collection trec-robust04 \
        --output data/robust04_index \
        --lucene-path lucene_jars \
        --analyzer english \
        --ram-buffer 4096 || echo "Robust04 Indexing failed (likely missing local data)"
else
    echo "[2/2] Robust04 Index already exists."
fi

echo "========================================================"
echo "Indexing Attempts Complete"
echo "========================================================"

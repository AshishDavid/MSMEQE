#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
set -e

echo "Precomputing document embeddings for NQ..."
python scripts/precompute_doc_embeddings.py \
    --collection beir/nq \
    --output-dir data/nq_index \
    --batch-size 1024

echo "Running NQ Evaluation..."
./scripts/run_nq_eval.sh

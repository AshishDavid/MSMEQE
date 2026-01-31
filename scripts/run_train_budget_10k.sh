#!/bin/bash
# MS-MEQE Budget Model Training (10k)
# Wrapped in script for easier screen execution

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

# Setup Environment
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Using Java at: $JAVA_HOME"
echo "JVM Path: $JVM_PATH"

# Run Training
./venv/bin/python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --value-model models_10k/value_model.pkl \
    --weight-model models_10k/weight_model.pkl \
    --index-path data/msmarco_index \
    --output-dir models_10k/ \
    --max-queries 2000 \
    --lucene-path ./lucene_jars \
    2>&1 | tee train_10k_budget_screen.log

#!/bin/bash
# scripts/train_hybrid_experiment.sh

# Exit on error
set -e

# Source venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

DATA_DIR="data/training_data_hybrid"
MODELS_DIR="models_hybrid"
mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR"

# Clean up previous run data to ensure valid regeneration
rm -rf "$DATA_DIR"/*

echo "=== 1. Generate Hybrid Training Data (MAP + nDCG) ==="
# Limit to 1000 queries for faster experimentation
python3 scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir "$DATA_DIR" \
    --max-queries 40 \
    --max-candidates-per-query 20 \
    --checkpoint-every 10 \
    --kb-candidates data/kb_candidates_train.jsonl \
    --llm-candidates data/llm_candidates_train.jsonl

echo "=== 2. Train Hybrid Value/Weight Models ==="
python3 scripts/train_value_weight_models.py \
    --value-data "$DATA_DIR/value_training_data.pkl" \
    --weight-data "$DATA_DIR/weight_training_data.pkl" \
    --output-dir "$MODELS_DIR"

echo "=== 3. Quick Evaluation (Hybrid Models) ==="
# Evaluate on the small tuning subset (100 queries) first
python3 scripts/analyze_llm_results.py \
    --best-lambda 0.1 \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --lucene-path lucene_jars/ \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --budget-model "models_llm/budget_model.pkl" \
    --llm-candidates data/llm_candidates_test.jsonl \
    --max-queries 40

echo "=== Hybrid Experiment Complete! ==="

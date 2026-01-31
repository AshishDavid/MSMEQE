#!/bin/bash
set -e  # Exit on error

# Configuration
DATA_DIR="data/training_data_full"
MODELS_DIR="models_final"
LUCENE_LIB="lucene_jars"

export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR"

echo "=== 1. Cleaning Previous Data ==="
rm -rf "$DATA_DIR"/*

echo "=== 2. Generating Training Data (MAP-Only, 5000 Queries) ==="
# Note: This uses the stopword-fixed pipeline
python3 scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir "$DATA_DIR" \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --lucene-path "$LUCENE_LIB" \
    --llm-candidates data/llm_candidates_train.jsonl \
    --max-queries 5000

echo "=== 3. Training Value & Weight Models ==="
python3 scripts/train_value_weight_models.py \
    --value-data "$DATA_DIR/value_training_data.pkl" \
    --weight-data "$DATA_DIR/weight_training_data.pkl" \
    --output-dir "$MODELS_DIR"

echo "=== 4. Training Budget Model ==="
python3 scripts/train_budget_model.py \
    --queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index data/msmarco_index \
    --output "$MODELS_DIR/budget_model.pkl" \
    --lucene-path "$LUCENE_LIB" \
    --llm-candidates data/llm_candidates_train.jsonl \
    --max-queries 2000

echo "=== 5. Final Evaluation ==="
# Evaluate on Test Set
python3 scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --budget-model "$MODELS_DIR/budget_model.pkl" \
    --llm-candidates data/llm_candidates_test.jsonl \
    --lucene-path "$LUCENE_LIB" \
    --output "$MODELS_DIR/final_eval_metrics.json"

echo "=== Full Training Complete ==="

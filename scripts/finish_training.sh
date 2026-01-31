#!/bin/bash
set -e

# Configuration
DATA_DIR="data/training_data_full"
MODELS_DIR="models_final"
LUCENE_LIB="lucene_jars"

export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

source venv/bin/activate

echo "=== Resuming Training Pipeline ==="

echo "=== 4. Training Budget Model ==="
python3 scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --output-dir "$MODELS_DIR" \
    --lucene-path "$LUCENE_LIB" \
    --llm-candidates data/llm_candidates_train.jsonl \
    --max-queries 2000

echo "=== 5. Final Evaluation ==="
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

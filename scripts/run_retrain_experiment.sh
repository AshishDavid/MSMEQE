#!/bin/bash
# scripts/run_retrain_experiment.sh
# Experiment 2: Retraining with Stricter Objective (Lambda=0.1 in Data Gen)

# Ensure Java environment is set
export JAVA_HOME=$(pwd)/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export PYTHONPATH=.

LOG_FILE="experiment_retrain.log"
DATA_DIR="data/training_data_strict"
MODEL_DIR="models_strict"

mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

echo "=== STEP 1: Generate Training Data (Strict Lambda=0.1) ===" | tee -a $LOG_FILE
# Note: create_training_data_msmeqe.py has been modified to use 0.1/0.9 interpolation
./venv/bin/python scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir $DATA_DIR \
    --max-queries 5000 \
    --max-candidates-per-query 20 \
    --kb-candidates data/kb_candidates_train.jsonl \
    --llm-candidates data/llm_candidates_train.jsonl \
    --lucene-path ./lucene_jars \
    2>&1 | tee -a $LOG_FILE

echo "=== STEP 2: Train Models ===" | tee -a $LOG_FILE
./venv/bin/python scripts/train_value_weight_models.py \
    --value-data $DATA_DIR/value_training_data.pkl \
    --weight-data $DATA_DIR/weight_training_data.pkl \
    --output-dir $MODEL_DIR \
    2>&1 | tee -a $LOG_FILE

./venv/bin/python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --value-model $MODEL_DIR/value_model.pkl \
    --weight-model $MODEL_DIR/weight_model.pkl \
    --output-dir $MODEL_DIR \
    --max-queries 1000 \
    --index-path data/msmarco_index \
    --lucene-path ./lucene_jars \
    2>&1 | tee -a $LOG_FILE

echo "=== STEP 3: Evaluate New Models ===" | tee -a $LOG_FILE
./venv/bin/python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --value-model $MODEL_DIR/value_model.pkl \
    --weight-model $MODEL_DIR/weight_model.pkl \
    --budget-model $MODEL_DIR/budget_model.pkl \
    --sbert-model sentence-transformers/all-MiniLM-L6-v2 \
    --kb-candidates data/kb_candidates_test.jsonl \
    --llm-candidates data/llm_candidates_test.jsonl \
    --output runs/msmeqe_retrained.txt \
    --run-name MS-MEQE-Retrained \
    --log-per-query results/retrained_stats.jsonl \
    --lambda 0.1 \
    --lucene-path ./lucene_jars \
    2>&1 | tee -a $LOG_FILE

echo "Experiment 2 Complete!" | tee -a $LOG_FILE

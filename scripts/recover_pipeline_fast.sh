#!/bin/bash

# scripts/recover_pipeline.sh

# Exit on error
set -e

# Logging
LOG_FILE="recovery_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Recovering MS-MEQE Pipeline"
echo "Skipping Data Generation (Step 2)"
echo "Resuming from Step 3 (Training)"
echo "Date: $(date)"
echo "========================================================"

# --- 1. Environment Setup ---
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH

# --- 3. Train Value and Weight Models ---
# echo ""
# echo "========================================================"
# echo "[Step 3] Training Value and Weight Models..."
# echo "========================================================"
#
# ./venv/bin/python scripts/train_value_weight_models.py \
#     --value-data data/training_data/value_training_data.pkl \
#     --weight-data data/training_data/weight_training_data.pkl \
#     --output-dir models/ \
#     --tune
#
# echo "[Step 3] V/W Model Training Complete."

# --- 4. Train Budget Model ---
echo ""
echo "========================================================"
echo "[Step 4] Training Budget Model..."
echo "========================================================"

./venv/bin/python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --output-dir models/

echo "[Step 4] Budget Model Training Complete."

# --- 5. Final Evaluation ---
echo ""
echo "========================================================"
echo "[Step 5] Running Final Evaluation on Test Set..."
echo "========================================================"

RUN_FILE="runs/dual_metric_optimized_run.txt"

./venv/bin/python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --lucene-path lucene_jars/ \
    --value-model "models/value_model.pkl" \
    --weight-model "models/weight_model.pkl" \
    --budget-model "models/budget_model.pkl" \
    --llm-candidates "data/llm_candidates_test.jsonl" \
    --output "$RUN_FILE" \
    --log-per-query runs/dual_metric_per_query.jsonl \
    --run-name "Dual-Metric-Opt"

echo "========================================================"
echo "RECOVERY PIPELINE COMPLETE SUCCESS!"
echo "Date: $(date)"
echo "Run File: $RUN_FILE"
echo "========================================================"

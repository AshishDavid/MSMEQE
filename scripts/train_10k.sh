#!/bin/bash
# Train MS-MEQE with 10,000 queries
# This script generates training data and trains value/weight models

set -e

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

export PYTHONPATH="$PWD:$PYTHONPATH"
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PATH="$JAVA_HOME/bin:$PATH"

echo "=========================================="
echo "MS-MEQE Training with 10,000 Queries"
echo "Started: $(date)"
echo "=========================================="

# Step 1: Generate Training Data
echo ""
echo "[Step 1/3] Generating training data (10k queries)..."
echo "This will take several hours..."

./venv/bin/python scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir data/training_data_10k \
    --max-queries 10000 \
    --kb-candidates data/kb_candidates_train.jsonl \
    --lucene-path ./lucene_jars

echo "[Step 1/3] Training data generation complete."

# Step 2: Train Value and Weight Models
echo ""
echo "[Step 2/3] Training Value and Weight models..."

./venv/bin/python scripts/train_value_weight_models.py \
    --value-data data/training_data_10k/value_training_data.pkl \
    --weight-data data/training_data_10k/weight_training_data.pkl \
    --output-dir models_10k/ \
    --tune-hyperparams

echo "[Step 2/3] Model training complete."

# Step 3: Train Budget Model
echo ""
echo "[Step 3/3] Training Budget model..."

./venv/bin/python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir models_10k/ \
    --max-queries 2000 \
    --lucene-path ./lucene_jars

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Finished: $(date)"
echo "Models saved to: models_10k/"
echo "=========================================="

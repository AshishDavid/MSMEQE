
#!/bin/bash
# scripts/train_llm_models.sh

# Exit on error
set -e
export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

LLM_TRAIN="data/llm_candidates_train.jsonl"
TRAIN_DATA_DIR="data/training_data_llm"
MODELS_DIR="models_llm"

echo "=== LLM-MEQE: Starting Training Phase ==="

if [ ! -f "$LLM_TRAIN" ]; then
    echo "Error: $LLM_TRAIN not found."
    exit 1
fi

echo "Step 1: Regenerating Training Data (with LLM Candidates)..."
# Using 5000 queries matched to the precomputed candidates
python scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir "$TRAIN_DATA_DIR" \
    --max-queries 5000 \
    --llm-candidates "$LLM_TRAIN"

echo "Step 2: Retraining Value/Weight Models..."
python scripts/train_value_weight_models.py \
    --value-data "$TRAIN_DATA_DIR/value_training_data.pkl" \
    --weight-data "$TRAIN_DATA_DIR/weight_training_data.pkl" \
    --output-dir "$MODELS_DIR" \
    --n-estimators 100 \
    --max-depth 6

echo "Step 3: Retraining Budget Model..."
python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --output-dir "$MODELS_DIR" \
    --llm-candidates "$LLM_TRAIN"

echo "=== Training Phase Complete! ==="
echo "You can now run the evaluation step once test candidates are ready."


#!/bin/bash
# scripts/train_budget_recovery.sh

# Exit on error
set -e
export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

LLM_TRAIN="data/llm_candidates_train.jsonl"
MODELS_DIR="models_llm"

echo "=== LLM-MEQE: Recovering Budget Training ==="

if [ ! -f "$LLM_TRAIN" ]; then
    echo "Error: $LLM_TRAIN not found."
    exit 1
fi

echo "Step 3 (Retry): Retraining Budget Model..."
python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --output-dir "$MODELS_DIR" \
    --llm-candidates "$LLM_TRAIN"

echo "=== Budget Training Complete! ==="


#!/bin/bash
# scripts/finish_llm_integration.sh

# Exit on error
set -e

LLM_TRAIN="data/llm_candidates_train.jsonl"
LLM_TEST="data/llm_candidates_test.jsonl"
TRAIN_DATA_DIR="data/training_data_llm"
MODELS_DIR="models_llm"
RUN_FILE="runs/llm_meqe_run1.txt"

echo "=== LLM-MEQE: Finishing Integration ==="

# 1. Check if LLM generation is done
if [ ! -f "$LLM_TRAIN" ]; then
    echo "Error: $LLM_TRAIN not found. Please wait for the 'llm_gen' screen session to finish."
    echo "Check progress: tail -f scripts/llm_gen_log.txt (if logged) or attach to screen."
    exit 1
fi

echo "Step 1: Regenerating Training Data (with LLM Candidates)..."
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

echo "Step 4: Running LLM-MEQE Evaluation..."
python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --lucene-path lucene_jars/ \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --budget-model "$MODELS_DIR/budget_model.pkl" \
    --llm-candidates "$LLM_TEST" \
    --output "$RUN_FILE" \
    --run-name "LLM-MEQE-Run1"

echo "=== LLM-MEQE Complete! ==="

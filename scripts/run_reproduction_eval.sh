
#!/bin/bash
# scripts/run_reproduction_eval.sh

# Exit on error
set -e
export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=$(pwd)/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

# Use models_final as they appear to be the latest full training
MODELS_DIR="models_final"
LLM_TEST="data/llm_candidates_test.jsonl"
RUN_FILE="runs/reproduction_run.txt"

echo "=== MS-MEQE: Running Reproduction Evaluation ==="
echo "Models Directory: $MODELS_DIR"
echo "Output Run File: $RUN_FILE"

if [ ! -f "$LLM_TEST" ]; then
    echo "Error: $LLM_TEST not found."
    exit 1
fi

# Ensure output directory exists
mkdir -p runs

venv/bin/python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --lucene-path lucene_jars/ \
    --value-model "$MODELS_DIR/value_model.pkl" \
    --weight-model "$MODELS_DIR/weight_model.pkl" \
    --budget-model "$MODELS_DIR/budget_model.pkl" \
    --llm-candidates "$LLM_TEST" \
    --output "$RUN_FILE" \
    --log-per-query runs/reproduction_per_query.jsonl \
    --run-name "Reproduction-Run"

echo "=== Evaluation Complete! ==="

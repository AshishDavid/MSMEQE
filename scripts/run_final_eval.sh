
#!/bin/bash
# scripts/run_final_eval.sh

# Exit on error
set -e
export PYTHONPATH=$PYTHONPATH:.
export JAVA_HOME=/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

LLM_TEST="data/llm_candidates_test.jsonl"
MODELS_DIR="models_llm"
RUN_FILE="runs/llm_meqe_run1.txt"

echo "=== LLM-MEQE: Running Final Evaluation ==="

if [ ! -f "$LLM_TEST" ]; then
    echo "Error: $LLM_TEST not found."
    exit 1
fi

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

echo "=== Evaluation Complete! ==="

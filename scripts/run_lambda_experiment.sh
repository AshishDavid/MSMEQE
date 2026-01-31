#!/bin/bash
# scripts/run_lambda_experiment.sh
# Experiment 1: Evaluate existing models with Low Lambda (0.05)

# Ensure Java environment is set
export JAVA_HOME=$(pwd)/jdk-21
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export PYTHONPATH=.

LOG_FILE="experiment_lambda_0.05.log"

echo "Starting Low Lambda Evaluation (Lambda=0.05)..." | tee -a $LOG_FILE

./venv/bin/python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --budget-model models/budget_model.pkl \
    --sbert-model sentence-transformers/all-MiniLM-L6-v2 \
    --kb-candidates data/kb_candidates_test.jsonl \
    --llm-candidates data/llm_candidates_test.jsonl \
    --output runs/msmeqe_lambda_0.05.txt \
    --run-name MS-MEQE-L0.05 \
    --log-per-query results/lambda_0.05_stats.jsonl \
    --lambda 0.05 \
    --lucene-path ./lucene_jars \
    2>&1 | tee -a $LOG_FILE

echo "Experiment 1 Complete!" | tee -a $LOG_FILE

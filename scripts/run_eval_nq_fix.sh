#!/bin/bash
# scripts/run_eval_nq_fix.sh
# Evaluate MS-MEQE 10k Models on NQ (Fix)

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

# Setup Environment
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Models
VALUE_MODEL="models_10k/value_model.pkl"
WEIGHT_MODEL="models_10k/weight_model.pkl"
BUDGET_MODEL="models_10k/budget_model.pkl"

echo "========================================================"
echo "Evaluating MS-MEQE (10k Training) on Natural Questions"
echo "========================================================"
./venv/bin/python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/nq/queries.tsv \
    --qrels data/nq/qrels_msmarco.txt \
    --value-model $VALUE_MODEL \
    --weight-model $WEIGHT_MODEL \
    --budget-model $BUDGET_MODEL \
    --kb-candidates data/kb_candidates_test.jsonl \
    --output runs/msmeqe_10k.nq.txt \
    --run-name MS-MEQE-10k \
    --lucene-path ./lucene_jars \
    2>&1 | tee results/eval_10k_nq_fix.log

echo "NQ evaluation complete."

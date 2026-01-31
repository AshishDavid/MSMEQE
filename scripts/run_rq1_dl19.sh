#!/bin/bash
# scripts/run_rq1_dl19.sh
# Run RQ1 (Greedy vs Knapsack) on TREC DL 2019

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

# Setup
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

VALUE_MODEL="models_10k/value_model.pkl"
WEIGHT_MODEL="models_10k/weight_model.pkl"
BUDGET_MODEL="models_10k/budget_model.pkl"

echo "========================================================"
echo "RQ1: Running GREEDY Baseline on DL19 (Budget=70)"
echo "========================================================"
./venv/bin/python scripts/run_rq1_ablation.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2019/queries.tsv \
    --qrels data/trec_dl_2019/qrels.txt \
    --selection-method greedy \
    --value-model $VALUE_MODEL \
    --weight-model $WEIGHT_MODEL \
    --budget-model $BUDGET_MODEL \
    --kb-candidates data/kb_candidates_test.jsonl \
    --output runs/rq1_greedy.dl19.txt \
    --lucene-path ./lucene_jars \
    2>&1 | tee results/rq1_greedy_dl19.log

echo "========================================================"
echo "RQ1: Running KNAPSACK (MS-MEQE) on DL19 (Budget=70)"
echo "========================================================"
./venv/bin/python scripts/run_rq1_ablation.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2019/queries.tsv \
    --qrels data/trec_dl_2019/qrels.txt \
    --selection-method knapsack \
    --value-model $VALUE_MODEL \
    --weight-model $WEIGHT_MODEL \
    --budget-model $BUDGET_MODEL \
    --kb-candidates data/kb_candidates_test.jsonl \
    --output runs/rq1_knapsack.dl19.txt \
    --lucene-path ./lucene_jars \
    2>&1 | tee results/rq1_knapsack_dl19.log

echo "RQ1 DL19 Comparison Complete."

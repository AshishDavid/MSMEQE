#!/bin/bash
# scripts/run_rq1_sweep.sh
# Run RQ1 Experiment 1.1: Tightness Sweep on DL19

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

# Setup
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

VALUE_MODEL="models_10k/value_model.pkl"
WEIGHT_MODEL="models_10k/weight_model.pkl"

echo "========================================================"
echo "RQ1 EXP 1.1: TIGHTNESS SWEEP (DL19)"
echo "Comparing Greedy vs Knapsack at W={1,2,3,5,8,13,21}"
echo "========================================================"

mkdir -p runs/rq1_sweep

./venv/bin/python scripts/run_rq1_tightness_sweep.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2019/queries.tsv \
    --qrels data/trec_dl_2019/qrels.txt \
    --value-model $VALUE_MODEL \
    --weight-model $WEIGHT_MODEL \
    --output-prefix runs/rq1_sweep/dl19 \
    --lucene-path ./lucene_jars \
    2>&1 | tee results/rq1_tightness_sweep.log

echo "Sweep Complete."

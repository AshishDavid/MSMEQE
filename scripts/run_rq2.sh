#!/bin/bash
# scripts/run_rq2.sh
# Run RQ2 Budget Sweep for SIGIR Paper

cd /home/ad6b8/Documents/Research/QueryExpansion/msmeqe

# Setup
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Enable debugging temporarily if needed
# export LOG_LEVEL=DEBUG

VALUE_MODEL="models_10k/value_model.pkl"
WEIGHT_MODEL="models_10k/weight_model.pkl"

MODELS=("rm3" "msmeqe")
# Reduced set for speed if needed, but paper needs range:
BUDGETS=(10 30 50 70) 
DATASETS=("dl19" "dl20")

runs_dir="runs/rq2"
res_dir="results/rq2"
mkdir -p $runs_dir $res_dir

echo "Starting RQ2 Sweep..."

for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "dl19" ]; then
        TOPICS="data/trec_dl_2019/queries.tsv"
        QRELS="data/trec_dl_2019/qrels.txt"
    else
        TOPICS="data/trec_dl_2020/queries.tsv"
        QRELS="data/trec_dl_2020/qrels.txt"
    fi

    for model in "${MODELS[@]}"; do
        for k in "${BUDGETS[@]}"; do
            echo "Running $model @ k=$k on $dataset..."
            
            ./venv/bin/python scripts/run_rq2_budget_sweep.py \
                --index data/msmarco_index \
                --topics $TOPICS \
                --qrels $QRELS \
                --model-type $model \
                --budget $k \
                --value-model $VALUE_MODEL \
                --weight-model $WEIGHT_MODEL \
                --output "$runs_dir/${model}_${k}_${dataset}.txt" \
                --lucene-path ./lucene_jars \
                > "$res_dir/${model}_${k}_${dataset}.log" 2>&1
                
            # Extract MAP for quick view
            map=$(grep "map:" "$res_dir/${model}_${k}_${dataset}.log" | tail -n 1)
            echo "  -> Completed. $map"
        done
    done
done

echo "RQ2 Sweep Complete."

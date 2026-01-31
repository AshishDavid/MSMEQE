#!/bin/bash
set -e

# Configuration
export JAVA_HOME="/home/ad6b8/Documents/Research/QueryExpansion/msmeqe/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
PYTHON="./venv/bin/python3"
QUERY_FILE="data/test_queries.tsv"
WAT_OUTPUT="data/wat_output_test.jsonl"
LLM_CANDS="data/llm_candidates_test.jsonl"
DENSE_RUN="runs/dense_baseline.txt"
MSMEQE_RUN="runs/msmeqe_run.txt"
QRELS="data/test_qrels.txt"
OUTPUT_DIR="results/advanced_opt"
STATS_FILE="$OUTPUT_DIR/oracle_stats_advanced.jsonl"
DATA_FILE="$OUTPUT_DIR/oracle_data_advanced.pkl"
MODEL_DIR="models/oracle_advanced"

mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_DIR

# 1. Parallel Stats Generation
echo "Starting Parallel Stats Generation..."
split -n l/4 $QUERY_FILE "${QUERY_FILE}_part_"

for part in ${QUERY_FILE}_part_*; do
    echo "Processing $part..."
    $PYTHON scripts/regenerate_oracle_stats.py \
        --queries $part \
        --wat-output $WAT_OUTPUT \
        --llm-candidates $LLM_CANDS \
        --output "${part}_stats.jsonl" &
done

wait # Wait for all parallel jobs to finish

# 2. Aggregate Stats
echo "Aggregating stats..."
cat ${QUERY_FILE}_part_*_stats.jsonl > $STATS_FILE
rm ${QUERY_FILE}_part_* ${QUERY_FILE}_part_*_stats.jsonl

# 3. Create Dataset
echo "Creating advanced oracle dataset..."
$PYTHON scripts/create_enhanced_oracle_data.py \
    --dense-run $DENSE_RUN \
    --msmeqe-run $MSMEQE_RUN \
    --qrels $QRELS \
    --oracle-stats $STATS_FILE \
    --output $DATA_FILE

# 4. Retrain Oracle
echo "Retraining Oracle Switch..."
$PYTHON scripts/train_oracle_switch.py \
    --data $DATA_FILE \
    --output-dir $MODEL_DIR

# 5. Evaluate
echo "Evaluating Final Performance..."
$PYTHON scripts/run_oracle_enhanced.py \
    --dense-run $DENSE_RUN \
    --msmeqe-run runs/msmeqe_multi_best.txt \
    --oracle-stats $STATS_FILE \
    --model-path "$MODEL_DIR/budget_model.pkl" \
    --output runs/msmeqe_oracle_advanced.txt

$PYTHON calculate_map.py $QRELS runs/msmeqe_oracle_advanced.txt > results/final_metrics_advanced.txt

echo "Pipeline Complete. Results in results/final_metrics_advanced.txt"

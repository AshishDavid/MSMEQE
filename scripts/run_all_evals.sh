#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
set -e

# Ensure data dictionary exists
mkdir -p runs

echo "========================================================"
echo "Starting MS-MEQE Evaluation Pipeline"
echo "========================================================"

# 1. Download Data
echo "[1/4] Downloading evaluation datasets..."
python scripts/download_eval_datasets.py

# 2. TREC DL 2019
if [ -f "data/trec_dl_2019/queries.tsv" ]; then
    echo "[2/4] Running Evaluation on TREC DL 2019..."
    python scripts/run_msmeqe_evaluation.py \
        --index data/msmarco_index \
        --topics data/trec_dl_2019/queries.tsv \
        --qrels data/trec_dl_2019/qrels.txt \
        --value-model models/value_model.pkl \
        --weight-model models/weight_model.pkl \
        --budget-model models/budget_model.pkl \
        --emb-vocab data/vocab_embeddings.pkl \
        --output runs/msmeqe.dl19.txt \
        --run-name MS-MEQE-DL19 \
        --lucene-path lucene_jars \
        --collection-size 8841823
else
    echo "Skipping DL 19 (Data not found)"
fi

# 3. TREC DL 2020
if [ -f "data/trec_dl_2020/queries.tsv" ]; then
    echo "[3/4] Running Evaluation on TREC DL 2020..."
    python scripts/run_msmeqe_evaluation.py \
        --index data/msmarco_index \
        --topics data/trec_dl_2020/queries.tsv \
        --qrels data/trec_dl_2020/qrels.txt \
        --value-model models/value_model.pkl \
        --weight-model models/weight_model.pkl \
        --budget-model models/budget_model.pkl \
        --emb-vocab data/vocab_embeddings.pkl \
        --output runs/msmeqe.dl20.txt \
        --run-name MS-MEQE-DL20 \
        --lucene-path lucene_jars \
        --collection-size 8841823
else
    echo "Skipping DL 20 (Data not found)"
fi

# 4. Natural Questions (NQ)
if [ -d "data/nq_index" ] && [ -f "data/nq/queries.tsv" ]; then
    echo "[4/5] Running Evaluation on Natural Questions..."
    python scripts/run_msmeqe_evaluation.py \
        --index data/nq_index \
        --topics data/nq/queries.tsv \
        --qrels data/nq/qrels.txt \
        --value-model models/value_model.pkl \
        --weight-model models/weight_model.pkl \
        --budget-model models/budget_model.pkl \
        --emb-vocab data/vocab_embeddings.pkl \
        --output runs/msmeqe.nq.txt \
        --run-name MS-MEQE-NQ \
        --lucene-path lucene_jars \
        --collection-size 2681468
else
    echo "Skipping NQ (Index or Data not found)"
fi

# 5. TREC Robust04
if [ -d "data/robust04_index" ] && [ -f "data/robust04/queries.tsv" ]; then
    echo "[5/5] Running Evaluation on TREC Robust04..."
    python scripts/run_msmeqe_evaluation.py \
        --index data/robust04_index \
        --topics data/robust04/queries.tsv \
        --qrels data/robust04/qrels.txt \
        --value-model models/value_model.pkl \
        --weight-model models/weight_model.pkl \
        --budget-model models/budget_model.pkl \
        --emb-vocab data/vocab_embeddings.pkl \
        --output runs/msmeqe.robust04.txt \
        --run-name MS-MEQE-Robust04 \
        --lucene-path lucene_jars \
        --collection-size 528155
else
    echo "Skipping Robust04 (Index or Data not found)"
fi

echo "========================================================"
echo "Evaluation Pipeline Complete"
echo "Results saved in runs/"
echo "========================================================"

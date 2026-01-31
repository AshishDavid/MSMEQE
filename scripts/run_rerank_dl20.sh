#!/bin/bash
# scripts/run_rerank_dl20.sh

source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH

python scripts/rerank_dl20.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2020/queries.tsv \
    --qrels data/trec_dl_2020/qrels.txt \
    --bm25-run bm25.run.txt \
    --output runs/msmeqe.dl20.rerank.txt \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --budget-model models/budget_model.pkl \
    --emb-vocab data/vocab_embeddings.pkl \
    --kb-data data/kb_candidates_all_v2.jsonl \
    --lambda-val 0.4 \
    --collection-size 8841823 2>&1 | tee dl20_rerank.log

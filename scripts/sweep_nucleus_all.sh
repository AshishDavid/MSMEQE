#!/bin/bash
# scripts/sweep_nucleus_all.sh

# Environment
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
source venv/bin/activate

echo "========================================================"
echo "RUNNING NUCLEUS SAMPLING (ALL SOURCES) ON ALL DATASETS"
echo "========================================================"

# 1. Robust04
echo ">>> Processing Robust04..."
python -u scripts/run_nucleus_generic.py \
    --index data/robust04_index \
    --topics trec_robust04_queries.tsv \
    --qrels trec_robust04_qrels.tsv \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --lucene-path lucene_jars \
    --p 0.9 --lambda-val 0.4 \
    --collection-size 528024 \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --kb-data data/kb_candidates_all_v2.jsonl

# 2. TREC DL 2019
echo ">>> Processing TREC DL 2019..."
python -u scripts/run_nucleus_generic.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2019/queries.tsv \
    --qrels data/trec_dl_2019/qrels.txt \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --lucene-path lucene_jars \
    --p 0.9 --lambda-val 0.4 \
    --collection-size 8841823 \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --kb-data data/kb_candidates_all_v2.jsonl

# 3. TREC DL 2020
echo ">>> Processing TREC DL 2020..."
python -u scripts/run_nucleus_generic.py \
    --index data/msmarco_index \
    --topics data/trec_dl_2020/queries.tsv \
    --qrels data/trec_dl_2020/qrels.txt \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --lucene-path lucene_jars \
    --p 0.9 --lambda-val 0.4 \
    --collection-size 8841823 \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --kb-data data/kb_candidates_all_v2.jsonl

# 4. Natural Questions (NQ)
echo ">>> Processing NQ..."
python -u scripts/run_nucleus_generic.py \
    --index data/nq_index \
    --topics data/nq/queries.tsv \
    --qrels data/nq/qrels.txt \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --lucene-path lucene_jars \
    --p 0.9 --lambda-val 0.4 \
    --collection-size 2681468 \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --kb-data data/kb_candidates_all_v2.jsonl

echo "========================================================"
echo "SWEEP COMPLETE"
echo "========================================================"

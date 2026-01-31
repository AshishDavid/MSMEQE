#!/bin/bash
source venv/bin/activate
export PYTHONPATH=.
export JAVA_HOME=$(pwd)/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
set -e

# Run NQ Evaluation
if [ -d "data/nq_index" ] && [ -f "data/nq/queries.tsv" ]; then
    echo "Running Evaluation on Natural Questions..."
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
    echo "NQ Data/Index missing!"
fi

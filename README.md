# MS-MEQE: Multi-Source Magnitude-Encoded Query Expansion

Official implementation of **"Query Expansion as Multi-Source Constrained Optimization"** (SIGIR 2025).

MS-MEQE reformulates query expansion as an **unbounded knapsack problem**, learning Value (benefit) and Weight (risk) functions for expansion candidates across multiple sources (RM3, Knowledge Base, Embeddings).

---

## Prerequisites

### Software Requirements

- Python 3.8+
- Java JDK 21 (for Lucene)

### Install Python Dependencies

```bash
pip install torch numpy scipy scikit-learn xgboost sentence-transformers nltk ir-datasets tqdm jnius joblib
```

### Download Lucene JARs

Download Lucene 10.1.0 from [Apache Archive](https://archive.apache.org/dist/lucene/java/10.1.0/) and extract to `lucene_jars/`:

```bash
mkdir lucene_jars
# Download and place these JARs in lucene_jars/:
#   - lucene-core-10.1.0.jar
#   - lucene-analysis-common-10.1.0.jar
#   - lucene-queryparser-10.1.0.jar
```

### Download JDK 21

```bash
wget https://download.java.net/java/GA/jdk21/fd2272bbf8e04c3dbaee13770090416c/35/GPL/openjdk-21_linux-x64_bin.tar.gz
tar -xzf openjdk-21_linux-x64_bin.tar.gz
mv jdk-21 ./jdk-21
```

---

## Complete Pipeline (Step-by-Step)

### Environment Setup

Run these before every session:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
```

---

### Step 1: Download NLTK Data

```bash
python -c "import nltk; nltk.download('wordnet')"
```

---

### Step 2: Download MS MARCO Dataset

This downloads train/test queries and relevance judgments (qrels):

```bash
python scripts/download_data.py
```

**Output files:**

- `data/train_queries.tsv`, `data/train_queries.json`
- `data/train_qrels.txt`
- `data/test_queries.tsv`, `data/test_queries.json`
- `data/test_qrels.txt`

---

### Step 3: Build Lucene Index

Index the MS MARCO passage collection (~8.8M documents):

```bash
python scripts/index_collection.py \
    --collection msmarco-passage \
    --output data/msmarco_index \
    --lucene-path ./lucene_jars \
    --analyzer english \
    --ram-buffer 4096
```

⏱️ **Time: ~2-3 hours** | 💾 **Space: ~15 GB**

---

### Step 4: Precompute Document Embeddings

Encode all documents with Sentence-BERT for dense retrieval:

```bash
python scripts/precompute_doc_embeddings.py \
    --collection msmarco-passage \
    --output-dir data/msmarco_index \
    --batch-size 512 \
    --model-name sentence-transformers/all-MiniLM-L6-v2
```

⏱️ **Time: ~4-6 hours** | 💾 **Space: ~25 GB**

---

### Step 5: Build SBERT Vocabulary

Create term embeddings for semantic neighbor expansion:

```bash
python scripts/build_sbert_vocab.py \
    --from-index data/msmarco_index \
    --output data/vocab_embeddings.pkl \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --max-terms 100000
```

---

### Step 6: Precompute KB Candidates (Optional)

Run entity linking using WAT API:

```bash
# First, convert queries to WAT input format
python scripts/convert_queries_to_wat_input.py

# Then run WAT entity linker (requires API token)
python scripts/wat_entity_linker.py

# Convert WAT output to KB candidates
python scripts/precompute_kb_candidates.py \
    --topics data/train_queries.tsv \
    --wat-output data/wat_raw_output.jsonl \
    --output data/kb_candidates_train.jsonl
```

> **Note:** You need a WAT/TagMe API token. Set it in `scripts/wat_entity_linker.py`.

---

### Step 7: Generate Training Data (5K Queries)

Generate value/weight training data:

```bash
python scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir data/training_data_5k \
    --max-queries 5000 \
    --lucene-path ./lucene_jars
```

⏱️ **Time: ~6-12 hours**

---

### Step 8: Train Value and Weight Models

```bash
python scripts/train_value_weight_models.py \
    --value-data data/training_data_5k/value_training_data.pkl \
    --weight-data data/training_data_5k/weight_training_data.pkl \
    --output-dir models/ \
    --tune-hyperparams
```

---

### Step 9: Train Budget Model

```bash
python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --output-dir models/ \
    --max-queries 2000 \
    --lucene-path ./lucene_jars
```

**Output models:**

- `models/value_model.pkl`
- `models/weight_model.pkl`
- `models/budget_model.pkl`

---

### Step 10: Run Evaluation

Evaluate on test set with MAP, nDCG, MRR:

```bash
python scripts/run_msmeqe_evaluation.py \
    --index data/msmarco_index \
    --topics data/test_queries.tsv \
    --qrels data/test_qrels.txt \
    --lucene-path ./lucene_jars \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --budget-model models/budget_model.pkl \
    --emb-vocab data/vocab_embeddings.pkl \
    --output runs/msmeqe_results.txt \
    --run-name MS-MEQE \
    --log-per-query results/per_query_stats.jsonl
```

**Metrics reported:**

- `MAP` - Mean Average Precision
- `nDCG@10`, `nDCG@20` - Normalized Discounted Cumulative Gain
- `MRR` - Mean Reciprocal Rank
- `Recall@100`, `Recall@1000`
- `P@10`, `P@20` - Precision

---

## Quick Start (All-in-One Script)

For convenience, after setup you can run the full training pipeline:

```bash
# Run everything (Steps 7-9)
./scripts/train_10k.sh
```

---

## Repository Structure

```
msmeqe/
├── src/                    # Core source code
│   ├── expansion/          # Candidate extraction & knapsack logic
│   ├── features/           # Feature extraction
│   ├── models/             # Budget predictor
│   ├── reranking/          # Semantic encoders
│   ├── retrieval/          # BM25 & evaluation
│   └── utils/              # Lucene utils, knapsack solver
├── scripts/                # Executable scripts
│   ├── download_data.py
│   ├── index_collection.py
│   ├── precompute_doc_embeddings.py
│   ├── create_training_data_msmeqe.py
│   ├── train_value_weight_models.py
│   ├── train_budget_model.py
│   └── run_msmeqe_evaluation.py
└── tests/                  # Unit tests
```


# MS-MEQE: Multi-Source Magnitude-Encoded Query Expansion

This repository contains the official implementation of the paper **"Query Expansion as Multi-Source Constrained Optimization"**.

MS-MEQE is a query expansion framework that reformulates term selection as an **unbounded knapsack problem**. It jointly optimizes expansion across three heterogeneous sources:

1.  **Pseudo-Relevant Documents** (via RM3) 
2.  **Knowledge Bases** (via Entity Linking) 

The system learns **Value** (benefit) and **Weight** (risk) functions for expansion candidates and dynamically predicts a query-specific expansion **Budget**.

-----

## 🛠 Prerequisites

### Software

  * **Python:** 3.8+
  * **Java JDK:** 11+ (Required for PyJnius/Lucene interaction)
  * **Apache Lucene:** Version 10.1.0 (JAR files)

### Python Dependencies

Install the required packages:

```bash
pip install torch numpy scipy scikit-learn xgboost sentence-transformers nltk ir-datasets tqdm jnius
```

### External Data

1.  **MS MARCO Passage Dataset:** Automatically handled via `ir_datasets`, but requires \~50GB disk space.
2.  **Lucene JARs:** You must download the following JARs from the [Apache Lucene Archive](https://archive.apache.org/dist/lucene/java/10.1.0/) and place them in a directory (e.g., `lucene_jars/`):
      * `lucene-core-10.1.0.jar`
      * `lucene-analysis-common-10.1.0.jar`
      * `lucene-queryparser-10.1.0.jar`
      * `lucene-memory-10.1.0.jar`

-----

## 🚀 Execution Pipeline

Follow these steps in exact order to reproduce the system.

### Step 1: NLTK Setup

Before running any scripts, ensure the WordNet corpus is downloaded for feature extraction features (Polysemy estimation).

```bash
python3 -c "import nltk; nltk.download('wordnet')"
```

### Step 2: Index the Collection

Create a Lucene index for the MS MARCO collection. [cite_start]This is required for RM3 expansion and statistical feature extraction (IDF, TF)[cite: 2, 85].

```bash
python scripts/index_collection.py \
    --collection msmarco-passage \
    --output data/msmarco_index \
    --lucene-path ./lucene_jars \
    --analyzer english \
    --ram-buffer 4096
```

  * `--lucene-path`: Directory containing the `.jar` files downloaded in Prerequisites.

### Step 3: Precompute Document Embeddings

[cite_start]Encode the entire document collection using Sentence-BERT for dense retrieval[cite: 83].

```bash
python scripts/precompute_doc_embeddings.py \
    --collection msmarco-passage \
    --output-dir data/msmarco_index \
    --batch-size 512 \
    --model-name sentence-transformers/all-MiniLM-L6-v2
```

  * **Note:** This step requires significant RAM. If running on a low-memory machine, reduce `--batch-size`.

### Step 4: Prepare Expansion Sources

Prepare the vocabulary and candidates for the Embedding and Knowledge Base sources.

#### 4a. Build Embedding Vocabulary ($s_{emb}$)

[cite_start]Creates a vocabulary of terms and their embeddings for semantic neighbor retrieval[cite: 94].

```bash
python scripts/build_sbert_vocab.py \
    --from-index data/msmarco_index \
    --output data/vocab_embeddings.pkl \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --max-terms 100000
```

#### 4b. Precompute KB Candidates ($s_{KB}$)

Runs Entity Linking (WAT) on queries.

  * **Important:** You must obtain a WAT/TagMe API token and update `scripts/wat_entity_linker.py` or use the provided `data/kb_candidates.jsonl` if available.

<!-- end list -->

```bash
python scripts/precompute_kb_candidates.py \
    --topics data/train_queries.tsv \
    --wat-output data/wat_raw_output.jsonl \
    --output data/kb_candidates.jsonl
```

### Step 5: Generate Training Data

[cite_start]This script simulates the expansion process to generate "Ground Truth" data ($\Delta$MAP) for training the Value and Weight models[cite: 130].

```bash
python scripts/create_training_data_msmeqe.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --output-dir data/training_data \
    --vocab-embeddings data/vocab_embeddings.pkl \
    --kb-wat-output data/wat_raw_output.jsonl \
    --max-queries 5000
```

### Step 6: Train MS-MEQE Models

Train the regressors and the budget classifier.

#### 6a. Train Value and Weight Models

[cite_start]Trains XGBoost regressors to predict term benefit and risk[cite: 141, 187].

```bash
python scripts/train_value_weight_models.py \
    --value-data data/training_data/value_training_data.pkl \
    --weight-data data/training_data/weight_training_data.pkl \
    --output-dir models/ \
    --n-estimators 100 \
    --max-depth 6
```

#### 6b. Train Budget Model

[cite_start]Trains a classifier to predict the optimal expansion budget ($W \in \{30, 50, 70\}$) based on query clarity and entropy[cite: 279, 289].

```bash
python scripts/train_budget_model.py \
    --training-queries data/train_queries.json \
    --qrels data/train_qrels.txt \
    --index-path data/msmarco_index \
    --value-model models/value_model.pkl \
    --weight-model models/weight_model.pkl \
    --output-dir models/ \
    --max-queries 2000
```

### Step 7: Run Evaluation

Run the full MS-MEQE pipeline on test queries. [cite_start]This performs candidate extraction, prediction, knapsack optimization, and dense retrieval[cite: 343].

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
    --kb-candidates data/kb_candidates_test.jsonl \
    --output runs/msmeqe_results.txt \
    --run-name MS-MEQE-Run1
```

-----

## 📂 Repository Structure

```text
.
├── data/                       # Data storage (indices, embeddings, training data)
├── models/                     # Saved XGBoost models
├── lucene_jars/                # External Lucene 10.1.0 dependencies
├── scripts/                    # Executable scripts
│   ├── index_collection.py
│   ├── precompute_doc_embeddings.py
│   ├── build_sbert_vocab.py
│   ├── create_training_data_msmeqe.py
│   ├── train_value_weight_models.py
│   ├── train_budget_model.py
│   └── run_msmeqe_evaluation.py
└── src/                        # Core source code
    ├── expansion/              # Candidate extraction & optimization logic
    │   ├── candidate_extraction_pipeline.py
    │   ├── msmeqe_expansion.py # Core Knapsack & Magnitude Encoding logic
    │   ├── rm_expansion.py     # Lucene RM3 implementation
    │   └── ...
    ├── features/               # Feature engineering
    │   └── feature_extraction.py
    ├── reranking/              # Semantic encoders
    ├── retrieval/              # BM25 & Evaluation utils
    └── utils/                  # Knapsack solver, file utils, etc.
```

## 📜 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{msmeqe2025,
  title={Query Expansion as Multi-Source Constrained Optimization},
  author={Unknown},
  booktitle={Proceedings of SIGIR},
  year={2025}
}
```
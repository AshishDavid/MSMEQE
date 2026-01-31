
import argparse
import logging
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.utils.file_utils import load_trec_run, load_qrels
from src.retrieval.evaluator import TRECEvaluator

logger = logging.getLogger(__name__)

def evaluate_per_query(run, qrels, evaluator):
    """Evaluate run per query."""
    per_query_metrics = {}
    for qid in run.keys():
        if qid not in qrels: continue
        mini_run = {qid: run[qid]}
        m = evaluator.evaluate_run(mini_run, qrels)
        per_query_metrics[qid] = m
    return per_query_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--msmeqe-run", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--training-queries", required=True) # Full text for feature extraction
    parser.add_argument("--output", required=True)
    
    # Needs components for feature extraction
    parser.add_argument("--index-path", default="data/msmarco_index")
    parser.add_argument("--collection-size", type=int, default=8841823)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Everything
    dense_run = load_trec_run(args.dense_run)
    msmeqe_run = load_trec_run(args.msmeqe_run)
    qrels = load_qrels(args.qrels)
    
    logger.info("Loading queries text...")
    queries = {} 
    # This might be big json or line format
    # The original script uses load_queries_from_file that handles JSON/TSV.
    # We'll just define a simple loader or import.
    # Let's import from existing script to be safe if possible, or just raw open.
    # Assuming standard format: JSON or TSV
    try:
        with open(args.training_queries, 'r') as f:
            if args.training_queries.endswith(".json"):
                queries = json.load(f)
            else:
                 for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        queries[parts[0]] = parts[1]
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        return

    # 2. Determine Oracle Labels
    evaluator = TRECEvaluator(metrics=['map'])
    
    logger.info("Evaluating/Labeling...")
    # Using the "evaluate_per_query" logic
    # Note: Evaluating 7000 queries is slow. If we already had stats it would be faster.
    # But let's verify.
    
    dense_map = evaluate_per_query(dense_run, qrels, evaluator)
    msmeqe_map = evaluate_per_query(msmeqe_run, qrels, evaluator)
    
    common = set(dense_map.keys()) & set(msmeqe_map.keys()) & set(queries.keys())
    logger.info(f"Processing {len(common)} queries")
    
    # 3. Extract Features
    # We need FeatureExtractor
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    feature_extractor = FeatureExtractor(collection_size=args.collection_size)
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=None,
        emb_extractor=None
    )
    
    features_list = []
    labels_list = []
    
    for qid in tqdm(common):
        d_score = dense_map[qid]['map']
        m_score = msmeqe_map[qid]['map']
        
        # Label 1 (Expand) if MS-MEQE is BETTER (strict >)
        # We can also add a margin threshold
        label = 1 if m_score > d_score else 0
        
        # Extract features
        qtext = queries[qid]
        stats = candidate_extractor.compute_query_stats(qtext)
        feats = feature_extractor.extract_query_features(qtext, stats)
        
        features_list.append(feats)
        labels_list.append(label)
        
    X = np.vstack(features_list)
    y = np.array(labels_list)
    
    # Save
    logger.info(f"Saving data to {args.output}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Format compatible with train_budget_model.py
    # budgets: raw values
    # budget_classes: indices
    # We will map: 0 -> 0 (No Exp), 1 -> 50 (Expand)
    
    budget_map = {0: 0, 1: 50}
    y_budgets = np.array([budget_map[l] for l in y])
    
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump({
            'features': X,
            'budgets': y_budgets,
            'budget_classes': y, # 0 or 1
            'budget_to_class': {0:0, 50:1},
            'class_to_budget': {0:0, 1:50}
        }, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    from src.utils.lucene_utils import initialize_lucene
    initialize_lucene(lucene_path="./lucene_jars")
    main()

#!/usr/bin/env python3
# scripts/robust_5fold_train_eval.py

import sys
import os
import logging
import argparse
import numpy as np
import json
import joblib
import xgboost as xgb
import pickle
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

# Ensure proper import resolution
sys.path.append(str(Path(__file__).parent.parent))

# Standard MS-MEQE Imports
from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.retrieval.evaluator import TRECEvaluator
from src.models.budget_predictor import BudgetPredictor
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm
from src.utils.lucene_utils import initialize_lucene
from src.utils.file_utils import load_json, load_qrels

# Generator imports (assumes they are present in scripts/ dir)
try:
    from scripts.create_training_data_msmeqe import TrainingDataGenerator
    from scripts.train_budget_model import BudgetTrainingDataGenerator
except ImportError:
    # If scripts is not in path, try adding it
    sys.path.append(str(Path(__file__).parent))
    from scripts.create_training_data_msmeqe import TrainingDataGenerator
    from scripts.train_budget_model import BudgetTrainingDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_xgboost_regressor(X, y, do_tuning=False):
    """Train an XGBoost regressor with optional tuning."""
    if do_tuning:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1.0]
        }
        grid = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X, y)
        logger.info(f"Best params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        return model

def train_xgboost_classifier(X, y, num_classes, do_tuning=False):
    """Train an XGBoost classifier for budget prediction."""
    if num_classes <= 1:
        logger.warning(f"Only {num_classes} found for budget classification. Using ConstantModel.")
        from src.models.budget_predictor import ConstantModel
        # Use first label if available, else 0
        val = y[0] if len(y) > 0 else 0
        return ConstantModel(val, n_features=X.shape[1])

    if do_tuning:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6],
            'learning_rate': [0.1]
        }
        objective = 'multi:softmax' if num_classes > 2 else 'binary:logistic'
        grid = GridSearchCV(xgb.XGBClassifier(objective=objective, num_class=num_classes if num_classes > 2 else None, random_state=42), 
                           param_grid, cv=3)
        grid.fit(X, y)
        logger.info(f"Best classifier params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        objective = 'multi:softmax' if num_classes > 2 else 'binary:logistic'
        model = xgb.XGBClassifier(objective=objective, num_class=num_classes if num_classes > 2 else None, n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

class Robust5FoldTrainer:
    def __init__(self, args):
        self.args = args
        # Custom Evaluator with requested metrics
        self.evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_20', 'P_20'])
        
        # Initialize Lucene
        initialize_lucene(args.lucene_path)
        
        # Initialize Encoder
        self.encoder = SemanticEncoder(model_name=args.sbert_model)
        
        # Initialize Feature Extractor
        self.feature_extractor = FeatureExtractor(collection_size=args.collection_size)
        
        # Initialize Candidate Extractors
        from src.expansion.kb_expansion import KBCandidateExtractor
        from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
        kb_extractor = KBCandidateExtractor(args.kb_data)
        emb_extractor = EmbeddingCandidateExtractor(self.encoder, args.emb_vocab)
        
        # Initialize Candidate Extractor
        self.candidate_extractor = MultiSourceCandidateExtractor(
            index_path=args.index,
            encoder=self.encoder,
            kb_extractor=kb_extractor,
            emb_extractor=emb_extractor,
        )
        
        # Load Queries and Qrels
        self.all_queries = self._load_queries(args.topics)
        self.qrels = load_qrels(args.qrels)
        
        # Load KB candidates map (Precomputed candidates in custom format)
        self.kb_candidates_map = {}
        if args.kb_data:
             logger.info(f"Loading precomputed KB candidates from {args.kb_data}")
             with open(args.kb_data, 'r') as f:
                 for line in f:
                     try:
                         data = json.loads(line)
                         self.kb_candidates_map[data['qid']] = data['candidates']
                     except:
                         continue
             logger.info(f"Loaded KB candidates for {len(self.kb_candidates_map)} queries.")
        
        # Filter queries to those that have qrels
        self.qids = sorted([qid for qid in self.all_queries.keys() if qid in self.qrels])
        logger.info(f"Loaded {len(self.qids)} queries with relevance judgments.")

    def _load_queries(self, path):
         # Robust04 TSV: qid\ttitle\tdesc\tnarr
         queries = {}
         with open(path, 'r') as f:
             for line in f:
                 parts = line.strip().split('\t')
                 if len(parts) >= 2:
                     queries[parts[0]] = parts[1]
         return queries

    def run_cv(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = []
        qid_array = np.array(self.qids)
        
        output_base = Path(self.args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(qid_array)):
            logger.info(f"=== FOLD {fold+1}/5 ===")
            train_qids = qid_array[train_idx]
            test_qids = qid_array[test_idx]
            
            train_queries_dict = {qid: self.all_queries[qid] for qid in train_qids}
            
            # 1. Generate Training Data (Value/Weight)
            logger.info("Generating Value/Weight training data...")
            val_weight_gen = TrainingDataGenerator(
                encoder=self.encoder,
                feature_extractor=self.feature_extractor,
                candidate_extractor=self.candidate_extractor,
                index_path=self.args.index,
                collection_size=self.args.collection_size
            )
            
            fold_data_dir = output_base / f"fold_{fold+1}"
            fold_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save
            val_weight_gen.generate_training_data(
                queries=train_queries_dict,
                qrels=self.qrels,
                output_dir=str(fold_data_dir),
                max_candidates_per_query=20,
                kb_candidates_map=self.kb_candidates_map
            )
            
            # Load back the data
            v_data = joblib.load(fold_data_dir / "value_training_data.pkl")
            w_data = joblib.load(fold_data_dir / "weight_training_data.pkl")
            
            # 2. Train Value/Weight Models
            logger.info("Training Value/Weight models...")
            value_model = train_xgboost_regressor(v_data['features'], v_data['targets'], do_tuning=self.args.tune)
            weight_model = train_xgboost_regressor(w_data['features'], w_data['targets'], do_tuning=self.args.tune)
            
            # 3. Generate Budget Training Data
            logger.info("Generating Budget training data...")
            budget_gen = BudgetTrainingDataGenerator(
                encoder=self.encoder,
                feature_extractor=self.feature_extractor,
                candidate_extractor=self.candidate_extractor,
                value_model=value_model,
                weight_model=weight_model,
                index_path=self.args.index,
                collection_size=self.args.collection_size
            )
            
            # We must use real retrieval here
            # But BudgetTrainingDataGenerator._init_retrieval_system has a bug (forces simulation)
            # WORKAROUND: manually set doc_embeddings if they were loaded in val_weight_gen
            budget_gen.doc_embeddings = val_weight_gen.doc_embeddings
            budget_gen.doc_ids = val_weight_gen.doc_ids
            
            X_b, y_b_encoded = budget_gen.generate_training_data(
                queries=train_queries_dict,
                qrels=self.qrels,
                output_path=str(fold_data_dir / "budget_training_data.pkl"),
                kb_candidates_map=self.kb_candidates_map
            )
            
            # 4. Train Budget Model
            logger.info("Training Budget model...")
            all_budgets = sorted(budget_gen.budget_options)
            class_to_budget = {i: b for i, b in enumerate(all_budgets)}
            
            # Use LabelEncoder to handle only present classes correctly
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # y_b_encoded already contains class indices for all possible budgets, 
            # but some budget options might not have been selected as "best" for any query.
            # However, the classifier should be trained on the actual labels provided.
            le.fit(y_b_encoded)
            y_b_final = le.transform(y_b_encoded)
            num_classes_found = len(le.classes_)
            
            logger.info(f"Budget training data: {X_b.shape}, labels distribution: {Counter(y_b_final)}")
            
            raw_budget_model = train_xgboost_classifier(X_b, y_b_final, num_classes_found, do_tuning=self.args.tune)
            
            # Remap class_to_budget for the predictor
            remapped_class_to_budget = {
                i: class_to_budget[le.classes_[i]]
                for i in range(num_classes_found)
            }
            budget_predictor = BudgetPredictor(raw_budget_model, remapped_class_to_budget)
            
            # 5. Evaluate on Test Fold
            logger.info(f"Evaluating on {len(test_qids)} test queries...")
            msmeqe_model = MSMEQEExpansionModel(
                encoder=self.encoder,
                value_model=value_model,
                weight_model=weight_model,
                budget_model=budget_predictor,
                collection_size=self.args.collection_size
            )
            
            fold_results = {}
            for qid in tqdm(test_qids, desc=f"Fold {fold+1} Eval"):
                query_text = self.all_queries[qid]
                
                # Get query stats and centroid
                stats = self.candidate_extractor.compute_query_stats(query_text)
                centroid = self.candidate_extractor.compute_pseudo_centroid(query_text)
                
                # Get candidates
                candidates = self.candidate_extractor.extract_all_candidates(
                    query_text, 
                    qid,
                    kb_override=self.kb_candidates_map.get(qid)
                )
                
                if not candidates:
                    q_star = self.encoder.encode([query_text])[0]
                else:
                    selected, q_star = msmeqe_model.expand(
                        query_text=query_text,
                        candidates=candidates,
                        pseudo_doc_centroid=centroid,
                        query_stats=stats
                    )
                
                # Dense Retrieval
                q_norm = q_star / (np.linalg.norm(q_star) + 1e-12)
                similarities = np.dot(val_weight_gen.doc_embeddings, q_norm)
                
                top_k = min(1000, len(similarities))
                if len(similarities) > top_k:
                    top_idx_unsorted = np.argpartition(-similarities, top_k)[:top_k]
                    top_indices = top_idx_unsorted[np.argsort(-similarities[top_idx_unsorted])]
                else:
                    top_indices = np.argsort(-similarities)
                
                fold_results[qid] = {
                    val_weight_gen.doc_ids[idx]: float(similarities[idx])
                    for idx in top_indices
                }
            
            # Format results for TRECEvaluator (Dict[str, List[Tuple[str, float]]])
            formatted_results = {
                qid: sorted(d_scores.items(), key=lambda x: x[1], reverse=True) 
                for qid, d_scores in fold_results.items()
            }
            
            eval_metrics = self.evaluator.evaluate_run(formatted_results, self.qrels)
            logger.info(f"Fold {fold+1} Results: MAP={eval_metrics['map']:.4f}, nDCG@20={eval_metrics['ndcg_cut_20']:.4f}, P@20={eval_metrics['P_20']:.4f}")
            results.append(eval_metrics)

            # Cleanup memory if needed
            del val_weight_gen
            del budget_gen

        # Aggregate Results
        avg_metrics = {}
        for metric in results[0].keys():
            avg_metrics[metric] = np.mean([r[metric] for r in results])
        
        logger.info("\n" + "="*40)
        logger.info("=== FINAL CV RESULTS (Direct Robust04 Training) ===")
        for m, v in avg_metrics.items():
            logger.info(f"{m:15}: {v:.4f}")
        logger.info("="*40)
        
        # Save summary
        with open(output_base / "cv_summary.json", 'w') as f:
            json.dump({
                "folds": results,
                "average": avg_metrics
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Robust04 5-Fold CV Training for MS-MEQE")
    parser.add_argument("--index", required=True, help="Path to index directory")
    parser.add_argument("--topics", required=True, help="Path to Robust04 topics TSV")
    parser.add_argument("--qrels", required=True, help="Path to Robust04 qrels")
    parser.add_argument("--kb-data", required=True, help="Path to KB candidates")
    parser.add_argument("--emb-vocab", required=True, help="Path to embedding vocab")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--lucene-path", default="lucene_jars")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection-size", type=int, default=528155)
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    
    args = parser.parse_args()
    
    trainer = Robust5FoldTrainer(args)
    trainer.run_cv()

if __name__ == "__main__":
    main()

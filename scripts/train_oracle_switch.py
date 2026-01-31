
import argparse
import logging
import pickle
# import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.models.budget_predictor import BudgetPredictor

logger = logging.getLogger(__name__)

def train_oracle_model(data_path, output_dir):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    X = data['features']
    y = data['budget_classes'] # 0 or 1
    budget_to_class = data['budget_to_class'] # {0:0, 50:1}
    class_to_budget = data['class_to_budget'] # {0:0, 1:50}
    
    logger.info(f"Loaded {len(X)} samples.")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight
    # y=0 is neg, y=1 is pos
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_val, preds, target_names=["No Expand", "Expand"]))
    
    # Wrap
    # We need to make sure class_to_budget matches what BudgetPredictor expects
    # data['class_to_budget'] is perfect {0:0, 1:50}
    wrapped = BudgetPredictor(model, class_to_budget)
    
    out_path = Path(output_dir) / "budget_model.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(wrapped, f)
    logger.info(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    train_oracle_model(args.data, args.output_dir)

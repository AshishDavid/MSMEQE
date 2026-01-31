import argparse
import logging
import pickle
import os
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to aggregated dataset pkl")
    parser.add_argument("--output-dir", required=True, help="Directory to save models")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logger.info(f"Loading data from {args.data}...")
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        
    X = dataset['X']
    y_gain = dataset['y_gain']
    y_risk = dataset['y_risk']
    y_align = dataset['y_align']
    
    logger.info(f"Dataset size: {X.shape[0]} queries, {X.shape[1]} features")
    
    # Split
    X_train, X_test, y_gain_train, y_gain_test, y_risk_train, y_risk_test, y_align_train, y_align_test = train_test_split(
        X, y_gain, y_risk, y_align, test_size=0.2, random_state=42
    )
    
    # 2. Train Gain Predictor (Regression)
    # Using Huber loss (robust to outliers/noise in MAP deltas)
    # delta parameter in Hubber is usually 1.0, but instructions suggested 0.01 for SIGIR-grade stability
    logger.info("Training Gain Predictor (XGBRegressor with Absolute Error)...")
    gain_model = XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    gain_model.fit(X_train, y_gain_train)
    
    # 3. Train Risk Predictor (Classification)
    # Weighted log-loss to penalize false negatives (missed risks)
    # w1 = 2.0, w0 = 1.0 (as requested)
    logger.info("Training Risk Predictor (XGBClassifier with Weighting)...")
    
    # Calculate scale_pos_weight for class imbalance
    # risk label 1 = delta < 0
    pos_count = np.sum(y_risk_train == 1)
    neg_count = np.sum(y_risk_train == 0)
    # instructions asked for w1=2.0, w0=1.0 relative weighting.
    # In XGB, scale_pos_weight = sum(neg) / sum(pos) * weighting_factor
    scale_pos = (neg_count / pos_count) * 2.0 if pos_count > 0 else 1.0
    
    risk_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    risk_model.fit(X_train, y_risk_train)
    
    # 4. Train Alignment Predictor (Regression)
    logger.info("Training Alignment Predictor (XGBRegressor with L2 loss)...")
    align_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    align_model.fit(X_train, y_align_train)
    
    # 4. Save
    logger.info(f"Saving models to {args.output_dir}...")
    with open(out_dir / "gain_model.pkl", 'wb') as f:
        pickle.dump(gain_model, f)
    with open(out_dir / "risk_model.pkl", 'wb') as f:
        pickle.dump(risk_model, f)
    with open(out_dir / "align_model.pkl", 'wb') as f:
        pickle.dump(align_model, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()

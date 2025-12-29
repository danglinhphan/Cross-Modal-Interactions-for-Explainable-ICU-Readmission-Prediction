#!/usr/bin/env python3
"""
Lightweight Balanced EBM Ensemble (GAM-Bagging)
===============================================
Trains a Bagging Ensemble of EBM models without interactions (GAM mode).
 Optimized for speed and handling strict class imbalance.

Safeguards:
- Saves to a NEW directory by default (outputs/ebm_balanced_gam)
- Suppresses C++ logging.
- Uses serial execution (n_jobs=1) to prevent crashing on Mac.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime

# Aggressively suppress logging before importing interpret
os.environ['INTERPRET_LOG_LEVEL'] = 'OFF'

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, 
    confusion_matrix
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

# Setup python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EBMEnsemble:
    """
    Bagging Ensemble of Explainable Boosting Machines.
    """
    def __init__(
        self, 
        n_estimators: int = 20,
        undersampling_ratio: float = 1.0,
        interactions: int = 0,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        self.n_estimators = n_estimators
        self.undersampling_ratio = undersampling_ratio
        self.interactions = interactions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []
        self.feature_names = None
        self.feature_types = None
        
    def fit(self, X, y):
        self.models = []
        self.feature_names = list(X.columns)
        
        logger.info(f"Training Ensemble with {self.n_estimators} estimators...")
        logger.info(f"Undersampling Ratio: {self.undersampling_ratio}")
        logger.info(f"Interactions: {self.interactions}")
        
        for i in range(self.n_estimators):
            # 1. Undersample
            seed = self.random_state + i
            rus = RandomUnderSampler(
                sampling_strategy=1.0/self.undersampling_ratio,
                random_state=seed
            )
            X_res, y_res = rus.fit_resample(X, y)
            
            # 2. Train EBM
            ebm = ExplainableBoostingClassifier(
                random_state=seed, 
                n_jobs=self.n_jobs,
                interactions=self.interactions,
                outer_bags=8,
                inner_bags=0,
                max_bins=256
            )
            
            if i % 5 == 0:
                logger.info(f"  Training model {i+1}/{self.n_estimators} (Data: {len(X_res)} samples)...")
                
            ebm.fit(X_res, y_res)
            self.models.append(ebm)
            
        logger.info("Ensemble training complete.")
        return self
        
    def predict_proba(self, X):
        """Average probabilities from all models."""
        probas = np.zeros(len(X))
        for model in self.models:
            probas += model.predict_proba(X)[:, 1]
        
        return probas / len(self.models)
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

def load_data(vital_path, nlp_path, labels_path):
    logger.info(f"Loading data...")
    
    # Load
    df_vital = pd.read_csv(vital_path)
    df_nlp = pd.read_csv(nlp_path)
    df_lbl = pd.read_csv(labels_path)
    
    # Standardize ID
    for df in [df_vital, df_nlp, df_lbl]:
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
            
    # Merge
    df_merged = df_vital.merge(df_nlp, on='HADM_ID', how='inner')
    df_full = df_merged.merge(df_lbl, on='HADM_ID', how='inner')
    
    # Features X and Target y
    target_col = 'Y' if 'Y' in df_full.columns else 'LABEL'
    y = df_full[target_col]
    X = df_full.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    X = X.fillna(0)
    
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital', default='cohort/features_phase4_clinical.csv')
    parser.add_argument('--nlp', default='cohort/nlp_features_enhanced.csv')
    parser.add_argument('--labels', default='cohort/new_cohort_icu_readmission_labels.csv')
    parser.add_argument('--output', default='outputs/ebm_balanced_gam')
    parser.add_argument('--estimators', type=int, default=20)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--interactions', type=int, default=0)
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data(args.vital, args.nlp, args.labels)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=170
    )
    
    # 3. Model
    ensemble = EBMEnsemble(
        n_estimators=args.estimators,
        undersampling_ratio=args.ratio, 
        interactions=args.interactions,
        random_state=170
    )
    ensemble.fit(X_train, y_train)
    
    # 4. Evaluate
    logger.info("Evaluating on Test Set...")
    y_proba = ensemble.predict_proba(X_test)
    
    # Calculate Metrics at 0.5 threshold (Initial check)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = {
        'threshold': 0.5,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba),
        'auprc': average_precision_score(y_test, y_proba)
    }
    
    logger.info("="*40)
    logger.info(f"Results (Threshold 0.5)")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUROC:     {metrics['auroc']:.4f}")
    logger.info("="*40)
    
    # Save
    save_path = os.path.join(args.output, 'ebm_ensemble_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

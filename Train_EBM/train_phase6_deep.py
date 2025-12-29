#!/usr/bin/env python3
"""
Phase 6: Balanced Interaction Ensemble with Deep Pathological Features
======================================================================
Trains a Bagging Ensemble of EBM models (Ratio 1.0, Interactions 50)
on a dataset enriched with actual biomarkers (Troponin, CRP, etc.)
to replace/augment monitoring intensity proxies.
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

# Aggressively suppress logging
os.environ['INTERPRET_LOG_LEVEL'] = 'OFF'

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score
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
    def __init__(
        self, 
        n_estimators: int = 10,
        undersampling_ratio: float = 1.0,
        interactions: int = 50,
        random_state: int = 42,
        n_jobs: int = 1
    ):
        self.n_estimators = n_estimators
        self.undersampling_ratio = undersampling_ratio
        self.interactions = interactions
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []
        
    def fit(self, X, y):
        self.models = []
        
        logger.info(f"Training Deep Ensemble with {self.n_estimators} estimators...")
        logger.info(f"Undersampling Ratio: {self.undersampling_ratio}")
        logger.info(f"Interactions: {self.interactions}")
        logger.info(f"Feature Count: {X.shape[1]}")
        
        for i in range(self.n_estimators):
            seed = self.random_state + i
            # 1. Undersample
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
            
            if i % 1 == 0:
                logger.info(f"  Training model {i+1}/{self.n_estimators} (Data: {len(X_res)} samples)...")
                
            ebm.fit(X_res, y_res)
            self.models.append(ebm)
            
        logger.info("Ensemble training complete.")
        return self
        
    def predict_proba(self, X):
        probas = np.zeros(len(X))
        for model in self.models:
            probas += model.predict_proba(X)[:, 1]
        return probas / len(self.models)
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

def load_merged_data(vital_path, nlp_path, pathology_path, labels_path):
    logger.info(f"Loading and merging data...")
    
    # Load
    df_vital = pd.read_csv(vital_path)
    df_nlp = pd.read_csv(nlp_path)
    df_pathology = pd.read_csv(pathology_path)
    df_lbl = pd.read_csv(labels_path)
    
    # Standardize ID
    for df in [df_vital, df_nlp, df_pathology, df_lbl]:
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
            
    # Merge Sequence: Vital -> NLP -> Pathology -> Labels
    df_merged = df_vital.merge(df_nlp, on='HADM_ID', how='inner')
    df_merged = df_merged.merge(df_pathology, on='HADM_ID', how='left') # Left merge key!
    
    # Fill NaN for pathology (some patients might not have these specific labs)
    # We impute with -1 as "Missing" indicator for EBM to learn.
    df_merged = df_merged.fillna(-1)
    
    df_full = df_merged.merge(df_lbl, on='HADM_ID', how='inner')
    
    # Features X and Target y
    target_col = 'Y' if 'Y' in df_full.columns else 'LABEL'
    y = df_full[target_col]
    X = df_full.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital', default='cohort/features_phase4_clinical.csv')
    parser.add_argument('--nlp', default='cohort/nlp_features_enhanced.csv')
    parser.add_argument('--pathology', default='cohort/new_pathology_features.csv')
    parser.add_argument('--labels', default='cohort/new_cohort_icu_readmission_labels.csv')
    parser.add_argument('--output', default='outputs/ebm_phase6_deep')
    
    parser.add_argument('--estimators', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--interactions', type=int, default=50)
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load Data
    X, y = load_merged_data(args.vital, args.nlp, args.pathology, args.labels)
    
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
    logger.info(f"Results (Deep Pathology)")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("="*40)
    
    # Save
    save_path = os.path.join(args.output, 'ebm_ensemble_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

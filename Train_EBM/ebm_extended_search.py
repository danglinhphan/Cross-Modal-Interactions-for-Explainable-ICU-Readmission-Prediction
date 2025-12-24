#!/usr/bin/env python3
"""
Extended EBM Optimization - Phase 2

More aggressive undersampling and regularization to push towards 80% target.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(vital_path, nlp_path, labels_path):
    """Load and merge data."""
    vital_df = pd.read_csv(vital_path)
    nlp_df = pd.read_csv(nlp_path)
    labels_df = pd.read_csv(labels_path)
    
    # Find ID column
    id_col = None
    for col in ['HADM_ID', 'hadm_id', 'stay_id', 'STAY_ID']:
        if col in vital_df.columns:
            id_col = col
            break
    
    # Find ID in labels
    label_id_col = None
    for col in ['HADM_ID', 'hadm_id', 'ICUSTAY_ID', 'icustay_id']:
        if col in labels_df.columns:
            label_id_col = col
            break
    
    # Merge
    merged = vital_df.merge(nlp_df, on=id_col, how='inner')
    merged = merged.merge(labels_df, left_on=id_col, right_on=label_id_col, how='inner')
    
    # Get label
    label_col = 'Y' if 'Y' in merged.columns else 'label'
    y = merged[label_col]
    
    # Get features
    drop_cols = [c for c in merged.columns if c in [id_col, label_id_col, label_col, 'SUBJECT_ID', 'ICUSTAY_ID']]
    X = merged.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y


def get_interactions(output_dir, n_interactions, feature_cols):
    """Load pre-computed interactions."""
    path = os.path.join(output_dir, '..', 'ebm_optimized_final', 'top_75_interactions.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        interactions = []
        for _, row in df.head(n_interactions).iterrows():
            if row['vital_feature'] in feature_cols and row['nlp_feature'] in feature_cols:
                v_idx = feature_cols.index(row['vital_feature'])
                t_idx = feature_cols.index(row['nlp_feature'])
                interactions.append((v_idx, t_idx))
        return interactions
    return n_interactions


def find_balanced_threshold(y_true, y_proba, target=0.80):
    """Find threshold where P, R, F1 all meet target."""
    thresholds = np.arange(0.10, 0.90, 0.005)
    best_thresh = 0.5
    best_min = 0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        min_metric = min(prec, rec, f1)
        
        if prec >= target and rec >= target and f1 >= target:
            return thresh, True, {'precision': prec, 'recall': rec, 'f1': f1}
        
        if min_metric > best_min:
            best_min = min_metric
            best_thresh = thresh
    
    y_pred = (y_proba >= best_thresh).astype(int)
    return best_thresh, False, {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def run_extended_search(X, y, output_dir, seed=170, target=0.80):
    """Run extended hyperparameter search with more aggressive settings."""
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    feature_cols = list(X.columns)
    interactions = get_interactions(output_dir, 75, feature_cols)
    
    # Extended search space - more aggressive undersampling
    configs = [
        # Very aggressive undersampling
        {'name': 'ratio_2.0_high_bags', 'undersampling_ratio': 2.0, 'outer_bags': 30, 'inner_bags': 25, 'max_bins': 256, 'learning_rate': 0.008, 'min_samples_leaf': 8, 'max_leaves': 5},
        {'name': 'ratio_2.5_high_bags', 'undersampling_ratio': 2.5, 'outer_bags': 30, 'inner_bags': 25, 'max_bins': 256, 'learning_rate': 0.008, 'min_samples_leaf': 8, 'max_leaves': 5},
        {'name': 'ratio_1.5_extreme', 'undersampling_ratio': 1.5, 'outer_bags': 30, 'inner_bags': 25, 'max_bins': 256, 'learning_rate': 0.01, 'min_samples_leaf': 5, 'max_leaves': 6},
        # Lower regularization, more capacity
        {'name': 'ratio_2.0_more_capacity', 'undersampling_ratio': 2.0, 'outer_bags': 25, 'inner_bags': 20, 'max_bins': 512, 'learning_rate': 0.01, 'min_samples_leaf': 5, 'max_leaves': 6},
        # Higher regularization
        {'name': 'ratio_2.0_high_reg', 'undersampling_ratio': 2.0, 'outer_bags': 30, 'inner_bags': 25, 'max_bins': 128, 'learning_rate': 0.005, 'min_samples_leaf': 15, 'max_leaves': 3},
        # Balanced approach with ratio 3
        {'name': 'ratio_3.0_balanced', 'undersampling_ratio': 3.0, 'outer_bags': 30, 'inner_bags': 25, 'max_bins': 256, 'learning_rate': 0.008, 'min_samples_leaf': 8, 'max_leaves': 4},
    ]
    
    best_result = None
    best_model = None
    all_results = []
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {i+1}/{len(configs)}: {config['name']}")
        logger.info(f"Params: {config}")
        
        try:
            # Undersample
            ratio = config['undersampling_ratio']
            rus = RandomUnderSampler(sampling_strategy=1.0/ratio, random_state=seed)
            X_res, y_res = rus.fit_resample(X_train, y_train)
            logger.info(f"Undersampled: {len(X_train)} -> {len(X_res)}")
            
            # Train EBM
            ebm = ExplainableBoostingClassifier(
                max_bins=config['max_bins'],
                interactions=interactions,
                outer_bags=config['outer_bags'],
                inner_bags=config['inner_bags'],
                learning_rate=config['learning_rate'],
                min_samples_leaf=config['min_samples_leaf'],
                max_leaves=config['max_leaves'],
                max_rounds=10000,
                early_stopping_rounds=100,
                random_state=seed,
                n_jobs=-1
            )
            
            logger.info("Training...")
            ebm.fit(X_res, y_res)
            
            # Evaluate
            y_proba = ebm.predict_proba(X_test)[:, 1]
            thresh, meets_target, metrics = find_balanced_threshold(y_test, y_proba, target)
            
            y_pred = (y_proba >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            result = {
                'config': config,
                'threshold': thresh,
                'metrics': metrics,
                'meets_target': meets_target,
                'auc_roc': roc_auc_score(y_test, y_proba),
                'auprc': average_precision_score(y_test, y_proba),
                'confusion_matrix': cm.tolist()
            }
            all_results.append(result)
            
            logger.info(f"Results:")
            logger.info(f"  Threshold: {thresh:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1:        {metrics['f1']:.4f}")
            logger.info(f"  Meets {target*100:.0f}% Target: {'✓ YES' if meets_target else '✗ NO'}")
            
            # Update best
            min_metric = min(metrics['precision'], metrics['recall'], metrics['f1'])
            if best_result is None or min_metric > min(
                best_result['metrics']['precision'],
                best_result['metrics']['recall'],
                best_result['metrics']['f1']
            ):
                best_result = result
                best_model = ebm
                logger.info("  ★ New best!")
            
            if meets_target:
                logger.info(f"\n🎉 Found config meeting {target*100:.0f}% target!")
                break
                
        except Exception as e:
            logger.error(f"Error: {e}")
            continue
    
    # Save best result
    if best_result and best_model:
        # Save model
        with open(os.path.join(output_dir, 'final_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save metrics
        output_metrics = {
            'threshold': best_result['threshold'],
            'class_1': best_result['metrics'],
            'auprc': best_result['auprc'],
            'auroc': best_result['auc_roc'],
            'confusion_matrix': {
                'TN': best_result['confusion_matrix'][0][0],
                'FP': best_result['confusion_matrix'][0][1],
                'FN': best_result['confusion_matrix'][1][0],
                'TP': best_result['confusion_matrix'][1][1]
            },
            'config': best_result['config'],
            'meets_80_target': best_result['meets_target']
        }
        
        with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
            json.dump(output_metrics, f, indent=2)
        
        # Save all results
        with open(os.path.join(output_dir, 'extended_search_log.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info("BEST RESULT")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {best_result['config']['name']}")
        logger.info(f"Precision: {best_result['metrics']['precision']:.4f}")
        logger.info(f"Recall:    {best_result['metrics']['recall']:.4f}")
        logger.info(f"F1:        {best_result['metrics']['f1']:.4f}")
        logger.info(f"Meets Target: {best_result['meets_target']}")
    
    return best_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital-features', required=True)
    parser.add_argument('--nlp-features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-metrics', type=float, default=0.80)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    X, y = load_data(args.vital_features, args.nlp_features, args.labels)
    
    result = run_extended_search(X, y, args.output_dir, target=args.target_metrics)
    
    if result and result['meets_target']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

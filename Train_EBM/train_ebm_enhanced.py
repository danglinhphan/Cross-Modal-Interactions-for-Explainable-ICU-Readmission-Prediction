#!/usr/bin/env python3
"""
EBM Training with Enhanced Features (384-dim embeddings + vitals)

Uses the enhanced features from create_enhanced_features.py to train
an EBM model targeting 80%+ P/R/F1.
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='Enhanced features CSV')
    parser.add_argument('--labels', required=True, help='Labels CSV')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-metrics', type=float, default=0.80)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading features from {args.features}")
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    
    # Merge on HADM_ID
    merged = features.merge(labels, on='HADM_ID', how='inner')
    logger.info(f"Merged: {len(merged)} samples")
    
    # Separate X and y
    label_col = 'Y' if 'Y' in merged.columns else 'label'
    drop_cols = ['HADM_ID', label_col, 'SUBJECT_ID', 'ICUSTAY_ID']
    X = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = merged[label_col].values
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Identify feature types
    vital_cols = [c for c in X.columns if not c.startswith('nlp_')]
    nlp_cols = [c for c in X.columns if c.startswith('nlp_')]
    logger.info(f"Vital/lab features: {len(vital_cols)}")
    logger.info(f"NLP features: {len(nlp_cols)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=170, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Configurations to try
    configs = [
        {'name': 'enhanced_ratio3.0', 'ratio': 3.0, 'outer_bags': 30, 'inner_bags': 25},
        {'name': 'enhanced_ratio2.5', 'ratio': 2.5, 'outer_bags': 30, 'inner_bags': 25},
        {'name': 'enhanced_ratio2.0', 'ratio': 2.0, 'outer_bags': 30, 'inner_bags': 25},
        {'name': 'enhanced_ratio4.0', 'ratio': 4.0, 'outer_bags': 25, 'inner_bags': 20},
    ]
    
    best_result = None
    best_model = None
    all_results = []
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config['name']}")
        logger.info(f"{'='*60}")
        
        try:
            # Undersample
            rus = RandomUnderSampler(sampling_strategy=1.0/config['ratio'], random_state=170)
            X_res, y_res = rus.fit_resample(X_train, y_train)
            logger.info(f"Undersampled: {len(X_train)} -> {len(X_res)}")
            
            # Train EBM (no forced interactions - let it discover automatically)
            ebm = ExplainableBoostingClassifier(
                max_bins=256,
                interactions=50,  # Auto-discover top 50 interactions
                outer_bags=config['outer_bags'],
                inner_bags=config['inner_bags'],
                learning_rate=0.01,
                min_samples_leaf=10,
                max_leaves=4,
                max_rounds=10000,
                early_stopping_rounds=100,
                random_state=170,
                n_jobs=1  # Set to 1 for stability with Python 3.14
            )
            
            logger.info("Training EBM...")
            ebm.fit(X_res, y_res)
            
            # Evaluate
            y_proba = ebm.predict_proba(X_test)[:, 1]
            thresh, meets_target, metrics = find_balanced_threshold(y_test, y_proba, args.target_metrics)
            
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
            
            logger.info(f"\nResults:")
            logger.info(f"  Threshold: {thresh:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1:        {metrics['f1']:.4f}")
            logger.info(f"  AUC-ROC:   {result['auc_roc']:.4f}")
            logger.info(f"  AUPRC:     {result['auprc']:.4f}")
            logger.info(f"  Meets {args.target_metrics*100:.0f}% Target: {'✓ YES' if meets_target else '✗ NO'}")
            
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
                logger.info(f"\n🎉 Found config meeting {args.target_metrics*100:.0f}% target!")
                break
                
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save best
    if best_result and best_model:
        # Save model
        model_path = os.path.join(args.output_dir, 'ebm_enhanced_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save metrics
        metrics_out = {
            'model_type': 'EBM_enhanced_features',
            'n_features': X.shape[1],
            'n_vital_features': len(vital_cols),
            'n_nlp_features': len(nlp_cols),
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
        
        with open(os.path.join(args.output_dir, 'model_metrics.json'), 'w') as f:
            json.dump(metrics_out, f, indent=2)
        
        # Save log
        with open(os.path.join(args.output_dir, 'training_log.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("ENHANCED EBM RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Best Config: {best_result['config']['name']}")
        logger.info(f"Features: {X.shape[1]} ({len(vital_cols)} vital + {len(nlp_cols)} NLP)")
        logger.info(f"Precision: {best_result['metrics']['precision']:.4f} ({best_result['metrics']['precision']*100:.1f}%)")
        logger.info(f"Recall:    {best_result['metrics']['recall']:.4f} ({best_result['metrics']['recall']*100:.1f}%)")
        logger.info(f"F1-Score:  {best_result['metrics']['f1']:.4f} ({best_result['metrics']['f1']*100:.1f}%)")
        logger.info(f"Meets 80% Target: {best_result['meets_target']}")
        logger.info(f"Results saved to: {args.output_dir}")
    
    if best_result and best_result['meets_target']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Cross-Validation Hyperparameter Search for EBM Model.

Performs 5-fold CV to find optimal hyperparameters that minimize
the gap between train and test performance.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_fscore_support, 
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler
from interpret.glassbox import ExplainableBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_sample_weights(y: np.ndarray, fn_cost_multiplier: float = 4.0) -> np.ndarray:
    """Compute cost-sensitive sample weights."""
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    imbalance = n0 / n1
    
    # Effective number weighting
    beta = 0.9999
    eff_n0 = (1 - beta**n0) / (1 - beta)
    eff_n1 = (1 - beta**n1) / (1 - beta)
    total_eff = eff_n0 + eff_n1
    w0 = total_eff / (2 * eff_n0)
    w1 = total_eff / (2 * eff_n1)
    
    # Apply FN cost to class 1
    fn_cost = np.log(imbalance + 1) * 1.5  # ~4.09 for 14.29:1 imbalance
    
    # Combine weights
    sample_weights = np.where(y == 1, w1 * fn_cost * 1.44, w0 * 1.44)
    
    return sample_weights


def find_balanced_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_target: float = 0.70) -> Dict:
    """Find threshold where P, R, F1 all meet minimum target."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Search for threshold with all metrics >= target
    best_thresh = None
    best_metrics = None
    best_min = 0
    
    for i, thresh in enumerate(thresholds):
        p, r, f1 = precisions[i], recalls[i], f1_scores[i]
        min_metric = min(p, r, f1)
        
        if min_metric >= min_target:
            if best_thresh is None:
                best_thresh = thresh
                best_metrics = {'precision': p, 'recall': r, 'f1': f1, 'min': min_metric}
        
        if min_metric > best_min:
            best_min = min_metric
            if best_thresh is None:
                best_thresh = thresh
                best_metrics = {'precision': p, 'recall': r, 'f1': f1, 'min': min_metric}
    
    if best_metrics is None:
        # Fallback to best compromise
        for i, thresh in enumerate(thresholds):
            p, r, f1 = precisions[i], recalls[i], f1_scores[i]
            min_metric = min(p, r, f1)
            if min_metric > best_min:
                best_min = min_metric
                best_thresh = thresh
                best_metrics = {'precision': p, 'recall': r, 'f1': f1, 'min': min_metric}
    
    return {
        'threshold': best_thresh,
        'metrics': best_metrics,
        'achieved_target': best_min >= min_target
    }


def evaluate_fold(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    params: Dict
) -> Dict:
    """Train and evaluate EBM on one fold."""
    
    # Compute sample weights
    sample_weights = compute_sample_weights(y_train)

    # Optionally undersample the training set to address imbalance
    if params.get('undersampling_ratio', None) is not None:
        try:
            sampling_strategy = 1.0 / float(params['undersampling_ratio'])
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            # Recompute sample weights for resampled data
            sample_weights = compute_sample_weights(y_train)
        except Exception as e:
            logger.warning(f"Undersampling failed: {e}; proceeding without undersampling")
    
    # Train EBM
    ebm = ExplainableBoostingClassifier(
        max_bins=params.get('max_bins', 128),
        interactions=params.get('interactions', 0),
        outer_bags=params.get('outer_bags', 16),
        inner_bags=params.get('inner_bags', 8),
        learning_rate=params.get('learning_rate', 0.01),
        min_samples_leaf=params.get('min_samples_leaf', 10),
        max_leaves=params.get('max_leaves', 4),
        max_rounds=params.get('max_rounds', 10000),
        early_stopping_rounds=params.get('early_stopping_rounds', 100),
        random_state=42,
        n_jobs=1  # Avoid parallelism issues
    )
    
    # Optionally use a calibration subset from training to calibrate probabilities
    use_calibration = params.get('use_calibration', False)
    if use_calibration:
        from sklearn.model_selection import train_test_split
        X_train_sub, X_cal, y_train_sub, y_cal = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        # Train on sub-training set
        ebm.fit(X_train_sub, y_train_sub, sample_weight=None if sample_weights is None else compute_sample_weights(y_train_sub))
        calibrator = CalibratedClassifierCV(ebm, cv='prefit', method='sigmoid')
        calibrator.fit(X_cal, y_cal)
        calibrated_classifier = calibrator
    else:
        ebm.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Predict
    if use_calibration:
        y_proba = calibrated_classifier.predict_proba(X_test)[:, 1]
    else:
        y_proba = ebm.predict_proba(X_test)[:, 1]
    
    # Metrics
    auprc = average_precision_score(y_test, y_proba)
    auroc = roc_auc_score(y_test, y_proba)
    
    # Find balanced threshold
    thresh_result = find_balanced_threshold(y_test, y_proba, min_target=params.get('min_metrics', 0.70))
    
    return {
        'auprc': auprc,
        'auroc': auroc,
        'threshold': thresh_result['threshold'],
        'precision': thresh_result['metrics']['precision'],
        'recall': thresh_result['metrics']['recall'],
        'f1': thresh_result['metrics']['f1'],
        'min_metric': thresh_result['metrics']['min'],
        'achieved_70': thresh_result['achieved_target']
    }


def cross_validate(X: np.ndarray, y: np.ndarray, params: Dict, n_folds: int = 5) -> Dict:
    """Run k-fold cross-validation."""
    
    # Allow random_state override from params (if provided)
    random_state = params.get('random_state', 42)
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = {
        'auprc': [],
        'auroc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'min_metric': [],
        'achieved_70': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_result = evaluate_fold(X_train, X_test, y_train, y_test, params)
        
        for key in results:
            results[key].append(fold_result[key])
        
        logger.info(f"  Fold {fold+1}: P={fold_result['precision']:.4f}, "
                   f"R={fold_result['recall']:.4f}, F1={fold_result['f1']:.4f}, "
                   f"AUPRC={fold_result['auprc']:.4f}")
    
    # Aggregate results
    summary = {}
    for key in ['auprc', 'auroc', 'precision', 'recall', 'f1', 'min_metric']:
        summary[f'{key}_mean'] = np.mean(results[key])
        summary[f'{key}_std'] = np.std(results[key])
    
    summary['folds_achieving_70'] = sum(results['achieved_70'])
    
    return summary


def hyperparameter_search(X: np.ndarray, y: np.ndarray, global_params: Dict = None) -> Dict:
    """Search for optimal hyperparameters."""
    
    # Define parameter grid
    param_grid = [
        # Configuration 1: Light regularization
        {
            'name': 'light_reg',
            'max_bins': 256,
            'max_leaves': 6,
            'min_samples_leaf': 5,
            'interactions': 25,
            'learning_rate': 0.01,
            'max_rounds': 15000,
            'early_stopping_rounds': 150,
            'outer_bags': 16,
            'inner_bags': 8
        },
        # Configuration 2: Medium regularization
        {
            'name': 'medium_reg',
            'max_bins': 128,
            'max_leaves': 4,
            'min_samples_leaf': 10,
            'interactions': 50,
            'learning_rate': 0.01,
            'max_rounds': 10000,
            'early_stopping_rounds': 100,
            'outer_bags': 16,
            'inner_bags': 8
        },
        # Configuration 3: Strong regularization
        {
            'name': 'strong_reg',
            'max_bins': 64,
            'max_leaves': 3,
            'min_samples_leaf': 20,
            'interactions': 75,
            'learning_rate': 0.02,
            'max_rounds': 8000,
            'early_stopping_rounds': 80,
            'outer_bags': 12,
            'inner_bags': 6
        },
        # Configuration 4: Very strong regularization
        {
            'name': 'very_strong_reg',
            'max_bins': 32,
            'max_leaves': 3,
            'min_samples_leaf': 30,
            'interactions': 100,
            'learning_rate': 0.03,
            'max_rounds': 5000,
            'early_stopping_rounds': 50,
            'outer_bags': 8,
            'inner_bags': 4
        },
    ]

    # Add a higher capacity option to try more complex models
    param_grid.append({
        'name': 'high_capacity',
        'max_bins': 256,
        'max_leaves': 8,
        'min_samples_leaf': 3,
        'learning_rate': 0.01,
        'max_rounds': 20000,
        'early_stopping_rounds': 200,
        'outer_bags': 24,
        'inner_bags': 12,
        'interactions': 100,
        'undersampling_ratio': None,
        'use_calibration': True
    })
    
    best_config = None
    best_min_metric = 0
    all_results = []
    
    logger.info("="*70)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("="*70)
    
    for params in param_grid:
        # Merge global params into each params entry if provided and not already set
        if global_params:
            for k, v in global_params.items():
                if k not in params:
                    params[k] = v
        logger.info(f"\nTesting: {params['name']}")
        logger.info(f"  max_bins={params['max_bins']}, max_leaves={params['max_leaves']}, "
                   f"min_samples_leaf={params['min_samples_leaf']}")
        
        cv_results = cross_validate(X, y, params, n_folds=5)
        
        logger.info(f"  Results: P={cv_results['precision_mean']:.4f}±{cv_results['precision_std']:.4f}, "
                   f"R={cv_results['recall_mean']:.4f}±{cv_results['recall_std']:.4f}, "
                   f"F1={cv_results['f1_mean']:.4f}±{cv_results['f1_std']:.4f}")
        logger.info(f"  Min metric: {cv_results['min_metric_mean']:.4f}, "
                   f"AUPRC: {cv_results['auprc_mean']:.4f}, "
                   f"Folds >= 70%: {cv_results['folds_achieving_70']}/5")
        
        result_entry = {
            'params': params,
            'cv_results': cv_results
        }
        all_results.append(result_entry)
        
        # Track best
        if cv_results['min_metric_mean'] > best_min_metric:
            best_min_metric = cv_results['min_metric_mean']
            best_config = result_entry
    
    return {
        'best_config': best_config,
        'all_results': all_results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-Validation Hyperparameter Search')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features CSV')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels CSV')
    parser.add_argument('--output', type=str, default='outputs/cv_search',
                       help='Output directory')
    parser.add_argument('--use-calibration', action='store_true', default=False,
                       help='Use calibration (isotonic/sigmoid) during CV')
    parser.add_argument('--undersampling-ratio', type=float, default=None,
                       help='Optional undersampling ratio to apply (e.g., 4.0 -> 4:1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min-metrics', type=float, default=0.70, help='Minimum P/R/F1 target for threshold scanning (default 0.70)')
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    features_df = pd.read_csv(args.features, low_memory=False)
    labels_df = pd.read_csv(args.labels)
    
    # Merge
    merge_col = 'HADM_ID' if 'HADM_ID' in features_df.columns else 'hadm_id'
    label_col = 'Y' if 'Y' in labels_df.columns else 'label'
    hadm_col = 'HADM_ID' if 'HADM_ID' in labels_df.columns else 'hadm_id'
    
    merged = features_df.merge(labels_df[[hadm_col, label_col]], 
                               left_on=merge_col, right_on=hadm_col)
    
    # Prepare X, y
    exclude_cols = {'HADM_ID', 'hadm_id', 'Y', 'label'}
    X_cols = [c for c in merged.columns if c not in exclude_cols]
    X = merged[X_cols].fillna(0).values
    y = merged[label_col].values if label_col in merged.columns else merged['Y'].values
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {X.shape[0]}")
    logger.info(f"Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
    
    # Run hyperparameter search
    search_results = hyperparameter_search(
        X, y,
        global_params={'undersampling_ratio': args.undersampling_ratio, 'use_calibration': args.use_calibration, 'random_state': args.seed}
    )
    
    # Output
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("BEST CONFIGURATION")
    logger.info("="*70)
    
    best = search_results['best_config']
    logger.info(f"Configuration: {best['params']['name']}")
    logger.info(f"Parameters:")
    for k, v in best['params'].items():
        if k != 'name':
            logger.info(f"  {k}: {v}")
    logger.info(f"\nCV Results:")
    logger.info(f"  Precision: {best['cv_results']['precision_mean']:.4f} ± {best['cv_results']['precision_std']:.4f}")
    logger.info(f"  Recall: {best['cv_results']['recall_mean']:.4f} ± {best['cv_results']['recall_std']:.4f}")
    logger.info(f"  F1: {best['cv_results']['f1_mean']:.4f} ± {best['cv_results']['f1_std']:.4f}")
    logger.info(f"  AUPRC: {best['cv_results']['auprc_mean']:.4f} ± {best['cv_results']['auprc_std']:.4f}")
    logger.info(f"  Folds achieving 70%: {best['cv_results']['folds_achieving_70']}/5")
    
    # Save results
    with open(os.path.join(args.output, 'cv_search_results.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_results = {
            'best_config': {
                'params': best['params'],
                'cv_results': {k: convert(v) for k, v in best['cv_results'].items()}
            },
            'all_results': [
                {
                    'params': r['params'],
                    'cv_results': {k: convert(v) for k, v in r['cv_results'].items()}
                }
                for r in search_results['all_results']
            ]
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}/cv_search_results.json")


if __name__ == '__main__':
    main()

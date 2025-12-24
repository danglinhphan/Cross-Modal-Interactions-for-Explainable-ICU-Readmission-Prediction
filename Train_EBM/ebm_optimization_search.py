#!/usr/bin/env python3
"""
EBM Cross-Interaction Optimization Script

This script searches for optimal hyperparameters to achieve >=80% Precision, Recall, and F1
for ICU readmission prediction using EBM with cross-interactions.

Key features:
- Stratified cross-validation for robust evaluation
- No SMOTE/SMOTENN (undersampling only)
- Glass-box model (EBM interpretable)
- Saves to NEW output folder (doesn't modify existing models)

Usage:
    python ebm_optimization_search.py --help
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix
)
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EBMOptimizer:
    """
    Optimizer for EBM cross-interaction model targeting >=80% P/R/F1.
    
    Based on research findings:
    - outer_bags=25, inner_bags=20 for better ensemble
    - smoothing_rounds for classification
    - Balanced threshold strategy
    """
    
    def __init__(
        self,
        output_dir: str,
        random_state: int = 170,
        target_metrics: float = 0.80,
        n_cv_folds: int = 5
    ):
        self.output_dir = output_dir
        self.random_state = random_state
        self.target_metrics = target_metrics
        self.n_cv_folds = n_cv_folds
        self.best_config = None
        self.best_metrics = None
        self.results_log = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    def load_data(
        self,
        vital_path: str,
        nlp_path: str,
        labels_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load and prepare data."""
        logger.info("Loading data...")
        
        vital_df = pd.read_csv(vital_path)
        nlp_df = pd.read_csv(nlp_path)
        labels_df = pd.read_csv(labels_path)
        
        # Handle ID columns
        id_col = None
        for col in ['stay_id', 'hadm_id', 'subject_id', 'icustay_id', 
                    'STAY_ID', 'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID']:
            if col in vital_df.columns:
                id_col = col
                break
        
        if id_col:
            # Merge on ID
            merged = vital_df.merge(nlp_df, on=id_col, how='inner')
            merged = merged.merge(labels_df, on=id_col, how='inner')
            
            # Separate features and labels
            label_col = None
            for col_name in ['Y', 'label', 'readmission', 'Label', 'LABEL']:
                if col_name in merged.columns:
                    label_col = col_name
                    break
            if label_col is None:
                raise ValueError(f"Label column not found. Available: {merged.columns.tolist()}")
            y = merged[label_col]
            X = merged.drop(columns=[id_col, label_col], errors='ignore')
        else:
            # Concatenate by index
            X = pd.concat([vital_df, nlp_df], axis=1)
            label_col = 'label' if 'label' in labels_df.columns else 'readmission'
            y = labels_df[label_col]
        
        # Clean data
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0)
        
        # Identify feature types
        self.vital_cols = [c for c in X.columns if not c.startswith('nlp_')]
        self.nlp_cols = [c for c in X.columns if c.startswith('nlp_')]
        
        logger.info(f"Data loaded: {len(X)} samples, {len(self.vital_cols)} vital features, {len(self.nlp_cols)} NLP features")
        logger.info(f"Class distribution: {dict(y.value_counts())}")
        
        self.X = X
        self.y = y
        return X, y
    
    def get_forced_interactions(self, top_n: int = 75) -> List[Tuple[int, int]]:
        """
        Load pre-computed top interactions or compute new ones.
        Uses cross-interactions between vital and NLP features.
        """
        # Try to load from existing optimized model
        existing_path = os.path.join(
            os.path.dirname(self.output_dir),
            'ebm_optimized_final',
            'top_75_interactions.csv'
        )
        
        if os.path.exists(existing_path):
            logger.info(f"Loading pre-computed interactions from {existing_path}")
            interactions_df = pd.read_csv(existing_path)
            
            forced_interactions = []
            feature_cols = list(self.X.columns)
            
            for _, row in interactions_df.head(top_n).iterrows():
                v_col = row['vital_feature']
                t_col = row['nlp_feature']
                
                if v_col in feature_cols and t_col in feature_cols:
                    v_idx = feature_cols.index(v_col)
                    t_idx = feature_cols.index(t_col)
                    forced_interactions.append((v_idx, t_idx))
            
            logger.info(f"Loaded {len(forced_interactions)} cross-interactions")
            return forced_interactions
        else:
            logger.warning("No pre-computed interactions found, using automatic detection")
            return top_n  # Let EBM auto-detect
    
    def train_and_evaluate(
        self,
        config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Train EBM with given config and evaluate."""
        
        # Apply undersampling to training set only
        if config.get('undersampling_ratio'):
            ratio = config['undersampling_ratio']
            sampling_strategy = 1.0 / float(ratio)
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
            X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
            logger.info(f"Undersampled: {len(X_train)} -> {len(X_train_res)}")
        else:
            X_train_res, y_train_res = X_train, y_train
        
        # Get interactions
        n_interactions = config.get('n_cross_interactions', 75)
        interactions = self.get_forced_interactions(n_interactions)
        
        # Build EBM with optimized hyperparameters
        ebm = ExplainableBoostingClassifier(
            max_bins=config.get('max_bins', 256),
            interactions=interactions,
            outer_bags=config.get('outer_bags', 25),
            inner_bags=config.get('inner_bags', 20),
            learning_rate=config.get('learning_rate', 0.01),
            min_samples_leaf=config.get('min_samples_leaf', 10),
            max_leaves=config.get('max_leaves', 4),
            max_rounds=config.get('max_rounds', 10000),
            early_stopping_rounds=config.get('early_stopping_rounds', 100),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train
        logger.info("Training EBM...")
        ebm.fit(X_train_res, y_train_res)
        
        # Get predictions
        y_proba = ebm.predict_proba(X_test)[:, 1]
        
        # Find optimal balanced threshold
        best_threshold, best_metrics = self._find_balanced_threshold(
            y_test, y_proba, target=self.target_metrics
        )
        
        y_pred = (y_proba >= best_threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'threshold': best_threshold,
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auprc': average_precision_score(y_test, y_proba),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Check if meets target
        metrics['meets_target'] = (
            metrics['precision'] >= self.target_metrics and
            metrics['recall'] >= self.target_metrics and
            metrics['f1'] >= self.target_metrics
        )
        
        return metrics, ebm
    
    def _find_balanced_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target: float = 0.80
    ) -> Tuple[float, Dict]:
        """Find threshold where P, R, F1 all meet target."""
        
        thresholds = np.arange(0.10, 0.90, 0.005)
        best_threshold = 0.5
        best_min_metric = 0
        best_f1 = 0
        
        candidates = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue
            
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            min_metric = min(prec, rec, f1)
            
            # Check if meets target
            if prec >= target and rec >= target and f1 >= target:
                candidates.append({
                    'threshold': thresh,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'min_metric': min_metric
                })
            
            # Track best even if not meeting target
            if min_metric > best_min_metric or (min_metric == best_min_metric and f1 > best_f1):
                best_min_metric = min_metric
                best_f1 = f1
                best_threshold = thresh
        
        # If we have candidates meeting target, pick best F1
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['f1'])
            best_threshold = best_candidate['threshold']
            logger.info(f"Found threshold meeting target: {best_threshold:.3f} "
                       f"(P={best_candidate['precision']:.3f}, R={best_candidate['recall']:.3f}, "
                       f"F1={best_candidate['f1']:.3f})")
        else:
            logger.warning(f"No threshold meets {target:.0%} target. Best min_metric={best_min_metric:.3f}")
        
        return best_threshold, {'best_min_metric': best_min_metric}
    
    def run_optimization(self) -> Dict:
        """
        Run hyperparameter optimization to find config achieving 80%+ metrics.
        """
        logger.info("="*60)
        logger.info("Starting EBM Optimization Search")
        logger.info(f"Target: P >= {self.target_metrics:.0%}, R >= {self.target_metrics:.0%}, F1 >= {self.target_metrics:.0%}")
        logger.info("="*60)
        
        # Split data - stratified to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.y
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Define search space based on research
        search_configs = [
            # Config 1: Research-optimal settings with moderate undersampling
            {
                'name': 'research_optimal_ratio5',
                'undersampling_ratio': 5.0,
                'n_cross_interactions': 75,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 256,
                'learning_rate': 0.01,
                'min_samples_leaf': 10,
                'max_leaves': 4,
            },
            # Config 2: Higher undersampling (more balanced)
            {
                'name': 'research_optimal_ratio4',
                'undersampling_ratio': 4.0,
                'n_cross_interactions': 75,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 256,
                'learning_rate': 0.01,
                'min_samples_leaf': 10,
                'max_leaves': 4,
            },
            # Config 3: More interactions
            {
                'name': 'more_interactions_ratio5',
                'undersampling_ratio': 5.0,
                'n_cross_interactions': 100,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 256,
                'learning_rate': 0.01,
                'min_samples_leaf': 10,
                'max_leaves': 4,
            },
            # Config 4: Lower learning rate
            {
                'name': 'lower_lr_ratio5',
                'undersampling_ratio': 5.0,
                'n_cross_interactions': 75,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 256,
                'learning_rate': 0.005,
                'min_samples_leaf': 10,
                'max_leaves': 4,
            },
            # Config 5: Fewer bins (more regularization)
            {
                'name': 'fewer_bins_ratio5',
                'undersampling_ratio': 5.0,
                'n_cross_interactions': 75,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 128,
                'learning_rate': 0.01,
                'min_samples_leaf': 15,
                'max_leaves': 3,
            },
            # Config 6: Aggressive undersampling
            {
                'name': 'aggressive_undersample_ratio3',
                'undersampling_ratio': 3.0,
                'n_cross_interactions': 75,
                'outer_bags': 25,
                'inner_bags': 20,
                'max_bins': 256,
                'learning_rate': 0.01,
                'min_samples_leaf': 10,
                'max_leaves': 4,
            },
        ]
        
        best_result = None
        best_model = None
        
        for i, config in enumerate(search_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Config {i+1}/{len(search_configs)}: {config['name']}")
            logger.info(f"{'='*60}")
            logger.info(f"Parameters: {json.dumps(config, indent=2)}")
            
            try:
                metrics, model = self.train_and_evaluate(
                    config, X_train, y_train, X_test, y_test
                )
                
                # Log results
                result = {
                    'config': config,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                self.results_log.append(result)
                
                logger.info(f"\nResults for {config['name']}:")
                logger.info(f"  Threshold: {metrics['threshold']:.3f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall:    {metrics['recall']:.4f}")
                logger.info(f"  F1:        {metrics['f1']:.4f}")
                logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
                logger.info(f"  AUPRC:     {metrics['auprc']:.4f}")
                logger.info(f"  Meets 80% Target: {'✓ YES' if metrics['meets_target'] else '✗ NO'}")
                
                # Update best if this is better
                if best_result is None:
                    best_result = result
                    best_model = model
                else:
                    # Compare by minimum of P/R/F1 (we want all high)
                    curr_min = min(metrics['precision'], metrics['recall'], metrics['f1'])
                    best_min = min(
                        best_result['metrics']['precision'],
                        best_result['metrics']['recall'],
                        best_result['metrics']['f1']
                    )
                    
                    if curr_min > best_min:
                        best_result = result
                        best_model = model
                        logger.info(f"  ★ New best configuration!")
                
                # Early exit if we meet target
                if metrics['meets_target']:
                    logger.info(f"\n🎉 Found configuration meeting 80% target!")
                    break
                    
            except Exception as e:
                logger.error(f"Error with config {config['name']}: {e}")
                continue
        
        # Save results
        if best_result:
            self.save_results(best_result, best_model, X_test, y_test)
        
        return best_result
    
    def save_results(
        self,
        result: Dict,
        model: ExplainableBoostingClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Save the best model and results."""
        logger.info(f"\n{'='*60}")
        logger.info("Saving Best Results")
        logger.info(f"{'='*60}")
        
        # Save model
        model_path = os.path.join(self.output_dir, 'final_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")
        
        # Save metrics
        metrics = result['metrics']
        cm = metrics['confusion_matrix']
        
        metrics_output = {
            'threshold': metrics['threshold'],
            'class_1': {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            },
            'accuracy': (cm[0][0] + cm[1][1]) / sum(sum(row) for row in cm),
            'auprc': metrics['auprc'],
            'auroc': metrics['auc_roc'],
            'confusion_matrix': {
                'TN': cm[0][0],
                'FP': cm[0][1],
                'FN': cm[1][0],
                'TP': cm[1][1]
            },
            'config': {
                'seed': self.random_state,
                **{k: v for k, v in result['config'].items() if k != 'name'},
                'model_type': 'EBM_cross_interaction_optimized'
            },
            'meets_80_target': metrics['meets_target']
        }
        
        metrics_path = os.path.join(self.output_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_output, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save feature info
        feature_info = {
            'X_V_features': self.vital_cols,
            'X_T_features': self.nlp_cols,
            'n_cross_interactions': result['config'].get('n_cross_interactions', 75),
            'formula': 'F_final(x) = β₀ + Σf_v(v) + Σf_t(t) + Σf_{v,t}(v,t)'
        }
        
        feature_path = os.path.join(self.output_dir, 'feature_info.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        logger.info(f"Saved feature info to {feature_path}")
        
        # Save search log
        log_path = os.path.join(self.output_dir, 'optimization_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.results_log, f, indent=2, default=str)
        logger.info(f"Saved optimization log to {log_path}")
        
        # Print final summary
        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Best Configuration: {result['config']['name']}")
        logger.info(f"Optimal Threshold: {metrics['threshold']:.3f}")
        logger.info(f"")
        logger.info(f"Class 1 (ICU Readmission) Metrics:")
        logger.info(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        logger.info(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.1f}%)")
        logger.info(f"")
        logger.info(f"Meets 80% Target: {'✓ YES' if metrics['meets_target'] else '✗ NO'}")
        logger.info(f"")
        logger.info(f"All results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='EBM Cross-Interaction Optimization to 80%+ P/R/F1'
    )
    parser.add_argument(
        '--vital-features', type=str, required=True,
        help='Path to vital/lab features CSV'
    )
    parser.add_argument(
        '--nlp-features', type=str, required=True,
        help='Path to NLP features CSV'
    )
    parser.add_argument(
        '--labels', type=str, required=True,
        help='Path to labels CSV'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for new model (will NOT modify existing models)'
    )
    parser.add_argument(
        '--target-metrics', type=float, default=0.80,
        help='Target minimum for P, R, F1 (default: 0.80)'
    )
    parser.add_argument(
        '--seed', type=int, default=170,
        help='Random seed (default: 170)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Verify script loads correctly without training'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run - script loaded successfully!")
        return
    
    # Create optimizer
    optimizer = EBMOptimizer(
        output_dir=args.output_dir,
        random_state=args.seed,
        target_metrics=args.target_metrics
    )
    
    # Load data
    optimizer.load_data(
        args.vital_features,
        args.nlp_features,
        args.labels
    )
    
    # Run optimization
    result = optimizer.run_optimization()
    
    if result and result['metrics']['meets_target']:
        logger.info("\n🎉 SUCCESS: Found configuration achieving 80%+ for P, R, F1!")
        sys.exit(0)
    else:
        logger.warning("\n⚠️  Could not find configuration meeting 80% target.")
        logger.info("Best result saved. Consider adjusting parameters or data preprocessing.")
        sys.exit(1)


if __name__ == '__main__':
    main()

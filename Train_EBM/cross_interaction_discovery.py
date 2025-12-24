#!/usr/bin/env python3
"""
Cross-Interaction Discovery using FAST Algorithm for EBM.

This implements the core strategy for "Contextualization" - discovering meaningful
interactions between vital/lab features (X_V) and NLP text features (X_T).

The FAST algorithm:
1. Train Main Effects EBM with single variables from X_V and X_T
2. Compute residuals r_i = y_i - F_main(x_i)
3. Rank cross-interactions (v,t) ∈ I_cross based on residual patterns
4. Select Top-N interactions and retrain final EBM with forced interactions

Usage:
    python cross_interaction_discovery.py \
        --vital-features cohort/features_locf_v2.csv \
        --nlp-features outputs/nlp_features_boc.csv \
        --labels cohort/new_cohort_icu_readmission_labels.csv \
        --output outputs/cross_interactions \
        --top-n 50

Author: Clinical NLP Pipeline
Date: 2025-12-03
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
import itertools
import logging
import os
import json
import warnings
from typing import List, Tuple, Dict, Set
import pickle

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossInteractionDiscovery:
    """
    Discover cross-interactions between vital/lab features and NLP features
    using the FAST algorithm with domain constraints.
    
    Supports dynamic class weighting to handle imbalanced datasets.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        min_samples_per_bin: int = 50,
        random_state: int = 42,
        use_dynamic_weighting: bool = True,
        class_weight_strategy: str = 'balanced',
        use_cost_sensitive: bool = True,
        fn_cost_multiplier: str = 'auto',
        threshold_strategy: str = 'auto',
        target_recall: float = None,
        clinical_severity: float = 1.0,
        min_metrics_threshold: float = 0.6,
        use_calibration: bool = False,
        undersampling_ratio: float = None,
        # New hyperparameters
        learning_rate: float = 0.01,
        max_leaves: int = 4,
        min_samples_leaf: int = 10,
        max_bins: int = 128,
        outer_bags: int = 16,
        inner_bags: int = 8,
        max_rounds: int = 10000,
        early_stopping_rounds: int = 100,
        interactions: int = 0
    ):
        """
        Initialize the discovery pipeline with FULLY DYNAMIC support.
        
        Args:
            n_bins: Number of bins for continuous features
            min_samples_per_bin: Minimum samples required per bin
            random_state: Random seed for reproducibility
            use_dynamic_weighting: Whether to use dynamic class weighting for imbalance
            class_weight_strategy: Strategy for computing class weights
                - 'balanced': n_samples / (n_classes * np.bincount(y))
                - 'sqrt_balanced': sqrt of balanced weights (less aggressive)
                - 'effective_num': effective number of samples (best for high imbalance)
            use_cost_sensitive: Whether to use cost-sensitive learning (penalize FN more)
            fn_cost_multiplier: Multiplier for False Negative cost
                - 'auto': Automatically computed from imbalance ratio (RECOMMENDED)
                - 'clinical': Based on clinical severity (uses clinical_severity param)
                - float: Fixed multiplier (e.g., 3.0)
            threshold_strategy: Strategy for optimal threshold selection
                - 'auto': Optimize based on cost matrix (RECOMMENDED)
                - 'f1': Optimize for best F1 score
                - 'recall': Optimize for target recall (uses target_recall param)
                - 'cost': Minimize total cost
                - 'balanced': Find threshold where P, R, F1 all >= min_metrics_threshold
            target_recall: Target recall for threshold tuning (only used if threshold_strategy='recall')
            clinical_severity: Clinical severity multiplier for cost computation (1.0 = normal, 2.0 = high risk)
            min_metrics_threshold: Minimum value for P, R, F1 when using 'balanced' strategy (default: 0.6)
        """
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.random_state = random_state
        self.use_dynamic_weighting = use_dynamic_weighting
        self.class_weight_strategy = class_weight_strategy
        self.use_cost_sensitive = use_cost_sensitive
        self.fn_cost_multiplier_config = fn_cost_multiplier  # Store config
        self.fn_cost_multiplier = None  # Will be computed dynamically
        self.threshold_strategy = threshold_strategy
        self.target_recall = target_recall
        self.clinical_severity = clinical_severity
        self.min_metrics_threshold = min_metrics_threshold
        
        self.main_effects_model = None
        self.final_model = None
        self.residuals = None
        self.interaction_scores = None
        self.X_V_cols = None  # Vital/lab feature columns
        self.X_T_cols = None  # Text/NLP feature columns
        self.class_weights = None  # Computed class weights
        self.sample_weights = None  # Per-sample weights
        self.cost_matrix = None  # Cost matrix for cost-sensitive learning
        self.imbalance_ratio = None  # Will be computed from data
        self.dynamic_config = {}  # Store all dynamic configurations
        # Optional undersampling during final model training (e.g., 3.0 -> 3:1 majority:minority)
        self.undersampling_ratio = undersampling_ratio
        self.use_calibration = use_calibration
        # Final model hyperparameters (configurable)
        self.final_learning_rate = learning_rate
        self.final_max_leaves = max_leaves
        self.final_min_samples_leaf = min_samples_leaf
        self.final_max_bins = max_bins
        self.final_outer_bags = outer_bags
        self.final_inner_bags = inner_bags
        self.final_max_rounds = max_rounds
        self.final_early_stopping_rounds = early_stopping_rounds
        self.final_interactions = interactions
        
    def _identify_feature_sets(
        self,
        vital_df: pd.DataFrame,
        nlp_df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Identify vital/lab features (X_V) and text features (X_T).
        
        X_V: Continuous features from vital signs, labs
        X_T: Binary features from NLP (Bag-of-Concepts)
        """
        # Vital/lab features - exclude ID columns and categorical
        exclude_cols = {'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'hadm_id', 'subject_id', 
                       'icustay_id', 'Y', 'label', 'Gender', 'Discharge_Disposition'}
        
        # Also exclude time-related columns
        time_cols = {col for col in vital_df.columns if 'ChartTime' in col or 'Time' in col}
        exclude_cols.update(time_cols)
        
        X_V_cols = [col for col in vital_df.columns 
                    if col not in exclude_cols and vital_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        # NLP features - binary columns (0/1)
        nlp_exclude = {'hadm_id', 'HADM_ID', 'label', 'Y'}
        X_T_cols = [col for col in nlp_df.columns 
                    if col not in nlp_exclude and nlp_df[col].nunique() == 2]
        
        logger.info(f"Identified {len(X_V_cols)} vital/lab features (X_V)")
        logger.info(f"Identified {len(X_T_cols)} NLP features (X_T)")
        
        return X_V_cols, X_T_cols
    
    def _prepare_data(
        self,
        vital_df: pd.DataFrame,
        nlp_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Merge and prepare data for training.
        """
        logger.info("Preparing merged dataset...")
        
        # Normalize column names
        for df in [vital_df, nlp_df, labels_df]:
            if 'HADM_ID' in df.columns:
                df.rename(columns={'HADM_ID': 'hadm_id'}, inplace=True)
        
        # Determine merge key
        merge_key = 'hadm_id' if 'hadm_id' in vital_df.columns else 'HADM_ID'
        
        # Drop label column from NLP if exists (we'll use labels_df)
        if 'label' in nlp_df.columns:
            nlp_df = nlp_df.drop(columns=['label'])
        
        # Check if NLP has the merge key 
        nlp_merge_key = merge_key if merge_key in nlp_df.columns else ('HADM_ID' if 'HADM_ID' in nlp_df.columns else 'hadm_id')
        
        # Merge vital and NLP features
        merged = vital_df.merge(nlp_df, left_on=merge_key, right_on=nlp_merge_key, how='inner')
        logger.info(f"After merging vital and NLP: {len(merged)} patients")
        
        # Merge with labels - handle different column names
        label_col = 'label' if 'label' in labels_df.columns else 'Y'
        hadm_col = 'hadm_id' if 'hadm_id' in labels_df.columns else 'HADM_ID'
        
        labels_subset = labels_df[[hadm_col, label_col]].copy()
        labels_subset.columns = ['hadm_id', 'label']
        
        merged = merged.merge(labels_subset, on='hadm_id', how='inner')
        logger.info(f"After merging with labels: {len(merged)} patients")
        
        # Extract features and labels
        y = merged['label']
        
        # Get all feature columns (X_V + X_T)
        feature_cols = self.X_V_cols + self.X_T_cols
        X = merged[feature_cols].copy()
        
        # Handle missing values in X_V (vital features)
        for col in self.X_V_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        logger.info(f"Final dataset: {X.shape[0]} patients, {X.shape[1]} features")
        
        return X, y
    
    def _compute_class_weights(self, y: pd.Series) -> Tuple[Dict, np.ndarray]:
        """
        Compute dynamic class weights to handle imbalanced data.
        
        Returns:
            class_weights: Dict mapping class label to weight
            sample_weights: Array of per-sample weights
        """
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        logger.info(f"\nClass Distribution:")
        for c, count in zip(classes, counts):
            logger.info(f"  Class {c}: {count} samples ({count/n_samples*100:.2f}%)")
        
        # Compute imbalance ratio
        imbalance_ratio = counts.max() / counts.min()
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if self.class_weight_strategy == 'balanced':
            # Standard balanced weighting: n_samples / (n_classes * count)
            weights = n_samples / (n_classes * counts)
        elif self.class_weight_strategy == 'sqrt_balanced':
            # Less aggressive: sqrt of balanced weights
            balanced_weights = n_samples / (n_classes * counts)
            weights = np.sqrt(balanced_weights)
        elif self.class_weight_strategy == 'inverse_freq':
            # Inverse frequency
            weights = n_samples / counts
            weights = weights / weights.sum() * n_classes  # Normalize
        elif self.class_weight_strategy == 'effective_num':
            # Effective number of samples (from "Class-Balanced Loss" paper)
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / effective_num
            weights = weights / weights.sum() * n_classes  # Normalize
        else:
            # Default to balanced
            weights = n_samples / (n_classes * counts)
        
        # Create class weight dict
        class_weights = {c: w for c, w in zip(classes, weights)}
        
        logger.info(f"\nDynamic Class Weights ({self.class_weight_strategy}):")
        for c, w in class_weights.items():
            logger.info(f"  Class {c}: weight = {w:.4f}")
        
        # Create per-sample weights
        sample_weights = np.array([class_weights[label] for label in y])
        
        return class_weights, sample_weights
    
    def _compute_dynamic_fn_cost(self, y: np.ndarray) -> float:
        """
        FULLY DYNAMIC: Compute FN cost multiplier based on data characteristics.
        
        Strategies:
        - 'auto': Based on imbalance ratio (recommended)
        - 'clinical': Based on clinical severity parameter
        - float: Use fixed value
        
        Returns:
            fn_cost: Dynamic FN cost multiplier
        """
        # Compute imbalance ratio from data
        classes, counts = np.unique(y, return_counts=True)
        self.imbalance_ratio = counts.max() / counts.min()
        
        logger.info(f"\n{'='*60}")
        logger.info("DYNAMIC FN COST COMPUTATION")
        logger.info(f"{'='*60}")
        logger.info(f"  Imbalance Ratio: {self.imbalance_ratio:.2f}:1")
        
        if self.fn_cost_multiplier_config == 'auto':
            # Auto mode: Scale with imbalance ratio
            # Formula: fn_cost = log(imbalance_ratio) * clinical_severity
            # Using log instead of sqrt for better precision-recall balance
            fn_cost = np.log(self.imbalance_ratio + 1) * self.clinical_severity
            fn_cost = np.clip(fn_cost, 1.5, 10.0)  # Reasonable bounds
            logger.info(f"  Mode: AUTO (balanced)")
            logger.info(f"  Formula: log(imbalance+1) * clinical_severity")
            logger.info(f"  = log({self.imbalance_ratio:.2f}+1) * {self.clinical_severity} = {fn_cost:.2f}")
            
        elif self.fn_cost_multiplier_config == 'clinical':
            # Clinical mode: Based on severity and mortality risk
            # ICU readmission has ~20-30% mortality increase
            base_cost = 5.0  # Base clinical cost
            fn_cost = base_cost * self.clinical_severity
            logger.info(f"  Mode: CLINICAL")
            logger.info(f"  Base cost: {base_cost}, Severity: {self.clinical_severity}")
            logger.info(f"  FN Cost: {fn_cost:.2f}")
            
        elif isinstance(self.fn_cost_multiplier_config, (int, float)):
            # Fixed mode: Use provided value
            fn_cost = float(self.fn_cost_multiplier_config)
            logger.info(f"  Mode: FIXED")
            logger.info(f"  FN Cost: {fn_cost:.2f} (user-specified)")
            
        else:
            # Default to auto
            fn_cost = np.sqrt(self.imbalance_ratio)
            fn_cost = np.clip(fn_cost, 2.0, 10.0)
            logger.info(f"  Mode: DEFAULT (auto)")
            logger.info(f"  FN Cost: {fn_cost:.2f}")
        
        self.fn_cost_multiplier = fn_cost
        
        # Store in dynamic config
        self.dynamic_config['fn_cost'] = {
            'mode': str(self.fn_cost_multiplier_config),
            'imbalance_ratio': float(self.imbalance_ratio),
            'clinical_severity': float(self.clinical_severity),
            'computed_cost': float(fn_cost)
        }
        
        return fn_cost
    
    def step1_train_main_effects(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        skip_training: bool = False
    ) -> np.ndarray:
        """
        Step 1: Train Main Effects EBM (no interactions).
        
        F_main(x) = β₀ + Σ f_v(v) + Σ f_t(t)
        
        Returns residuals r_i = y_i - F_main(x_i)
        """
        logger.info("="*60)
        logger.info("STEP 1: Training Main Effects EBM")
        logger.info("="*60)
        
        # DYNAMIC: Compute FN cost from data
        if self.use_cost_sensitive:
            self._compute_dynamic_fn_cost(y.values)
        
        # Compute dynamic class weights for imbalance handling
        if self.use_dynamic_weighting:
            self.class_weights, self.sample_weights = self._compute_class_weights(y)
        else:
            self.class_weights = None
            self.sample_weights = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Split sample weights accordingly
        if self.sample_weights is not None:
            train_indices = X_train.index
            test_indices = X_test.index
            sample_weights_train = np.array([
                self.class_weights[y.loc[idx]] for idx in train_indices
            ])
            sample_weights_test = np.array([
                self.class_weights[y.loc[idx]] for idx in test_indices
            ])
        else:
            sample_weights_train = None
            sample_weights_test = None
        
        # Apply cost-sensitive weighting if enabled
        if self.use_cost_sensitive:
            logger.info("\n" + "="*60)
            logger.info("APPLYING COST-SENSITIVE LEARNING")
            logger.info("="*60)
            sample_weights_train = self._compute_cost_sensitive_weights(
                y_train.values,
                sample_weights_train
            )
            sample_weights_test = self._compute_cost_sensitive_weights(
                y_test.values,
                sample_weights_test
            )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train EBM with NO interactions (main effects only)
        # STRONG REGULARIZATION: Prevent overfitting, improve generalization
        if skip_training:
            logger.info("Skipping training of main effects model (using cached interactions)...")
            # Store train/test split for later
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X_full = X
            self.y_full = y
            self.sample_weights_train = sample_weights_train
            self.sample_weights_test = sample_weights_test
            return None

        self.main_effects_model = ExplainableBoostingClassifier(
            max_bins=128,             # Reduced from 512 - less overfitting
            interactions=0,           # No interactions - main effects only
            outer_bags=16,            # Reduced bagging for faster training
            inner_bags=8,             # Reduced inner bagging
            learning_rate=0.01,       # Moderate learning rate
            min_samples_leaf=10,      # Strong regularization
            max_leaves=4,             # Reduced from 8 - simpler model
            max_rounds=10000,         # Reduced rounds
            early_stopping_rounds=100, # Earlier stopping
            random_state=self.random_state,
            n_jobs=-1
        )
        
        logger.info("Training main effects model (no interactions)...")
        if self.use_dynamic_weighting:
            logger.info("Using dynamic class weighting for imbalance handling")
            self.main_effects_model.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            self.main_effects_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.main_effects_model.score(X_train, y_train)
        test_score = self.main_effects_model.score(X_test, y_test)
        logger.info(f"Main effects model - Train accuracy: {train_score:.4f}")
        logger.info(f"Main effects model - Test accuracy: {test_score:.4f}")
        
        # Evaluate with class-weighted metrics (important for imbalanced data)
        if self.use_dynamic_weighting:
            from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
            
            y_pred_train = self.main_effects_model.predict(X_train)
            y_pred_test = self.main_effects_model.predict(X_test)
            y_proba_train = self.main_effects_model.predict_proba(X_train)[:, 1]
            y_proba_test = self.main_effects_model.predict_proba(X_test)[:, 1]
            
            train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
            test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            train_auc = roc_auc_score(y_train, y_proba_train)
            test_auc = roc_auc_score(y_test, y_proba_test)
            train_auprc = average_precision_score(y_train, y_proba_train)
            test_auprc = average_precision_score(y_test, y_proba_test)
            
            logger.info(f"\nImbalance-aware metrics:")
            logger.info(f"  Train Balanced Accuracy: {train_balanced_acc:.4f}")
            logger.info(f"  Test Balanced Accuracy: {test_balanced_acc:.4f}")
            logger.info(f"  Train F1 (weighted): {train_f1:.4f}")
            logger.info(f"  Test F1 (weighted): {test_f1:.4f}")
            logger.info(f"  Train AUC-ROC: {train_auc:.4f}")
            logger.info(f"  Test AUC-ROC: {test_auc:.4f}")
            logger.info(f"  Train AUPRC (PR-AUC): {train_auprc:.4f}")
            logger.info(f"  Test AUPRC (PR-AUC): {test_auprc:.4f}")
        
        # ============================================================
        # QUAN TRỌNG: Tính residuals CHỈ trên TRAINING data
        # Để tránh data leakage khi ranking interactions
        # ============================================================
        # For classification, use predicted probabilities
        y_pred_proba_train = self.main_effects_model.predict_proba(X_train)[:, 1]
        self.residuals_train = y_train.values - y_pred_proba_train
        
        # Cũng tính residuals trên full data để có thể rank với nhiều samples hơn
        # Nhưng việc này chỉ dùng cho analysis, không ảnh hưởng model training
        y_pred_proba_full = self.main_effects_model.predict_proba(X)[:, 1]
        self.residuals = y.values - y_pred_proba_full
        
        logger.info(f"Residuals (train only): mean={self.residuals_train.mean():.4f}, std={self.residuals_train.std():.4f}")
        logger.info(f"Residuals (full): mean={self.residuals.mean():.4f}, std={self.residuals.std():.4f}")
        
        # Store train/test split for later
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_full = X
        self.y_full = y
        self.sample_weights_train = sample_weights_train
        self.sample_weights_test = sample_weights_test
        
        return self.residuals
    
    def step2_rank_interactions(
        self,
        X: pd.DataFrame,
        residuals: np.ndarray
    ) -> pd.DataFrame:
        """
        Step 2: Rank cross-interactions based on residual patterns.
        
        For each pair (v, t) ∈ I_cross:
        - Bin the continuous variable v
        - Split by binary variable t (0/1)
        - Compute interaction score based on residual differences
        
        The score measures whether the risk curve of v changes shape
        significantly when t=0 vs t=1.
        """
        logger.info("="*60)
        logger.info("STEP 2: Ranking Cross-Interactions")
        logger.info("="*60)
        
        interaction_results = []
        total_pairs = len(self.X_V_cols) * len(self.X_T_cols)
        
        logger.info(f"Total cross-interaction pairs to evaluate: {total_pairs:,}")
        logger.info(f"X_V features: {len(self.X_V_cols)}")
        logger.info(f"X_T features: {len(self.X_T_cols)}")
        
        processed = 0
        for v_col in self.X_V_cols:
            for t_col in self.X_T_cols:
                processed += 1
                if processed % 10000 == 0:
                    logger.info(f"Progress: {processed:,}/{total_pairs:,} pairs evaluated")
                
                try:
                    score, details = self._compute_interaction_score(
                        X[v_col].values,
                        X[t_col].values,
                        residuals
                    )
                    
                    if score is not None:
                        interaction_results.append({
                            'vital_feature': v_col,
                            'nlp_feature': t_col,
                            'interaction_score': score,
                            **details
                        })
                except Exception as e:
                    # Skip problematic pairs
                    continue
        
        # Create DataFrame and sort by score
        self.interaction_scores = pd.DataFrame(interaction_results)
        self.interaction_scores = self.interaction_scores.sort_values(
            'interaction_score', ascending=False
        ).reset_index(drop=True)
        
        logger.info(f"Valid interaction pairs: {len(self.interaction_scores):,}")
        logger.info("\nTop 20 cross-interactions:")
        logger.info("-"*80)
        for i, row in self.interaction_scores.head(20).iterrows():
            logger.info(f"{i+1:2d}. {row['vital_feature']:<30} × {row['nlp_feature']:<35} | score={row['interaction_score']:.4f}")
        
        return self.interaction_scores
    
    def _compute_interaction_score(
        self,
        v_values: np.ndarray,
        t_values: np.ndarray,
        residuals: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Compute interaction score for a (v, t) pair.
        
        Uses multiple methods:
        1. Residual variance reduction when stratifying by t
        2. F-statistic from 2-way ANOVA-like analysis
        3. Correlation difference between groups
        """
        # Ensure t is binary
        t_unique = np.unique(t_values[~np.isnan(t_values)])
        if len(t_unique) != 2:
            return None, {}
        
        # Create mask for valid values
        valid_mask = ~np.isnan(v_values) & ~np.isnan(t_values)
        v = v_values[valid_mask]
        t = t_values[valid_mask]
        r = residuals[valid_mask]
        
        if len(v) < self.min_samples_per_bin * 4:
            return None, {}
        
        # Split by t
        t0_mask = t == 0
        t1_mask = t == 1
        
        n_t0 = t0_mask.sum()
        n_t1 = t1_mask.sum()
        
        if n_t0 < self.min_samples_per_bin * 2 or n_t1 < self.min_samples_per_bin * 2:
            return None, {}
        
        # Bin the continuous variable
        try:
            v_bins = pd.qcut(v, q=self.n_bins, labels=False, duplicates='drop')
        except:
            return None, {}
        
        # Method 1: Residual pattern difference
        # Check if residual patterns differ between t=0 and t=1 groups
        residual_diff_score = self._compute_residual_pattern_diff(
            v_bins, t, r, t0_mask, t1_mask
        )
        
        # Method 2: Two-way interaction F-statistic
        f_score = self._compute_interaction_fstat(v_bins, t, r)
        
        # Method 3: Correlation difference
        corr_t0 = np.corrcoef(v[t0_mask], r[t0_mask])[0, 1] if n_t0 > 10 else 0
        corr_t1 = np.corrcoef(v[t1_mask], r[t1_mask])[0, 1] if n_t1 > 10 else 0
        corr_diff = abs(corr_t0 - corr_t1)
        
        # Combined score (weighted average)
        combined_score = 0.4 * residual_diff_score + 0.4 * f_score + 0.2 * corr_diff * 100
        
        details = {
            'residual_diff_score': residual_diff_score,
            'f_score': f_score,
            'corr_t0': corr_t0,
            'corr_t1': corr_t1,
            'corr_diff': corr_diff,
            'n_t0': n_t0,
            'n_t1': n_t1,
            'n_total': len(v)
        }
        
        return combined_score, details
    
    def _compute_residual_pattern_diff(
        self,
        v_bins: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        t0_mask: np.ndarray,
        t1_mask: np.ndarray
    ) -> float:
        """
        Compute how much the residual pattern differs between t=0 and t=1.
        
        For each bin of v, compute mean residual for t=0 and t=1 groups.
        Score is the sum of squared differences.
        """
        unique_bins = np.unique(v_bins[~np.isnan(v_bins)])
        
        diff_sum = 0
        valid_bins = 0
        
        for b in unique_bins:
            bin_mask = v_bins == b
            
            r_t0 = r[bin_mask & t0_mask]
            r_t1 = r[bin_mask & t1_mask]
            
            if len(r_t0) >= 5 and len(r_t1) >= 5:
                mean_diff = abs(r_t0.mean() - r_t1.mean())
                diff_sum += mean_diff ** 2
                valid_bins += 1
        
        if valid_bins == 0:
            return 0
        
        return np.sqrt(diff_sum / valid_bins) * 100
    
    def _compute_interaction_fstat(
        self,
        v_bins: np.ndarray,
        t: np.ndarray,
        r: np.ndarray
    ) -> float:
        """
        Compute F-statistic for interaction effect using 2-way ANOVA approach.
        """
        try:
            # Create group labels
            groups = []
            residuals_by_group = []
            
            unique_bins = np.unique(v_bins[~np.isnan(v_bins)])
            
            for b in unique_bins:
                for t_val in [0, 1]:
                    mask = (v_bins == b) & (t == t_val)
                    if mask.sum() >= 3:
                        groups.append(f"b{b}_t{t_val}")
                        residuals_by_group.append(r[mask])
            
            if len(residuals_by_group) < 4:
                return 0
            
            # One-way ANOVA on combined groups
            f_stat, p_value = stats.f_oneway(*residuals_by_group)
            
            if np.isnan(f_stat):
                return 0
            
            return min(f_stat, 100)  # Cap at 100 to avoid outliers
            
        except:
            return 0
    
    def step3_train_final_model(
        self,
        top_n: int = 50
    ) -> ExplainableBoostingClassifier:
        """
        Step 3: Train final EBM with forced cross-interactions.
        
        ĐÚNG CÔNG THỨC (theo paper):
        
        F_final(x) = β₀ + Σ f_v(v) + Σ f_t(t) + Σ f_{v,t}(v, t)
                          v∈X_V      t∈X_T      (v,t)∈Top-N
        
        QUAN TRỌNG: 
        - Residuals từ Step 1 chỉ dùng để RANKING interactions (Step 2)
        - Final model được TRAIN LẠI TỪ ĐẦU với all main effects + forced interactions
        - KHÔNG PHẢI cộng thêm interaction model vào main effects model
        
        Sử dụng dynamic class weighting để xử lý imbalance.
        """
        logger.info("="*60)
        logger.info(f"STEP 3: Training Final EBM with Top-{top_n} Interactions")
        logger.info("="*60)
        
        # Get top-N interaction pairs
        top_interactions = self.interaction_scores.head(top_n)
        
        # Get feature column names from training data
        feature_cols = list(self.X_train.columns)
        
        # Create interaction list for EBM using column INDICES (not names)
        # EBM expects interactions as list of tuples of column indices
        forced_interactions = []
        for _, row in top_interactions.iterrows():
            v_col = row['vital_feature']
            t_col = row['nlp_feature']
            
            # Check if columns exist and get their indices
            if v_col in feature_cols and t_col in feature_cols:
                v_idx = feature_cols.index(v_col)
                t_idx = feature_cols.index(t_col)
                forced_interactions.append((v_idx, t_idx))
            else:
                logger.warning(f"Column not found: {v_col} or {t_col}")
        
        logger.info(f"Selected {len(forced_interactions)} cross-interactions from FAST ranking")
        logger.info("\nĐÚNG CÔNG THỨC:")
        logger.info("  F_final(x) = β₀ + Σf_v(v) + Σf_t(t) + Σf_{v,t}(v,t)")
        logger.info("  → Train lại từ đầu với ALL main effects + forced interactions")
        logger.info("  → Residuals chỉ dùng để ranking, KHÔNG dùng để train final model")
        
        if self.use_dynamic_weighting:
            logger.info("  → Sử dụng dynamic class weighting cho imbalance")
        
        # ============================================================
        # ĐÚNG CÔNG THỨC: Train Final EBM từ đầu
        # F_final = β₀ + Σf_v(v) + Σf_t(t) + Σf_{v,t}(v,t)
        # ============================================================
        
        logger.info("\n" + "="*60)
        logger.info("Training FINAL MODEL (all main effects + interactions)")
        logger.info("="*60)

        # Optionally undersample training set to change class balance for final model
        X_train_final = self.X_train
        y_train_final = self.y_train
        sample_weights_final = getattr(self, 'sample_weights_train', None)

        if getattr(self, 'undersampling_ratio', None) is not None:
            try:
                sampling_strategy = 1.0 / float(self.undersampling_ratio)
                rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
                X_res, y_res = rus.fit_resample(X_train_final, y_train_final)
                if sample_weights_final is not None:
                    # Align sample weights by index after resampling
                    sample_weights_res = []
                    for idx in rus.sample_indices_:
                        sample_weights_res.append(sample_weights_final[idx])
                    sample_weights_res = np.array(sample_weights_res)
                else:
                    sample_weights_res = None

                X_train_final = X_res
                y_train_final = y_res
                sample_weights_final = sample_weights_res
                logger.info(f"Applied undersampling: ratio={self.undersampling_ratio} -> train size {len(X_train_final)}")
            except Exception as e:
                logger.warning(f"Undersampling failed: {e}. Proceeding without undersampling.")

        # STRONG REGULARIZATION: Better generalization to test set
        # If calibration is requested, split the training set for calibration
        calibrated_classifier = None
        if getattr(self, 'use_calibration', False):
            from sklearn.model_selection import train_test_split
            X_train_sub, X_cal, y_train_sub, y_cal = train_test_split(
                X_train_final, y_train_final, test_size=0.2, random_state=self.random_state, stratify=y_train_final
            )
            # Recompute sample weights for the sub-training split using class/cost-sensitive weighting
            if self.use_dynamic_weighting:
                class_weights_sub, _ = self._compute_class_weights(pd.Series(y_train_sub))
                # Use cost-sensitive weights if enabled
                if self.use_cost_sensitive:
                    sample_weights_sub = self._compute_cost_sensitive_weights(y_train_sub, base_weights=np.array([class_weights_sub[int(label)] for label in y_train_sub]))
                else:
                    sample_weights_sub = np.array([class_weights_sub[int(label)] for label in y_train_sub])
            else:
                sample_weights_sub = None
            self.final_model = ExplainableBoostingClassifier(
            max_bins=self.final_max_bins,             
            interactions=forced_interactions,  # Forced cross-interactions from Top-N
            outer_bags=self.final_outer_bags,            
            inner_bags=self.final_inner_bags,               
            learning_rate=self.final_learning_rate,       
            min_samples_leaf=self.final_min_samples_leaf,      
            max_leaves=self.final_max_leaves,             
            max_rounds=self.final_max_rounds,         
            early_stopping_rounds=self.final_early_stopping_rounds, 
            random_state=self.random_state,
            n_jobs=-1
        )
            # Train on sub-training set
            if self.use_dynamic_weighting or self.use_cost_sensitive:
                if sample_weights_sub is not None:
                    logger.info("Training final model (sub-train) with combined weights (class + cost-sensitive)...")
                    self.final_model.fit(X_train_sub, y_train_sub, sample_weight=sample_weights_sub)
                else:
                    logger.info("Training final model (sub-train) without sample weights...")
                    self.final_model.fit(X_train_sub, y_train_sub)
            else:
                self.final_model.fit(X_train_sub, y_train_sub)

            # Calibrate using the calibration subset
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(self.final_model, cv='prefit', method='sigmoid')
            calibrator.fit(X_cal, y_cal)
            calibrated_classifier = calibrator
            self.calibrated_final_classifier = calibrated_classifier
        else:
            self.final_model = ExplainableBoostingClassifier(
                max_bins=self.final_max_bins,             
                interactions=forced_interactions,  # Forced cross-interactions from Top-N
                outer_bags=self.final_outer_bags,            
                inner_bags=self.final_inner_bags,            
                learning_rate=self.final_learning_rate,       
                min_samples_leaf=self.final_min_samples_leaf,      
                max_leaves=self.final_max_leaves,             
                max_rounds=self.final_max_rounds,         
                early_stopping_rounds=self.final_early_stopping_rounds, 
                random_state=self.random_state,
                n_jobs=-1
            )

            # Train với sample weights nếu có (class weights + cost-sensitive)
            if self.use_dynamic_weighting or self.use_cost_sensitive:
                if sample_weights_final is not None:
                    logger.info("Training with combined weights (class + cost-sensitive)...")
                    self.final_model.fit(X_train_final, y_train_final, sample_weight=sample_weights_final)
                else:
                    logger.info("Training without sample weights...")
                    self.final_model.fit(X_train_final, y_train_final)
            else:
                self.final_model.fit(X_train_final, y_train_final)
        
        # ============================================================
        # Evaluate: Main Effects vs Final Model
        # ============================================================
        
        # Main effects model (no interactions)
        if self.main_effects_model is not None:
            main_accuracy = self.main_effects_model.score(self.X_test, self.y_test)
        else:
            main_accuracy = 0
        
        # Final model (main effects + interactions)
        final_accuracy = self.final_model.score(self.X_test, self.y_test)
        
        # ============================================================
        # FULL METRICS for both models (per-class breakdown)
        # ============================================================
        from sklearn.metrics import (
            balanced_accuracy_score, f1_score, roc_auc_score, 
            precision_score, recall_score, classification_report,
            confusion_matrix, average_precision_score
        )
        
        # Main effects predictions
        if self.main_effects_model is not None:
            main_pred_binary = self.main_effects_model.predict(self.X_test)
            main_proba = self.main_effects_model.predict_proba(self.X_test)[:, 1]
        else:
             main_pred_binary = None
             main_proba = None
        
        # Final model predictions (ĐÚNG CÔNG THỨC: trained từ đầu)
        final_pred_binary = self.final_model.predict(self.X_test)
        if hasattr(self, 'calibrated_final_classifier') and self.calibrated_final_classifier is not None:
            final_proba = self.calibrated_final_classifier.predict_proba(self.X_test)[:, 1]
            # For binary predictions, prefer using the calibrated classifier's predict to match proba thresholding
            final_pred_binary = self.calibrated_final_classifier.predict(self.X_test)
        else:
            final_proba = self.final_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate per-class metrics
        def compute_full_metrics(y_true, y_pred, y_proba):
            """Compute full metrics for both classes"""
            return {
                'accuracy': (y_true == y_pred).mean(),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'auc_roc': roc_auc_score(y_true, y_proba),
                'auprc': average_precision_score(y_true, y_proba),  # AUPRC / PR-AUC
                # Class 0 (Negative - No Readmission)
                'class_0': {
                    'precision': precision_score(y_true, y_pred, pos_label=0),
                    'recall': recall_score(y_true, y_pred, pos_label=0),
                    'f1': f1_score(y_true, y_pred, pos_label=0),
                },
                # Class 1 (Positive - Readmission)
                'class_1': {
                    'precision': precision_score(y_true, y_pred, pos_label=1),
                    'recall': recall_score(y_true, y_pred, pos_label=1),
                    'f1': f1_score(y_true, y_pred, pos_label=1),
                },
                # Aggregated
                'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            }
        
        if self.main_effects_model is not None:
             main_metrics = compute_full_metrics(self.y_test, main_pred_binary, main_proba)
        else:
             main_metrics = None

        final_metrics = compute_full_metrics(self.y_test, final_pred_binary, final_proba)
        
        # Store metrics
        metrics = {
            'main_effects_only': main_metrics,
            'final_model': final_metrics
        }
        
        # ============================================================
        # PRINT DETAILED RESULTS
        # ============================================================
        
        logger.info(f"\n{'='*80}")
        logger.info("DETAILED MODEL COMPARISON (Per-Class Metrics)")
        logger.info(f"{'='*80}")
        
        if main_metrics is not None:
            # Overall metrics
            logger.info(f"\n{'OVERALL METRICS':<40} {'Main Effects':<15} {'Final Model':<15}")
            logger.info("-"*80)
            logger.info(f"{'Accuracy':<40} {main_metrics['accuracy']:<15.4f} {final_metrics['accuracy']:<15.4f}")
            logger.info(f"{'Balanced Accuracy':<40} {main_metrics['balanced_accuracy']:<15.4f} {final_metrics['balanced_accuracy']:<15.4f}")
            logger.info(f"{'AUC-ROC':<40} {main_metrics['auc_roc']:<15.4f} {final_metrics['auc_roc']:<15.4f}")
            logger.info(f"{'AUPRC (PR-AUC)':<40} {main_metrics['auprc']:<15.4f} {final_metrics['auprc']:<15.4f}")
            
            # Class 0 (Negative)
            logger.info(f"\n{'CLASS 0 (No Readmission)':<40} {'Main Effects':<15} {'Final Model':<15}")
            logger.info("-"*80)
            logger.info(f"{'Precision (Class 0)':<40} {main_metrics['class_0']['precision']:<15.4f} {final_metrics['class_0']['precision']:<15.4f}")
            logger.info(f"{'Recall (Class 0)':<40} {main_metrics['class_0']['recall']:<15.4f} {final_metrics['class_0']['recall']:<15.4f}")
            logger.info(f"{'F1-Score (Class 0)':<40} {main_metrics['class_0']['f1']:<15.4f} {final_metrics['class_0']['f1']:<15.4f}")
            
            # Class 1 (Positive - Important!)
            logger.info(f"\n{'CLASS 1 (ICU Readmission) ⚠️ IMPORTANT':<40} {'Main Effects':<15} {'Final Model':<15}")
            logger.info("-"*80)
            logger.info(f"{'Precision (Class 1)':<40} {main_metrics['class_1']['precision']:<15.4f} {final_metrics['class_1']['precision']:<15.4f}")
            logger.info(f"{'Recall (Class 1)':<40} {main_metrics['class_1']['recall']:<15.4f} {final_metrics['class_1']['recall']:<15.4f}")
            logger.info(f"{'F1-Score (Class 1)':<40} {main_metrics['class_1']['f1']:<15.4f} {final_metrics['class_1']['f1']:<15.4f}")
        else:
             logger.info(f"\n{'OVERALL METRICS':<40} {'Final Model':<15}")
             logger.info("-"*80)
             logger.info(f"{'Accuracy':<40} {final_metrics['accuracy']:<15.4f}")
             logger.info(f"{'Balanced Accuracy':<40} {final_metrics['balanced_accuracy']:<15.4f}")
             logger.info(f"{'AUC-ROC':<40} {final_metrics['auc_roc']:<15.4f}")
             
             logger.info(f"\n{'CLASS 1 (ICU Readmission) ⚠️ IMPORTANT':<40} {'Final Model':<15}")
             logger.info("-"*80)
             logger.info(f"{'Precision (Class 1)':<40} {final_metrics['class_1']['precision']:<15.4f}")
             logger.info(f"{'Recall (Class 1)':<40} {final_metrics['class_1']['recall']:<15.4f}")
             logger.info(f"{'F1-Score (Class 1)':<40} {final_metrics['class_1']['f1']:<15.4f}")
        
        # Aggregated
        if main_metrics is not None:
             logger.info(f"\n{'AGGREGATED METRICS':<40} {'Main Effects':<15} {'Final Model':<15}")
             logger.info("-"*80)
             logger.info(f"{'F1 (Weighted)':<40} {main_metrics['f1_weighted']:<15.4f} {final_metrics['f1_weighted']:<15.4f}")
             logger.info(f"{'F1 (Macro)':<40} {main_metrics['f1_macro']:<15.4f} {final_metrics['f1_macro']:<15.4f}")
             logger.info(f"{'Precision (Weighted)':<40} {main_metrics['precision_weighted']:<15.4f} {final_metrics['precision_weighted']:<15.4f}")
             logger.info(f"{'Recall (Weighted)':<40} {main_metrics['recall_weighted']:<15.4f} {final_metrics['recall_weighted']:<15.4f}")
        else:
             logger.info(f"\n{'AGGREGATED METRICS':<40} {'Final Model':<15}")
             logger.info("-"*80)
             logger.info(f"{'F1 (Weighted)':<40} {final_metrics['f1_weighted']:<15.4f}")
             logger.info(f"{'F1 (Macro)':<40} {final_metrics['f1_macro']:<15.4f}")
             logger.info(f"{'Precision (Weighted)':<40} {final_metrics['precision_weighted']:<15.4f}")
             logger.info(f"{'Recall (Weighted)':<40} {final_metrics['recall_weighted']:<15.4f}")
        
        # Confusion Matrix
        logger.info(f"\n{'='*80}")
        logger.info("CONFUSION MATRICES")
        logger.info(f"{'='*80}")
        
        if main_metrics is not None:
             main_cm = confusion_matrix(self.y_test, main_pred_binary)
             logger.info("\nMain Effects Model:")
             logger.info(f"                  Predicted 0    Predicted 1")
             logger.info(f"  Actual 0        {main_cm[0,0]:<14} {main_cm[0,1]:<14}")
             logger.info(f"  Actual 1        {main_cm[1,0]:<14} {main_cm[1,1]:<14}")
        
        final_cm = confusion_matrix(self.y_test, final_pred_binary)
        
        logger.info("\nFinal Model (with Interactions):")
        logger.info(f"                  Predicted 0    Predicted 1")
        logger.info(f"  Actual 0        {final_cm[0,0]:<14} {final_cm[0,1]:<14}")
        logger.info(f"  Actual 1        {final_cm[1,0]:<14} {final_cm[1,1]:<14}")
        
        if main_metrics is not None:
             logger.info(f"\n{'='*80}")
             logger.info("IMPROVEMENT SUMMARY")
             logger.info(f"{'='*80}")
             logger.info(f"  Accuracy:           {(final_metrics['accuracy'] - main_metrics['accuracy'])*100:+.2f}%")
             logger.info(f"  Balanced Accuracy:  {(final_metrics['balanced_accuracy'] - main_metrics['balanced_accuracy'])*100:+.2f}%")
             logger.info(f"  AUC-ROC:            {(final_metrics['auc_roc'] - main_metrics['auc_roc'])*100:+.2f}%")
             logger.info(f"  AUPRC (PR-AUC):     {(final_metrics['auprc'] - main_metrics['auprc'])*100:+.2f}%")
             logger.info(f"  F1 (Class 1):       {(final_metrics['class_1']['f1'] - main_metrics['class_1']['f1'])*100:+.2f}%")
             logger.info(f"  Recall (Class 1):   {(final_metrics['class_1']['recall'] - main_metrics['class_1']['recall'])*100:+.2f}%")
        
        # ============================================================
        # THRESHOLD TUNING for optimal clinical performance
        # ============================================================
        logger.info(f"\n{'='*80}")
        logger.info("THRESHOLD TUNING (Clinical Optimization)")
        logger.info(f"{'='*80}")
        
        best_threshold, threshold_metrics = self._find_optimal_threshold(
            self.y_test, final_proba, target_recall=0.80
        )
        
        self.optimal_threshold = best_threshold
        self.threshold_metrics = threshold_metrics
        
        # Apply optimal threshold predictions
        final_pred_optimal = (final_proba >= best_threshold).astype(int)
        optimal_metrics = compute_full_metrics(self.y_test, final_pred_optimal, final_proba)
        
        logger.info(f"\n📊 Default Threshold (0.5):")
        logger.info(f"   Precision: {final_metrics['class_1']['precision']:.4f}")
        logger.info(f"   Recall:    {final_metrics['class_1']['recall']:.4f}")
        logger.info(f"   F1:        {final_metrics['class_1']['f1']:.4f}")
        
        logger.info(f"\n🏆 Optimal Threshold ({best_threshold:.2f}) - High Recall:")
        logger.info(f"   Precision: {optimal_metrics['class_1']['precision']:.4f}")
        logger.info(f"   Recall:    {optimal_metrics['class_1']['recall']:.4f}")
        logger.info(f"   F1:        {optimal_metrics['class_1']['f1']:.4f}")
        
        # Best F1 threshold
        if 'best_f1' in threshold_metrics:
            best_f1_thresh = threshold_metrics['best_f1']['threshold']
            logger.info(f"\n⭐ Best F1 Threshold ({best_f1_thresh:.2f}):")
            logger.info(f"   Precision: {threshold_metrics['best_f1']['precision']:.4f}")
            logger.info(f"   Recall:    {threshold_metrics['best_f1']['recall']:.4f}")
            logger.info(f"   F1:        {threshold_metrics['best_f1']['f1']:.4f}")
        
        # Store optimal metrics
        metrics['optimal_threshold'] = {
            'threshold': best_threshold,
            **optimal_metrics
        }
        
        # ============================================================
        # COST-SENSITIVE EVALUATION
        # ============================================================
        if self.use_cost_sensitive:
            logger.info(f"\n{'='*80}")
            logger.info("COST-SENSITIVE ANALYSIS")
            logger.info(f"{'='*80}")
            
            cost_analysis = self._compute_cost_sensitive_metrics(
                self.y_test, final_pred_binary, final_proba
            )
            metrics['cost_sensitive'] = cost_analysis
            
            # Also compute for optimal threshold
            cost_analysis_optimal = self._compute_cost_sensitive_metrics(
                self.y_test, final_pred_optimal, final_proba
            )
            metrics['cost_sensitive_optimal'] = cost_analysis_optimal
        
        # Store accuracies and metrics
        self.model_accuracies = {
            'main_effects_only': main_accuracy,
            'final_model': final_accuracy
        }
        self.model_metrics = metrics
        self.forced_interactions = forced_interactions
        
        # Store interaction names for saving later
        self.forced_interaction_names = [
            (row['vital_feature'], row['nlp_feature'])
            for _, row in top_interactions.iterrows()
            if row['vital_feature'] in feature_cols and row['nlp_feature'] in feature_cols
        ]
        
        # Log model structure
        logger.info(f"\n{'='*80}")
        logger.info("FINAL MODEL STRUCTURE (ĐÚNG CÔNG THỨC):")
        logger.info(f"{'='*80}")
        logger.info(f"F_final(x) = β₀ + Σf_v(v) + Σf_t(t) + Σf_{{v,t}}(v,t)")
        logger.info(f"  - Main effects X_V: {len(self.X_V_cols)} features")
        logger.info(f"  - Main effects X_T: {len(self.X_T_cols)} features")
        logger.info(f"  - Cross-interactions: {len(forced_interactions)} pairs")
        logger.info(f"  - Total terms: {len(self.X_V_cols) + len(self.X_T_cols) + len(forced_interactions)}")
        
        return self.final_model
    
    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recall: float = 0.80
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold for clinical use.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_recall: Target recall for Class 1 (default 80%)
        
        Returns:
            best_threshold: Optimal threshold
            metrics_dict: Dictionary with metrics for different thresholds
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        # Use finer threshold steps for better precision-recall balance
        thresholds = np.arange(0.05, 0.99, 0.01)
        results = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                continue
            
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Compute cost for cost-based threshold selection
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fn_cost_mult = self.fn_cost_multiplier if self.fn_cost_multiplier is not None else 1.0
            total_cost = fp * 1.0 + fn * fn_cost_mult
            
            results.append({
                'threshold': thresh,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'cost': total_cost,
                'fp': fp,
                'fn': fn
            })
        
        if not results:
            return 0.5, {}
        
        results_df = pd.DataFrame(results)
        
        # ============================================================
        # DYNAMIC THRESHOLD SELECTION based on strategy
        # ============================================================
        logger.info(f"\n  Threshold Strategy: {self.threshold_strategy}")
        
        # Get minimum required metrics (default 0.6 for all)
        min_metrics = getattr(self, 'min_metrics_threshold', 0.6)
        
        if self.threshold_strategy == 'balanced':
            # BALANCED: Find threshold where ALL metrics > min_metrics
            # Priority: maximize F1 while ensuring P >= min, R >= min, F1 >= min
            balanced_df = results_df[
                (results_df['precision'] >= min_metrics) & 
                (results_df['recall'] >= min_metrics) & 
                (results_df['f1'] >= min_metrics)
            ]
            
            if len(balanced_df) > 0:
                # Among balanced thresholds, pick best F1
                best_idx = balanced_df['f1'].idxmax()
                best_threshold = balanced_df.loc[best_idx, 'threshold']
                selection_method = f'balanced (all metrics >= {min_metrics:.0%})'
                logger.info(f"  ✓ Found {len(balanced_df)} thresholds meeting criteria")
            else:
                # Fall back: find closest to meeting all criteria
                logger.warning(f"  ⚠ No threshold found with all metrics >= {min_metrics:.0%}")
                logger.info(f"  → Searching for best compromise...")
                
                # Calculate how far each threshold is from meeting criteria
                results_df['min_metric'] = results_df[['precision', 'recall', 'f1']].min(axis=1)
                results_df['sum_metrics'] = results_df['precision'] + results_df['recall'] + results_df['f1']
                
                # First try: find threshold with highest minimum metric
                best_min_idx = results_df['min_metric'].idxmax()
                best_threshold = results_df.loc[best_min_idx, 'threshold']
                
                best_row = results_df.loc[best_min_idx]
                logger.info(f"  → Best compromise: P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, F1={best_row['f1']:.3f}")
                selection_method = f'balanced-compromise (min_metric={best_row["min_metric"]:.3f})'
        
        elif self.threshold_strategy == 'auto':
            # AUTO: Use cost-based optimization (best overall)
            best_cost_idx = results_df['cost'].idxmin()
            best_threshold = results_df.loc[best_cost_idx, 'threshold']
            selection_method = 'cost-optimized (auto)'
            
        elif self.threshold_strategy == 'f1':
            # F1: Optimize for best F1 score
            best_f1_idx = results_df['f1'].idxmax()
            best_threshold = results_df.loc[best_f1_idx, 'threshold']
            selection_method = 'f1-optimized'
            
        elif self.threshold_strategy == 'recall':
            # RECALL: Target specific recall
            effective_target = target_recall if target_recall else 0.80
            high_recall = results_df[results_df['recall'] >= effective_target]
            if len(high_recall) > 0:
                best_high_rec_idx = high_recall['precision'].idxmax()
                best_threshold = high_recall.loc[best_high_rec_idx, 'threshold']
            else:
                best_threshold = results_df.loc[results_df['recall'].idxmax(), 'threshold']
            selection_method = f'recall-targeted (>={effective_target:.0%})'
            
        elif self.threshold_strategy == 'cost':
            # COST: Minimize total cost
            best_cost_idx = results_df['cost'].idxmin()
            best_threshold = results_df.loc[best_cost_idx, 'threshold']
            selection_method = 'cost-minimized'
            
        else:
            # Default to F1
            best_f1_idx = results_df['f1'].idxmax()
            best_threshold = results_df.loc[best_f1_idx, 'threshold']
            selection_method = 'f1-optimized (default)'
        
        logger.info(f"  Selection Method: {selection_method}")
        logger.info(f"  Optimal Threshold: {best_threshold:.2f}")
        
        # Get metrics for selected threshold
        selected_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
        
        # Best F1 for reference
        best_f1_idx = results_df['f1'].idxmax()
        best_f1_row = results_df.loc[best_f1_idx]
        
        # Best cost for reference
        best_cost_idx = results_df['cost'].idxmin()
        best_cost_row = results_df.loc[best_cost_idx]
        
        metrics_dict = {
            'selected': {
                'threshold': float(best_threshold),
                'method': selection_method,
                'precision': float(selected_row['precision']),
                'recall': float(selected_row['recall']),
                'f1': float(selected_row['f1']),
                'cost': float(selected_row['cost'])
            },
            'best_f1': {
                'threshold': float(best_f1_row['threshold']),
                'precision': float(best_f1_row['precision']),
                'recall': float(best_f1_row['recall']),
                'f1': float(best_f1_row['f1'])
            },
            'best_cost': {
                'threshold': float(best_cost_row['threshold']),
                'precision': float(best_cost_row['precision']),
                'recall': float(best_cost_row['recall']),
                'f1': float(best_cost_row['f1']),
                'cost': float(best_cost_row['cost'])
            },
            'all_thresholds': results
        }
        
        # Store dynamic config
        self.dynamic_config['threshold'] = {
            'strategy': self.threshold_strategy,
            'selected_threshold': float(best_threshold),
            'selection_method': selection_method
        }
        
        return best_threshold, metrics_dict
    
    def _compute_cost_sensitive_weights(
        self,
        y: np.ndarray,
        base_weights: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute cost-sensitive sample weights.
        
        Cost Matrix (Clinical Context):
        - TN (True Negative): Cost = 0 (correct, no action needed)
        - TP (True Positive): Cost = 0 (correct, intervention applied)
        - FP (False Positive): Cost = 1 (unnecessary intervention, but safe)
        - FN (False Negative): Cost = fn_cost_multiplier (missed readmission, DANGEROUS)
        
        For training, we want to penalize misclassifying Class 1 (positive) more.
        Sample weights for Class 1 = fn_cost_multiplier
        Sample weights for Class 0 = 1.0 (or from base_weights)
        
        Args:
            y: True labels
            base_weights: Base weights from class weighting (optional)
        
        Returns:
            cost_sensitive_weights: Array of per-sample weights
        """
        logger.info(f"\nCost-Sensitive Learning Configuration:")
        logger.info(f"  FN Cost Multiplier: {self.fn_cost_multiplier}x")
        logger.info(f"  Rationale: Missing ICU readmission is {self.fn_cost_multiplier}x worse than false alarm")
        
        # Create cost matrix
        self.cost_matrix = {
            'TN': 0,
            'TP': 0,
            'FP': 1,
            'FN': self.fn_cost_multiplier
        }
        
        logger.info(f"\n  Cost Matrix:")
        logger.info(f"    - True Negative (TN):   {self.cost_matrix['TN']} (correct: no readmission predicted correctly)")
        logger.info(f"    - True Positive (TP):   {self.cost_matrix['TP']} (correct: readmission predicted correctly)")
        logger.info(f"    - False Positive (FP):  {self.cost_matrix['FP']} (false alarm, safe but wasteful)")
        logger.info(f"    - False Negative (FN):  {self.cost_matrix['FN']} (MISSED readmission, DANGEROUS)")
        
        # Sample weights: Class 1 gets higher weight
        # This makes the model penalize FN more during training
        n_samples = len(y)
        cost_weights = np.ones(n_samples)
        
        # Apply FN cost to positive class (Class 1)
        # Higher weight = model tries harder to correctly classify these
        positive_mask = (y == 1)
        cost_weights[positive_mask] = self.fn_cost_multiplier
        
        # Combine with base weights if provided
        if base_weights is not None:
            combined_weights = base_weights * cost_weights
            logger.info(f"\n  Combined with class weights: Yes")
        else:
            combined_weights = cost_weights
            logger.info(f"\n  Combined with class weights: No (cost-only)")
        
        # Normalize to prevent exploding gradients
        combined_weights = combined_weights / combined_weights.mean()
        
        logger.info(f"  Final weight for Class 0: {combined_weights[~positive_mask].mean():.4f}")
        logger.info(f"  Final weight for Class 1: {combined_weights[positive_mask].mean():.4f}")
        
        return combined_weights
    
    def _compute_cost_sensitive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Compute cost-sensitive metrics.
        
        Total Cost = FP * cost_FP + FN * cost_FN
        
        Returns:
            Dictionary with cost metrics
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute costs
        cost_fp = fp * self.cost_matrix['FP']  # False alarms
        cost_fn = fn * self.cost_matrix['FN']  # Missed readmissions (EXPENSIVE!)
        total_cost = cost_fp + cost_fn
        
        # Normalize by number of samples
        avg_cost_per_sample = total_cost / len(y_true)
        
        # Cost savings compared to always predicting negative
        # If we always predict negative: all positives are FN
        baseline_cost = y_true.sum() * self.cost_matrix['FN']
        cost_savings = (baseline_cost - total_cost) / baseline_cost * 100 if baseline_cost > 0 else 0
        
        logger.info(f"\n📊 Cost-Sensitive Metrics:")
        logger.info(f"   Confusion Matrix:")
        logger.info(f"     TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.info(f"\n   Costs Incurred:")
        logger.info(f"     FP cost (false alarms): {fp} × {self.cost_matrix['FP']} = {cost_fp}")
        logger.info(f"     FN cost (missed):       {fn} × {self.cost_matrix['FN']} = {cost_fn}")
        logger.info(f"     Total Cost:             {total_cost}")
        logger.info(f"     Avg Cost/Sample:        {avg_cost_per_sample:.4f}")
        logger.info(f"\n   💰 Cost Savings vs Baseline (all negative): {cost_savings:.2f}%")
        
        return {
            'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
            'cost_fp': int(cost_fp),
            'cost_fn': float(cost_fn),
            'total_cost': float(total_cost),
            'avg_cost_per_sample': float(avg_cost_per_sample),
            'baseline_cost': float(baseline_cost),
            'cost_savings_pct': float(cost_savings)
        }
    
    def _create_interaction_features(
        self,
        X: pd.DataFrame,
        interactions: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Tạo features từ các cặp interaction: v * t
        """
        interaction_df = pd.DataFrame(index=X.index)
        
        for v_col, t_col in interactions:
            if v_col in X.columns and t_col in X.columns:
                # Tạo interaction feature: v * t
                col_name = f"{v_col}__x__{t_col}"
                interaction_df[col_name] = X[v_col].values * X[t_col].values
        
        return interaction_df
    
    def predict_final(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict sử dụng Final Model (trained từ đầu theo đúng công thức).
        
        F_final(x) = β₀ + Σf_v(v) + Σf_t(t) + Σf_{v,t}(v,t)
        """
        return self.final_model.predict(X)
    
    def predict_proba_final(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability sử dụng Final Model.
        """
        return self.final_model.predict_proba(X)
    
    def predict_with_optimal_threshold(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using optimal threshold (high recall for clinical use).
        
        Returns binary predictions using the optimal threshold found during training.
        """
        if not hasattr(self, 'optimal_threshold'):
            logger.warning("No optimal threshold found. Using default 0.5")
            return self.final_model.predict(X)
        
        proba = self.final_model.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)
    
    def predict_with_threshold(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        Predict using a custom threshold.
        
        Args:
            X: Feature dataframe
            threshold: Custom threshold (0-1)
        
        Returns:
            Binary predictions
        """
        proba = self.final_model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def run_full_pipeline(
        self,
        vital_df: pd.DataFrame,
        nlp_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        top_n: int = 50,
        cached_interactions_path: str = None
    ) -> Dict:
        """
        Run the complete FAST pipeline for cross-interaction discovery.
        """
        logger.info("="*70)
        logger.info("CROSS-INTERACTION DISCOVERY PIPELINE (FAST Algorithm)")
        logger.info("="*70)
        
        # Identify feature sets
        self.X_V_cols, self.X_T_cols = self._identify_feature_sets(vital_df, nlp_df)
        
        # Prepare data
        X, y = self._prepare_data(vital_df, nlp_df, labels_df)
        
        # Step 1: Train main effects model (or just prepare data if using cache)
        # If cached interactions provided, we skip training main effects
        residuals = self.step1_train_main_effects(X, y, skip_training=(cached_interactions_path is not None))
        
        # Step 2: Rank interactions (dùng training data để tránh data leakage)
        # Sử dụng X_train và residuals_train thay vì full data
        if cached_interactions_path and os.path.exists(cached_interactions_path):
            logger.info(f"Loading cached interactions from {cached_interactions_path}")
            interaction_scores = pd.read_csv(cached_interactions_path)
            # Ensure required columns exist
            if 'vital_feature' in interaction_scores.columns and 'nlp_feature' in interaction_scores.columns:
                self.interaction_scores = interaction_scores
                logger.info(f"Loaded {len(interaction_scores)} interactions from cache.")
            else:
                logger.error("Cached interaction file is missing required columns. Re-running discovery.")
                interaction_scores = self.step2_rank_interactions(self.X_train, self.residuals_train)
        else:
            if cached_interactions_path:
                logger.warning(f"Cached interaction file {cached_interactions_path} not found. Re-running discovery.")
            interaction_scores = self.step2_rank_interactions(self.X_train, self.residuals_train)
        
        # Step 3: Train final model
        final_model = self.step3_train_final_model(top_n=top_n)
        
        # Compile results
        results = {
            'X_V_features': self.X_V_cols,
            'X_T_features': self.X_T_cols,
            'interaction_scores': interaction_scores,
            'main_effects_model': self.main_effects_model,
            'final_model': self.final_model,
            'residuals': self.residuals,
            'top_n': top_n
        }
        
        return results
    
    def save_results(self, output_dir: str, results: Dict):
        """
        Save all results to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save interaction scores
        results['interaction_scores'].to_csv(
            os.path.join(output_dir, 'interaction_scores.csv'),
            index=False
        )
        
        # Save models
        if hasattr(results, 'get') and results.get('main_effects_model') is not None:
             with open(os.path.join(output_dir, 'main_effects_model.pkl'), 'wb') as f:
                pickle.dump(results['main_effects_model'], f)
        elif hasattr(self, 'main_effects_model') and self.main_effects_model is not None:
             with open(os.path.join(output_dir, 'main_effects_model.pkl'), 'wb') as f:
                pickle.dump(self.main_effects_model, f)
        
        with open(os.path.join(output_dir, 'final_model.pkl'), 'wb') as f:
            pickle.dump(results['final_model'], f)
        
        # Save model accuracies
        if hasattr(self, 'model_accuracies'):
            with open(os.path.join(output_dir, 'model_accuracies.json'), 'w') as f:
                json.dump(self.model_accuracies, f, indent=2)
        
        # Save detailed metrics (including imbalance-aware metrics)
        if hasattr(self, 'model_metrics'):
            with open(os.path.join(output_dir, 'model_metrics_detailed.json'), 'w') as f:
                json.dump(self.model_metrics, f, indent=2)

        # Also save a compact summary used by other scripts
        if hasattr(self, 'model_metrics'):
            try:
                final_metrics = self.model_metrics['final_model']
                threshold = getattr(self, 'optimal_threshold', 0.5)
                summary = {
                    'threshold': float(threshold),
                    'class_1': final_metrics.get('class_1', {}),
                    'accuracy': float(final_metrics.get('accuracy', 0)),
                    'auprc': float(final_metrics.get('auprc', 0)),
                    'auroc': float(final_metrics.get('auc_roc', 0)),
                    'confusion_matrix': self.model_metrics.get('cost_sensitive', {}).get('confusion_matrix', {}),
                    'config': {
                        'seed': int(self.random_state),
                        'undersampling_ratio': float(self.undersampling_ratio) if self.undersampling_ratio else None,
                        'n_features': len(self.X_full.columns) if hasattr(self, 'X_full') else None,
                        'n_vital_features': len(self.X_V_cols) if self.X_V_cols else None,
                        'n_nlp_features': len(self.X_T_cols) if self.X_T_cols else None,
                        'n_cross_interactions': len(self.forced_interaction_names) if hasattr(self, 'forced_interaction_names') else 0,
                        'model_type': 'EBM_cross_interaction'
                    }
                }
                with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write compact model_metrics.json: {e}")
        
        # Save class weights info
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            class_weights_info = {
                'use_dynamic_weighting': self.use_dynamic_weighting,
                'class_weight_strategy': self.class_weight_strategy,
                'class_weights': {str(k): v for k, v in self.class_weights.items()}
            }
            with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
                json.dump(class_weights_info, f, indent=2)
        
        # Save cost-sensitive learning info
        if hasattr(self, 'cost_matrix') and self.cost_matrix is not None:
            cost_sensitive_info = {
                'use_cost_sensitive': self.use_cost_sensitive,
                'fn_cost_multiplier': float(self.fn_cost_multiplier) if self.fn_cost_multiplier else None,
                'fn_cost_config': str(self.fn_cost_multiplier_config),
                'cost_matrix': {k: float(v) for k, v in self.cost_matrix.items()},
                'rationale': 'Missing ICU readmission (FN) is more costly than false alarm (FP)'
            }
            with open(os.path.join(output_dir, 'cost_sensitive_config.json'), 'w') as f:
                json.dump(cost_sensitive_info, f, indent=2)
        
        # Save FULLY DYNAMIC configuration
        if hasattr(self, 'dynamic_config') and self.dynamic_config:
            dynamic_info = {
                'fully_dynamic': True,
                'description': 'All parameters automatically computed from data',
                'configurations': self.dynamic_config,
                'imbalance_ratio': float(self.imbalance_ratio) if self.imbalance_ratio else None,
                'clinical_severity': float(self.clinical_severity),
                'threshold_strategy': self.threshold_strategy
            }
            with open(os.path.join(output_dir, 'dynamic_config.json'), 'w') as f:
                json.dump(dynamic_info, f, indent=2)
        
        # Save forced interactions
        if hasattr(self, 'forced_interaction_names'):
            interactions_info = {
                'forced_interactions': [
                    {'vital_feature': v, 'nlp_feature': t} 
                    for v, t in self.forced_interaction_names
                ],
                'count': len(self.forced_interaction_names)
            }
            with open(os.path.join(output_dir, 'forced_interactions.json'), 'w') as f:
                json.dump(interactions_info, f, indent=2)
        
        # Save feature lists
        feature_info = {
            'X_V_features': results['X_V_features'],
            'X_T_features': results['X_T_features'],
            'top_n': results['top_n'],
            'formula': 'F_final(x) = β₀ + Σf_v(v) + Σf_t(t) + Σf_{v,t}(v,t)'
        }
        with open(os.path.join(output_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save top interactions for easy reference
        top_interactions = results['interaction_scores'].head(results['top_n'])
        top_interactions.to_csv(
            os.path.join(output_dir, f'top_{results["top_n"]}_interactions.csv'),
            index=False
        )
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_xai_visualizations(
        self,
        output_dir: str,
        top_n: int = 10,
        language: str = 'vi'
    ):
        """
        Tạo các biểu đồ XAI (Explainable AI) cho cross-interactions.
        
        Bao gồm:
        1. Heatmaps 2D: Trục X = Vital sign, Trục Y = Bệnh lý, Màu = Log-odds
        2. Threshold Comparison: So sánh ngưỡng nguy hiểm giữa các bệnh lý
        3. Clinical Dashboard: Dashboard tổng hợp cho bác sĩ
        
        Args:
            output_dir: Thư mục lưu visualizations
            top_n: Số lượng top interactions để visualize
            language: 'vi' cho tiếng Việt, 'en' cho tiếng Anh
        """
        from ebm_xai_visualization import EBMXAIVisualizer
        
        logger.info("="*60)
        logger.info("GENERATING XAI VISUALIZATIONS")
        logger.info("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize visualizer
        visualizer = EBMXAIVisualizer(
            self.final_model, 
            language=language
        )
        
        # Get top interactions
        if hasattr(self, 'interaction_scores') and self.interaction_scores is not None:
            top_interactions = self.interaction_scores.head(top_n)
        else:
            logger.warning("No interaction scores available. Run pipeline first.")
            return
        
        # 1. Generate Clinical Dashboard
        logger.info("Generating clinical dashboard...")
        X_data = self.X_full if hasattr(self, 'X_full') else None
        visualizer.generate_clinical_dashboard(
            top_interactions,
            X_data=X_data,
            top_n=min(6, top_n),
            save_dir=output_dir
        )
        
        # 2. Generate individual heatmaps for each top interaction
        logger.info(f"Generating {top_n} individual heatmaps...")
        heatmap_dir = os.path.join(output_dir, 'heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        for idx, row in top_interactions.iterrows():
            vital_feat = row['vital_feature']
            nlp_feat = row['nlp_feature']
            
            try:
                visualizer.plot_interaction_heatmap(
                    vital_feat, nlp_feat,
                    X_data=X_data,
                    save_path=os.path.join(
                        heatmap_dir, 
                        f'heatmap_{idx+1:02d}_{vital_feat}_{nlp_feat}.png'
                    )
                )
            except Exception as e:
                logger.warning(f"Could not generate heatmap for {vital_feat} × {nlp_feat}: {e}")
        
        # 3. Generate threshold comparison charts
        logger.info("Generating threshold comparison charts...")
        threshold_dir = os.path.join(output_dir, 'threshold_comparisons')
        os.makedirs(threshold_dir, exist_ok=True)
        
        # Get unique vital features from top interactions
        vital_features = top_interactions['vital_feature'].unique()[:5]
        
        for vital_feat in vital_features:
            # Get NLP features that interact with this vital
            nlp_features = top_interactions[
                top_interactions['vital_feature'] == vital_feat
            ]['nlp_feature'].tolist()[:5]
            
            if len(nlp_features) >= 2:
                try:
                    visualizer.plot_risk_threshold_comparison(
                        vital_feat,
                        nlp_features,
                        X_data=X_data,
                        save_path=os.path.join(
                            threshold_dir, 
                            f'threshold_{vital_feat}.png'
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not generate threshold chart for {vital_feat}: {e}")
        
        # 4. Export interaction data for web visualization
        logger.info("Exporting interaction data for web...")
        visualizer.export_interaction_data(
            top_interactions,
            X_data if X_data is not None else pd.DataFrame(),
            os.path.join(output_dir, 'interaction_data.json')
        )
        
        # 5. Generate multiple interaction heatmaps in one figure
        logger.info("Generating multi-interaction heatmap grid...")
        interactions_list = [
            (row['vital_feature'], row['nlp_feature'])
            for _, row in top_interactions.head(6).iterrows()
        ]
        
        try:
            visualizer.plot_multiple_interactions(
                interactions_list,
                X_data=X_data,
                ncols=3,
                save_path=os.path.join(output_dir, 'multi_interaction_grid.png')
            )
        except Exception as e:
            logger.warning(f"Could not generate multi-interaction grid: {e}")
        
        logger.info(f"XAI visualizations saved to {output_dir}")
        logger.info(f"  - xai_clinical_dashboard.png")
        logger.info(f"  - heatmaps/ ({top_n} individual heatmaps)")
        logger.info(f"  - threshold_comparisons/")
        logger.info(f"  - interaction_data.json")
        logger.info(f"  - multi_interaction_grid.png")


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Interaction Discovery using FAST Algorithm'
    )
    parser.add_argument(
        '--vital-features', required=False,
        help='CSV with vital/lab features'
    )
    parser.add_argument(
        '--nlp-features', required=False,
        help='CSV with NLP Bag-of-Concepts features'
    )
    parser.add_argument(
        '--features', required=False,
        help='CSV with combined features (alternative to --vital-features and --nlp-features)'
    )
    parser.add_argument(
        '--labels', required=True,
        help='CSV with labels (HADM_ID, Y)'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--top-n', type=int, default=50,
        help='Number of top interactions to include (default: 50)'
    )
    parser.add_argument(
        '--n-bins', type=int, default=10,
        help='Number of bins for continuous features (default: 10)'
    )
    parser.add_argument(
        '--min-samples', type=int, default=50,
        help='Minimum samples per bin (default: 50)'
    )
    parser.add_argument(
        '--use-dynamic-weighting', action='store_true', default=True,
        help='Use dynamic class weighting for imbalanced data (default: True)'
    )
    parser.add_argument(
        '--no-dynamic-weighting', action='store_false', dest='use_dynamic_weighting',
        help='Disable dynamic class weighting'
    )
    parser.add_argument(
        '--weight-strategy', type=str, default='balanced',
        choices=['balanced', 'sqrt_balanced', 'inverse_freq', 'effective_num'],
        help='Strategy for computing class weights (default: balanced)'
    )
    parser.add_argument(
        '--use-cost-sensitive', action='store_true', default=True,
        help='Use cost-sensitive learning to penalize False Negatives more (default: True)'
    )
    parser.add_argument(
        '--no-cost-sensitive', action='store_false', dest='use_cost_sensitive',
        help='Disable cost-sensitive learning'
    )
    parser.add_argument(
        '--fn-cost', type=str, default='auto',
        help='False Negative cost multiplier. "auto" computes from imbalance ratio (RECOMMENDED), '
             '"clinical" uses clinical severity, or a float value (e.g., 5.0)'
    )
    parser.add_argument(
        '--clinical-severity', type=float, default=1.5,
        help='Clinical severity multiplier for cost computation (default: 1.5)'
    )
    parser.add_argument(
        '--threshold-strategy', type=str, default='auto', 
        choices=['auto', 'f1', 'recall', 'cost', 'balanced'],
        help='Strategy for threshold selection: auto, f1, recall, cost, balanced (default: auto)'
    )
    parser.add_argument(
        '--target-recall', type=float, default=0.7,
        help='Target recall for Class 1 when threshold_strategy=recall (default: 0.7)'
    )
    parser.add_argument(
        '--min-metrics', type=float, default=0.6,
        help='Minimum required P/R/F1 when using balanced strategy (default: 0.6)'
    )
    parser.add_argument(
        '--use-calibration', action='store_true', default=False,
        help='Use calibration for final model (split train into subtrain + cal set)'
    )
    parser.add_argument(
        '--undersampling-ratio', type=float, default=None,
        help='Undersampling ratio for final model (e.g., 4.0 = majority:minority 4:1)'
    )
    # EBM Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--max-leaves', type=int, default=4)
    parser.add_argument('--min-samples-leaf', type=int, default=10)
    parser.add_argument('--max-bins', type=int, default=128)
    parser.add_argument('--outer-bags', type=int, default=16)
    parser.add_argument('--inner-bags', type=int, default=8)
    parser.add_argument('--max-rounds', type=int, default=10000)
    parser.add_argument('--early-stopping', type=int, default=100)
    parser.add_argument('--interactions', type=int, default=0, help='Max interactions (0=auto based on top-n)')
    # XAI Visualization arguments
    parser.add_argument(
        '--generate-xai', action='store_true', default=False,
        help='Generate XAI visualizations (heatmaps, dashboards)'
    )
    parser.add_argument(
        '--xai-top-n', type=int, default=10,
        help='Number of top interactions to visualize (default: 10)'
    )
    parser.add_argument(
        '--xai-language', type=str, default='vi', choices=['vi', 'en'],
        help='Language for XAI labels (vi=Vietnamese, en=English, default: vi)'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cached-interactions', type=str, default=None, help='Path to cached interaction_scores.csv')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.features is None and (args.vital_features is None or args.nlp_features is None):
        parser.error("Either --features OR both --vital-features and --nlp-features must be provided")
    
    # Load data
    logger.info("Loading data...")
    
    if args.features:
        # Combined features mode
        combined_df = pd.read_csv(args.features, low_memory=False)
        labels_df = pd.read_csv(args.labels)
        logger.info(f"Combined features: {combined_df.shape}")
        
        # Split into vital and nlp features based on column patterns
        nlp_patterns = ['nlp_', 'concept_', 'entity_', 'umls_', 'mention_', 'discharge_']
        nlp_cols = [c for c in combined_df.columns 
                    if any(p in c.lower() for p in nlp_patterns)]
        
        # ID columns
        id_cols = ['HADM_ID', 'hadm_id', 'SUBJECT_ID', 'subject_id', 'icustay_id', 'ICUSTAY_ID']
        id_cols_present = [c for c in id_cols if c in combined_df.columns]
        
        # Vital columns = everything except NLP and IDs
        vital_cols = [c for c in combined_df.columns 
                      if c not in nlp_cols and c not in id_cols and c != 'Y' and c != 'label']
        
        vital_df = combined_df[id_cols_present + vital_cols].copy()
        
        if nlp_cols:
            nlp_df = combined_df[id_cols_present + nlp_cols].copy()
        else:
            # If no NLP columns detected, create a dummy
            nlp_df = combined_df[id_cols_present].copy()
            nlp_df['dummy_nlp'] = 0
            nlp_cols = ['dummy_nlp']
        
        logger.info(f"Detected {len(vital_cols)} vital/lab features")
        logger.info(f"Detected {len(nlp_cols)} NLP features")
    else:
        # Separate files mode
        vital_df = pd.read_csv(args.vital_features, low_memory=False)
        nlp_df = pd.read_csv(args.nlp_features, low_memory=False)
        labels_df = pd.read_csv(args.labels)
        logger.info(f"Vital features: {vital_df.shape}")
        logger.info(f"NLP features: {nlp_df.shape}")
    
    logger.info(f"Labels: {labels_df.shape}")
    
    # Normalize label column
    if 'Y' in labels_df.columns:
        labels_df = labels_df.rename(columns={'Y': 'label'})
    if 'HADM_ID' in labels_df.columns:
        labels_df = labels_df.rename(columns={'HADM_ID': 'hadm_id'})
    
    # Parse fn_cost - can be 'auto', 'clinical', or a float
    fn_cost = args.fn_cost
    if fn_cost not in ['auto', 'clinical']:
        try:
            fn_cost = float(fn_cost)
        except ValueError:
            logger.warning(f"Invalid fn_cost '{fn_cost}', using 'auto'")
            fn_cost = 'auto'
    
    # Initialize and run pipeline
    discovery = CrossInteractionDiscovery(
        n_bins=args.n_bins,
        min_samples_per_bin=args.min_samples,
        random_state=args.seed,
        use_dynamic_weighting=args.use_dynamic_weighting,
        class_weight_strategy=args.weight_strategy,
        use_cost_sensitive=args.use_cost_sensitive,
        fn_cost_multiplier=fn_cost,
        threshold_strategy=args.threshold_strategy,
        target_recall=args.target_recall,
        clinical_severity=args.clinical_severity,
        min_metrics_threshold=args.min_metrics
        ,use_calibration=args.use_calibration
        ,undersampling_ratio=args.undersampling_ratio
        ,learning_rate=args.learning_rate
        ,max_leaves=args.max_leaves
        ,min_samples_leaf=args.min_samples_leaf
        ,max_bins=args.max_bins
        ,outer_bags=args.outer_bags
        ,inner_bags=args.inner_bags
        ,max_rounds=args.max_rounds
        ,early_stopping_rounds=args.early_stopping
        ,interactions=args.interactions
    )
    
    results = discovery.run_full_pipeline(
        vital_df, nlp_df, labels_df,
        top_n=args.top_n,
        cached_interactions_path=args.cached_interactions
    )
    
    # Save results
    discovery.save_results(args.output, results)
    
    # Generate XAI visualizations if requested
    if args.generate_xai:
        xai_output_dir = os.path.join(args.output, 'xai_visualizations')
        discovery.generate_xai_visualizations(
            output_dir=xai_output_dir,
            top_n=args.xai_top_n,
            language=args.xai_language
        )
    
    # Print summary
    print("\n" + "="*70)
    print("CROSS-INTERACTION DISCOVERY SUMMARY")
    print("="*70)
    print(f"\nFeature Sets:")
    print(f"  Vital/Lab features (X_V): {len(results['X_V_features'])}")
    print(f"  NLP features (X_T): {len(results['X_T_features'])}")
    print(f"  Cross-interaction space: {len(results['X_V_features']) * len(results['X_T_features']):,} pairs")
    
    if args.use_dynamic_weighting:
        print(f"\nDynamic Class Weighting: ENABLED")
        print(f"  Strategy: {args.weight_strategy}")
    else:
        print(f"\nDynamic Class Weighting: DISABLED")
    
    if args.use_cost_sensitive:
        print(f"\nCost-Sensitive Learning: ENABLED")
        print(f"  FN Cost Multiplier: {args.fn_cost}x")
        print(f"  Rationale: Missing ICU readmission is {args.fn_cost}x worse than false alarm")
    else:
        print(f"\nCost-Sensitive Learning: DISABLED")
    
    print(f"\nTop {args.top_n} Cross-Interactions Discovered:")
    print("-"*70)
    for i, row in results['interaction_scores'].head(20).iterrows():
        print(f"{i+1:2d}. {row['vital_feature']:<25} × {row['nlp_feature']:<30}")
        print(f"    Score: {row['interaction_score']:.4f} | "
              f"Residual diff: {row['residual_diff_score']:.4f} | "
              f"F-score: {row['f_score']:.4f}")
    
    # Print XAI output info if generated
    if args.generate_xai:
        print(f"\n📊 XAI Visualizations Generated:")
        print(f"  - {os.path.join(args.output, 'xai_visualizations')}/")
        print(f"    ├── xai_clinical_dashboard.png")
        print(f"    ├── heatmaps/ ({args.xai_top_n} individual heatmaps)")
        print(f"    ├── threshold_comparisons/")
        print(f"    ├── multi_interaction_grid.png")
        print(f"    └── interaction_data.json")
    
    print(f"\nOutput files saved to: {args.output}")
    print(f"  - interaction_scores.csv (all ranked interactions)")
    print(f"  - top_{args.top_n}_interactions.csv")
    print(f"  - main_effects_model.pkl")
    print(f"  - final_model.pkl")
    print(f"  - feature_info.json")


if __name__ == "__main__":
    main()

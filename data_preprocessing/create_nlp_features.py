#!/usr/bin/env python3
"""
Create NLP features (Bag-of-Concepts) from extracted clinical entities with UMLS CUIs.

This script:
1. Aggregates entities by HADM_ID 
2. Filters by frequency (keep concepts in at least X% of patients)
3. Creates binary encoding (0/1) for concept presence
4. Performs feature selection using Chi-square or Mutual Information
5. Outputs feature matrix ready for EBM model

Usage:
    python create_nlp_features.py \
        --entities outputs/cohort_clinical_entities_umls_fixed.csv \
        --labels cohort/features_locf_v2.csv \
        --output outputs/nlp_features_boc.csv \
        --min-freq 0.01 \
        --top-k 500 \
        --method chi2
"""

import argparse
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from collections import defaultdict
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_entities(filepath: str) -> pd.DataFrame:
    """Load extracted entities with UMLS CUIs."""
    logger.info(f"Loading entities from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} entities")
    return df


def load_labels(filepath: str, label_col: str = 'Y') -> pd.DataFrame:
    """Load labels (readmission) from features file."""
    logger.info(f"Loading labels from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Find HADM_ID column
    hadm_col = None
    for col in ['HADM_ID', 'hadm_id']:
        if col in df.columns:
            hadm_col = col
            break
    
    if hadm_col is None:
        raise ValueError("Could not find HADM_ID column in labels file")
    
    # Find label column
    if label_col not in df.columns:
        # Try to find it
        for col in ['Y', 'y', 'label', 'readmission', 'READMISSION']:
            if col in df.columns:
                label_col = col
                break
    
    if label_col not in df.columns:
        raise ValueError(f"Could not find label column '{label_col}' in labels file")
    
    df = df[[hadm_col, label_col]].copy()
    df.columns = ['hadm_id', 'label']
    df['hadm_id'] = df['hadm_id'].astype(int)
    
    logger.info(f"Loaded {len(df):,} patients with labels")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def create_concept_features(
    entities_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    min_freq: float = 0.01,
    include_negation: bool = True,
    include_category: bool = True
) -> tuple:
    """
    Create concept-based features from entities.
    
    Args:
        entities_df: DataFrame with entities and UMLS CUIs
        labels_df: DataFrame with HADM_ID and labels
        min_freq: Minimum frequency threshold (fraction of patients)
        include_negation: Whether to create separate features for negated concepts
        include_category: Whether to include category (PROBLEM/TREATMENT/TEST) in feature name
    
    Returns:
        feature_matrix: DataFrame with binary features
        concept_info: Dictionary with concept statistics
    """
    logger.info("Creating concept features...")
    
    # Filter to entities with CUI
    entities_with_cui = entities_df[entities_df['umls_cui_mapped'].notna()].copy()
    logger.info(f"Entities with CUI: {len(entities_with_cui):,}")
    
    # Normalize hadm_id
    entities_with_cui['hadm_id'] = entities_with_cui['hadm_id'].astype(int)
    
    # Get unique patients from labels
    valid_hadm_ids = set(labels_df['hadm_id'].unique())
    entities_with_cui = entities_with_cui[entities_with_cui['hadm_id'].isin(valid_hadm_ids)]
    logger.info(f"Entities for valid patients: {len(entities_with_cui):,}")
    
    # Create concept identifier
    def create_concept_id(row):
        cui = row['umls_cui_mapped']
        category = row['entity_label'] if include_category else ''
        negated = '_NEG' if include_negation and row.get('is_negated', False) else ''
        
        if include_category:
            return f"{category}_{cui}{negated}"
        else:
            return f"{cui}{negated}"
    
    entities_with_cui['concept_id'] = entities_with_cui.apply(create_concept_id, axis=1)
    
    # Count concepts per patient
    logger.info("Counting concepts per patient...")
    patient_concepts = entities_with_cui.groupby(['hadm_id', 'concept_id']).size().reset_index(name='count')
    
    # Get unique concepts
    all_concepts = patient_concepts['concept_id'].unique()
    logger.info(f"Total unique concepts: {len(all_concepts):,}")
    
    # Calculate concept frequency (fraction of patients)
    n_patients = len(valid_hadm_ids)
    concept_patient_count = patient_concepts.groupby('concept_id')['hadm_id'].nunique()
    concept_freq = concept_patient_count / n_patients
    
    # Filter by minimum frequency
    min_count = int(min_freq * n_patients)
    frequent_concepts = concept_freq[concept_freq >= min_freq].index.tolist()
    logger.info(f"Concepts with freq >= {min_freq:.1%} ({min_count}+ patients): {len(frequent_concepts):,}")
    
    # Create binary feature matrix
    logger.info("Creating binary feature matrix...")
    
    # Filter to frequent concepts
    patient_concepts_filtered = patient_concepts[patient_concepts['concept_id'].isin(frequent_concepts)]
    
    # Pivot to create feature matrix
    feature_matrix = patient_concepts_filtered.pivot_table(
        index='hadm_id',
        columns='concept_id',
        values='count',
        fill_value=0
    )
    
    # Convert to binary (presence/absence)
    feature_matrix = (feature_matrix > 0).astype(int)
    
    # Ensure all patients from labels are in the matrix
    missing_patients = set(labels_df['hadm_id']) - set(feature_matrix.index)
    if missing_patients:
        logger.info(f"Adding {len(missing_patients)} patients with no concepts...")
        missing_df = pd.DataFrame(
            0,
            index=list(missing_patients),
            columns=feature_matrix.columns
        )
        feature_matrix = pd.concat([feature_matrix, missing_df])
    
    # Sort by hadm_id
    feature_matrix = feature_matrix.sort_index()
    
    # Create concept info
    concept_info = {}
    for concept in frequent_concepts:
        # Get original CUI info
        cui = concept.split('_')[1] if include_category else concept.split('_')[0]
        cui = cui.replace('_NEG', '')
        
        sample = entities_with_cui[entities_with_cui['umls_cui_mapped'] == cui].iloc[0] if len(entities_with_cui[entities_with_cui['umls_cui_mapped'] == cui]) > 0 else None
        
        concept_info[concept] = {
            'cui': cui,
            'name': sample['umls_name_mapped'] if sample is not None else 'Unknown',
            'frequency': concept_freq.get(concept, 0),
            'patient_count': concept_patient_count.get(concept, 0)
        }
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, concept_info


def select_features(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    method: str = 'chi2',
    top_k: int = 500
) -> tuple:
    """
    Select top-K features using statistical tests.
    
    Args:
        feature_matrix: Binary feature matrix
        labels: Target labels
        method: 'chi2' for Chi-square test, 'mi' for Mutual Information
        top_k: Number of top features to select
    
    Returns:
        selected_features: DataFrame with selected features
        feature_scores: DataFrame with feature importance scores
    """
    logger.info(f"Selecting top {top_k} features using {method}...")
    
    # Align labels with feature matrix
    aligned_labels = labels.loc[feature_matrix.index]
    
    X = feature_matrix.values
    y = aligned_labels.values
    
    # Select method
    if method == 'chi2':
        scores, pvalues = chi2(X, y)
        score_name = 'chi2_score'
        pvalue_name = 'chi2_pvalue'
    elif method == 'mi':
        scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        pvalues = np.zeros_like(scores)  # MI doesn't have p-values
        score_name = 'mi_score'
        pvalue_name = 'mi_pvalue'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create scores DataFrame
    feature_scores = pd.DataFrame({
        'feature': feature_matrix.columns,
        score_name: scores,
        pvalue_name: pvalues
    })
    feature_scores = feature_scores.sort_values(score_name, ascending=False)
    
    # Select top-K
    top_k = min(top_k, len(feature_scores))
    selected_features_list = feature_scores.head(top_k)['feature'].tolist()
    
    selected_features = feature_matrix[selected_features_list]
    
    logger.info(f"Selected {len(selected_features_list)} features")
    
    # Log top features
    logger.info("Top 20 features:")
    for i, row in feature_scores.head(20).iterrows():
        logger.info(f"  {row['feature']}: {row[score_name]:.4f}")
    
    return selected_features, feature_scores


def main():
    parser = argparse.ArgumentParser(description='Create NLP features (Bag-of-Concepts)')
    parser.add_argument('--entities', required=True, help='Input CSV with entities and UMLS CUIs')
    parser.add_argument('--labels', required=True, help='CSV with HADM_ID and labels')
    parser.add_argument('--output', required=True, help='Output CSV for features')
    parser.add_argument('--output-scores', default=None, help='Output CSV for feature scores')
    parser.add_argument('--min-freq', type=float, default=0.01, help='Minimum frequency threshold (default: 0.01 = 1%%)')
    parser.add_argument('--top-k', type=int, default=500, help='Number of top features to select (default: 500)')
    parser.add_argument('--method', choices=['chi2', 'mi'], default='chi2', help='Feature selection method')
    parser.add_argument('--no-negation', action='store_true', help='Do not create separate features for negated concepts')
    parser.add_argument('--no-category', action='store_true', help='Do not include category in feature names')
    parser.add_argument('--label-col', default='Y', help='Label column name (default: Y)')
    
    args = parser.parse_args()
    
    # Load data
    entities_df = load_entities(args.entities)
    labels_df = load_labels(args.labels, args.label_col)
    
    # Create concept features
    feature_matrix, concept_info = create_concept_features(
        entities_df,
        labels_df,
        min_freq=args.min_freq,
        include_negation=not args.no_negation,
        include_category=not args.no_category
    )
    
    # Create labels series indexed by hadm_id
    labels_series = labels_df.set_index('hadm_id')['label']
    
    # Select features
    selected_features, feature_scores = select_features(
        feature_matrix,
        labels_series,
        method=args.method,
        top_k=args.top_k
    )
    
    # Add hadm_id column
    selected_features = selected_features.reset_index()
    selected_features = selected_features.rename(columns={'index': 'hadm_id'})
    
    # Merge with labels
    selected_features = selected_features.merge(labels_df, on='hadm_id', how='left')
    
    # Save features
    logger.info(f"Saving features to {args.output}...")
    selected_features.to_csv(args.output, index=False)
    
    # Save feature scores
    if args.output_scores:
        logger.info(f"Saving feature scores to {args.output_scores}...")
        
        # Add concept info to scores
        feature_scores['cui'] = feature_scores['feature'].apply(
            lambda x: concept_info.get(x, {}).get('cui', '')
        )
        feature_scores['name'] = feature_scores['feature'].apply(
            lambda x: concept_info.get(x, {}).get('name', '')
        )
        feature_scores['frequency'] = feature_scores['feature'].apply(
            lambda x: concept_info.get(x, {}).get('frequency', 0)
        )
        feature_scores['patient_count'] = feature_scores['feature'].apply(
            lambda x: concept_info.get(x, {}).get('patient_count', 0)
        )
        
        feature_scores.to_csv(args.output_scores, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("BAG-OF-CONCEPTS FEATURE EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total entities processed: {len(entities_df):,}")
    print(f"Entities with UMLS CUI: {entities_df['umls_cui_mapped'].notna().sum():,}")
    print(f"Unique patients: {len(labels_df):,}")
    print(f"\nFiltering:")
    print(f"  Minimum frequency: {args.min_freq:.1%}")
    print(f"  Concepts after filtering: {feature_matrix.shape[1]:,}")
    print(f"\nFeature Selection ({args.method.upper()}):")
    print(f"  Top-K selected: {args.top_k}")
    print(f"  Final features: {selected_features.shape[1] - 2}")  # -2 for hadm_id and label
    print(f"\nOutput shape: {selected_features.shape}")
    print(f"  Rows (patients): {selected_features.shape[0]:,}")
    print(f"  Columns: {selected_features.shape[1]} (including hadm_id, label)")
    print(f"\nLabel distribution in output:")
    print(f"  {selected_features['label'].value_counts().to_dict()}")
    print(f"\nFiles saved:")
    print(f"  Features: {args.output}")
    if args.output_scores:
        print(f"  Scores: {args.output_scores}")
    
    # Show top features
    print(f"\nTop 30 most significant features ({args.method.upper()}):")
    print("-"*70)
    score_col = 'chi2_score' if args.method == 'chi2' else 'mi_score'
    for i, (_, row) in enumerate(feature_scores.head(30).iterrows(), 1):
        name = concept_info.get(row['feature'], {}).get('name', 'Unknown')
        freq = concept_info.get(row['feature'], {}).get('frequency', 0)
        print(f"{i:2d}. {row['feature']:<40} | {name:<30} | freq={freq:.2%} | score={row[score_col]:.2f}")


if __name__ == "__main__":
    main()

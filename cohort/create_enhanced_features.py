#!/usr/bin/env python3
"""
Create Enhanced Features by Combining:
1. Lab/Vital features from features_phase5.csv
2. Clinical note embeddings (384-dim) from discharge_embeddings_full.npz
3. Optional: Add variability features

Output: features_enhanced_v2.csv with rich NLP embeddings
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_npz: str, index_csv: str) -> pd.DataFrame:
    """Load clinical note embeddings and create DataFrame."""
    logger.info(f"Loading embeddings from {embeddings_npz}")
    
    # Load numpy embeddings
    data = np.load(embeddings_npz)
    X = data['X']
    logger.info(f"Embeddings shape: {X.shape}")
    
    # Load index mapping
    index_df = pd.read_csv(index_csv)
    logger.info(f"Index shape: {index_df.shape}")
    
    # Create DataFrame with embedding columns
    emb_cols = [f'emb_{i}' for i in range(X.shape[1])]
    emb_df = pd.DataFrame(X, columns=emb_cols)
    emb_df['HADM_ID'] = index_df['HADM_ID'].values
    
    return emb_df


def reduce_dimensions(emb_df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    """Apply PCA to reduce embedding dimensions."""
    logger.info(f"Reducing embeddings from {emb_df.shape[1]-1} to {n_components} dimensions")
    
    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    X = emb_df[emb_cols].values
    
    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create new DataFrame
    new_cols = [f'nlp_emb_{i}' for i in range(n_components)]
    reduced_df = pd.DataFrame(X_reduced, columns=new_cols)
    reduced_df['HADM_ID'] = emb_df['HADM_ID'].values
    
    return reduced_df


def add_variability_features(vitals_df: pd.DataFrame) -> pd.DataFrame:
    """Add variability features for numeric columns."""
    logger.info("Adding variability features")
    
    # Find columns that have both Mean and Std
    mean_cols = [c for c in vitals_df.columns if c.endswith('_Avg') or c.endswith('_Mean')]
    
    new_features = {}
    for col in mean_cols:
        base = col.replace('_Avg', '').replace('_Mean', '')
        std_col = f'{base}_Std'
        
        if std_col in vitals_df.columns:
            # Coefficient of Variation (CV)
            mean_vals = vitals_df[col]
            std_vals = vitals_df[std_col]
            # Avoid division by zero
            cv = np.where(mean_vals != 0, std_vals / np.abs(mean_vals), 0)
            new_features[f'{base}_CV'] = cv
    
    if new_features:
        for name, values in new_features.items():
            vitals_df[name] = values
        logger.info(f"Added {len(new_features)} variability features")
    
    return vitals_df


def main():
    parser = argparse.ArgumentParser(description='Create enhanced features')
    parser.add_argument('--vitals', default='features_phase5.csv', help='Vitals/labs features CSV')
    parser.add_argument('--embeddings-npz', default='note_embeddings/discharge_embeddings_full.npz')
    parser.add_argument('--embeddings-index', default='note_embeddings/index_embeddings_full.csv')
    parser.add_argument('--output', default='features_enhanced_v2.csv')
    parser.add_argument('--pca-components', type=int, default=0, help='PCA components (0=no reduction)')
    parser.add_argument('--add-variability', action='store_true', help='Add variability features')
    
    args = parser.parse_args()
    
    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load vital/lab features
    vitals_path = os.path.join(base_path, args.vitals)
    logger.info(f"Loading vitals from {vitals_path}")
    vitals_df = pd.read_csv(vitals_path)
    logger.info(f"Vitals shape: {vitals_df.shape}")
    
    # Add variability features
    if args.add_variability:
        vitals_df = add_variability_features(vitals_df)
    
    # Load embeddings
    emb_npz_path = os.path.join(base_path, args.embeddings_npz)
    emb_idx_path = os.path.join(base_path, args.embeddings_index)
    emb_df = load_embeddings(emb_npz_path, emb_idx_path)
    
    # Optionally reduce dimensions
    if args.pca_components > 0:
        emb_df = reduce_dimensions(emb_df, args.pca_components)
    else:
        # Rename columns to have nlp_ prefix
        rename_map = {c: f'nlp_{c}' for c in emb_df.columns if c.startswith('emb_')}
        emb_df = emb_df.rename(columns=rename_map)
    
    # Merge on HADM_ID
    logger.info("Merging vitals with embeddings")
    merged = vitals_df.merge(emb_df, on='HADM_ID', how='inner')
    logger.info(f"Merged shape: {merged.shape}")
    
    # Count feature types
    vital_cols = [c for c in merged.columns if not c.startswith('nlp_') and c != 'HADM_ID']
    nlp_cols = [c for c in merged.columns if c.startswith('nlp_')]
    logger.info(f"Vital/lab features: {len(vital_cols)}")
    logger.info(f"NLP embedding features: {len(nlp_cols)}")
    
    # Save
    output_path = os.path.join(base_path, args.output)
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced features to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("ENHANCED FEATURES SUMMARY")
    print("="*60)
    print(f"Total samples: {len(merged)}")
    print(f"Vital/Lab features: {len(vital_cols)}")
    print(f"NLP embedding features: {len(nlp_cols)}")
    print(f"Total features: {len(merged.columns) - 1}")
    print(f"Output: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()

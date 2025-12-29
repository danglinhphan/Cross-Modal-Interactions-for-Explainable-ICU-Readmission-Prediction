import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

def load_merged_data(vital_path, nlp_path, pathology_path, text_path, labels_path):
    print("Loading and merging data...")
    try:
        df_vital = pd.read_csv(vital_path)
        df_nlp = pd.read_csv(nlp_path)
        df_pathology = pd.read_csv(pathology_path)
        df_text = pd.read_csv(text_path)
        df_lbl = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None
        
    for df in [df_vital, df_nlp, df_pathology, df_text, df_lbl]:
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
            
    # Merge Sequence: Vital -> NLP -> Pathology -> Text -> Labels
    df_merged = df_vital.merge(df_nlp, on='HADM_ID', how='inner')
    df_merged = df_merged.merge(df_pathology, on='HADM_ID', how='left')
    df_merged = df_merged.merge(df_text, on='HADM_ID', how='left')
    df_merged = df_merged.fillna(-1)
    
    df_full = df_merged.merge(df_lbl, on='HADM_ID', how='inner')
    
    target_col = 'Y' if 'Y' in df_full.columns else 'LABEL'
    y = df_full[target_col]
    X = df_full.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    return X, y

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    
    # Load Data
    X, y = load_merged_data(
        os.path.join(base_dir, 'cohort/features_phase4_clinical.csv'),
        os.path.join(base_dir, 'cohort/nlp_features_enhanced.csv'),
        os.path.join(base_dir, 'cohort/new_pathology_features.csv'),
        os.path.join(base_dir, 'cohort/text_tfidf_features.csv'),
        os.path.join(base_dir, 'cohort/new_cohort_icu_readmission_labels.csv')
    )
    
    if X is None:
        return

    # Split (Same seed as training phase 8: 42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Load Model (Phase 8 Single)
    model_path = os.path.join(base_dir, 'outputs/ebm_single_interaction/ebm_single_model.pkl')
    print(f"Loading model from {model_path}...")
    
    try:
        with open(model_path, 'rb') as f:
             model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print("Generating predictions...")
    # Single EBM has predict_proba returning (N, 2)
    # Check shape
    y_proba = model.predict_proba(X_test)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    
    # PR Curve Analysis
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    print("\n--- Analysis (Phase 8 Single Model - Correct Seed) ---")
    
    # 1. Max F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    print(f"Max Possible F1: {f1_scores[best_idx]:.4f}")
    print(f"  at Threshold: {thresholds[best_idx] if best_idx < len(thresholds) else 1.0:.4f}")
    print(f"  Precision:    {precisions[best_idx]:.4f}")
    print(f"  Recall:       {recalls[best_idx]:.4f}")
    
    # 2. Feasibility of 0.85/0.85
    feasible_points = (precisions >= 0.85) & (recalls >= 0.85)
    if np.any(feasible_points):
        print("\n[YES] Target (0.85/0.85) IS spatially feasible!")
        indices = np.where(feasible_points)[0]
        # Sort by F1
        best_feasible_idx = indices[np.argmax(f1_scores[indices])]
        print(f"Best Feasible Configuration:")
        print(f"  Threshold: {thresholds[best_feasible_idx]:.4f}")
        print(f"  Precision: {precisions[best_feasible_idx]:.4f}")
        print(f"  Recall:    {recalls[best_feasible_idx]:.4f}")
        print(f"  F1:        {f1_scores[best_feasible_idx]:.4f}")
    else:
        print("\n[NO] Target (0.85/0.85) is NOT feasible with this model.")
        print("Closest points:")
        idx_r85 = np.argmin(np.abs(recalls - 0.85))
        print(f"  At Recall=0.85 -> Precision={precisions[idx_r85]:.4f} (Thresh: {thresholds[idx_r85]:.4f})")
        
        idx_p85 = np.argmin(np.abs(precisions - 0.85))
        print(f"  At Precision=0.85 -> Recall={recalls[idx_p85]:.4f} (Thresh: {thresholds[idx_p85]:.4f})")

if __name__ == "__main__":
    main()

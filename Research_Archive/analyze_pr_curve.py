import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

# Re-use loading logic to ensure same test set
def load_data(vital_path, nlp_path, labels_path):
    print("Loading data...")
    df_vital = pd.read_csv(vital_path)
    df_nlp = pd.read_csv(nlp_path)
    df_lbl = pd.read_csv(labels_path)
    
    for df in [df_vital, df_nlp, df_lbl]:
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
            
    df_merged = df_vital.merge(df_nlp, on='HADM_ID', how='inner')
    df_full = df_merged.merge(df_lbl, on='HADM_ID', how='inner')
    
    target_col = 'Y' if 'Y' in df_full.columns else 'LABEL'
    y = df_full[target_col]
    X = df_full.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    X = X.fillna(0)
    return X, y

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    
    # Load Data
    X, y = load_data(
        os.path.join(base_dir, 'cohort/features_phase4_clinical.csv'),
        os.path.join(base_dir, 'cohort/nlp_features_enhanced.csv'),
        os.path.join(base_dir, 'cohort/new_cohort_icu_readmission_labels.csv')
    )
    
    # Split (Same seed as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=170
    )
    
    # Load Model
    model_path = os.path.join(base_dir, 'outputs/ebm_balanced_interaction/ebm_ensemble_model.pkl')
    print(f"Loading model from {model_path}...")
    
    # Define class relative to EBM structure if needed, but since it was pickled from __main__, 
    # we need to make sure pickle can find it.
    class EBMEnsemble:
        def __init__(self, n_estimators=10, undersampling_ratio=1.0, interactions=50, random_state=42, n_jobs=1):
            self.n_estimators = n_estimators
            self.undersampling_ratio = undersampling_ratio
            self.interactions = interactions
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.models = []
        
        def predict_proba(self, X):
            probas = np.zeros(len(X))
            for model in self.models:
                probas += model.predict_proba(X)[:, 1]
            return probas / len(self.models)
        
        def fit(self, X, y):
             pass

    # Hack to allow unpickling if it was saved as __main__.EBMEnsemble
    import __main__
    setattr(__main__, 'EBMEnsemble', EBMEnsemble)

    try:
        with open(model_path, 'rb') as f:
             model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print("Generating predictions...")
    y_proba = model.predict_proba(X_test)
    
    # PR Curve Analysis
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    print("\n--- Analysis ---")
    
    # 1. Max F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    print(f"Max Possible F1: {f1_scores[best_idx]:.4f}")
    print(f"  at Threshold: {thresholds[best_idx] if best_idx < len(thresholds) else 1.0:.4f}")
    print(f"  Precision:    {precisions[best_idx]:.4f}")
    print(f"  Recall:       {recalls[best_idx]:.4f}")
    
    # 2. Feasibility of 0.80
    feasible_points = (precisions >= 0.80) & (recalls >= 0.80)
    if np.any(feasible_points):
        print("\n[YES] Target (0.8/0.8) IS spatially feasible!")
        indices = np.where(feasible_points)[0]
        best_feasible_idx = indices[np.argmax(f1_scores[indices])]
        print(f"Best Feasible Configuration:")
        print(f"  Threshold: {thresholds[best_feasible_idx]:.4f}")
        print(f"  Precision: {precisions[best_feasible_idx]:.4f}")
        print(f"  Recall:    {recalls[best_feasible_idx]:.4f}")
        print(f"  F1:        {f1_scores[best_feasible_idx]:.4f}")
    else:
        print("\n[NO] Target (0.8/0.8) is NOT feasible with this model.")
        print("Closest points:")
        # Check P at R=0.8
        idx_r80 = np.argmin(np.abs(recalls - 0.80))
        print(f"  At Recall=0.80 -> Precision={precisions[idx_r80]:.4f} (Thresh: {thresholds[idx_r80]:.4f})")
        
        # Check R at P=0.8
        idx_p80 = np.argmin(np.abs(precisions - 0.80))
        print(f"  At Precision=0.80 -> Recall={recalls[idx_p80]:.4f} (Thresh: {thresholds[idx_p80]:.4f})")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------
def load_merged_data():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    print("Loading merged data...")
    
    paths = {
        'vital': 'cohort/features_phase4_clinical.csv',
        'nlp': 'cohort/nlp_features_enhanced.csv',
        'path': 'cohort/new_pathology_features.csv',
        'text': 'cohort/text_tfidf_features.csv',
        'lbl': 'cohort/new_cohort_icu_readmission_labels.csv'
    }
    
    dfs = []
    for k, p in paths.items():
        full_p = os.path.join(base_dir, p)
        df = pd.read_csv(full_p)
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
        dfs.append((k, df))
        
    df_main = dfs[0][1] # Vital
    for k, df in dfs[1:]:
        how = 'inner' if k in ['nlp', 'lbl'] else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(-1)
    
    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

# -----------------------------------------
# OPTIMAL THRESHOLD determined by analysis
# Phase 8 Single Model (Seed 42 Validated)
# precision=0.8551, recall=0.7957, f1=0.8243
OPTIMAL_THRESHOLD = 0.3440
# -----------------------------------------

class ProductionEBM:
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
        # Copy necessary attributes for interpretability tools
        if hasattr(base_model, 'term_names_'):
            self.term_names_ = base_model.term_names_
        if hasattr(base_model, 'term_importances'):
            self.term_importances = base_model.term_importances
            
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
        
    def predict(self, X):
        probas = self.predict_proba(X)
        if probas.ndim == 2:
            probas = probas[:, 1]
        return (probas >= self.threshold).astype(int)

def main():
    # Load Best Model (Phase 8 Single)
    model_path = 'outputs/ebm_single_interaction/ebm_single_model.pkl'
    final_path = 'outputs/ebm_single_interaction/ebm_single_final.pkl'
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    print(f"Wrapping model with Threshold={OPTIMAL_THRESHOLD}...")
    prod_model = ProductionEBM(best_model, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(prod_model, f)
        
    print(f"Saved optimized model to {final_path}")
    
    # Validation
    X, y = load_merged_data()
    # VALIDATION SPLIT (Seed 42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = prod_model.predict(X_test)
    y_proba_full = prod_model.predict_proba(X_test)
    y_proba_pos = y_proba_full[:, 1] if y_proba_full.ndim == 2 else y_proba_full
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba_pos),
        'auprc': average_precision_score(y_test, y_proba_pos),
        'note': 'Final Phase 8 Single Optimal'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    
    with open('outputs/ebm_single_interaction/metrics_final.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

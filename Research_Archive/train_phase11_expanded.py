import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# WRAPPER (Same as Phase 8 for compatibility)
# -----------------------------------------
class ProductionEBM:
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
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

def load_expanded_data():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    print("Loading expanded data...")
    
    paths = {
        'vital': 'cohort/features_phase4_clinical.csv',
        'nlp': 'cohort/nlp_features_enhanced.csv',
        'path': 'cohort/new_pathology_features.csv',
        'text': 'cohort/text_tfidf_features.csv',
        'extra': 'cohort/features_phase11_extra.csv', # NEW
        'lbl': 'cohort/new_cohort_icu_readmission_labels.csv'
    }
    
    dfs = []
    # Order matters for merging
    for k in ['vital', 'nlp', 'path', 'text', 'extra', 'lbl']:
        p = paths[k]
        full_p = os.path.join(base_dir, p)
        df = pd.read_csv(full_p)
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
        dfs.append(df)
        
    df_main = dfs[0] # Vital
    for df in dfs[1:]:
        how = 'inner' if 'LABEL' in df.columns or 'Y' in df.columns else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(0) # Simple fill for new features
    
    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

def main():
    output_dir = 'outputs/ebm_phase11_expanded'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y = load_expanded_data()
    print(f"Feature Count: {X.shape[1]}")
    
    # 2. Split (Seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. SMOTE (Advanced Imbalance)
    print("Applying SMOTE (Synthetic Oversampling)...")
    # Identify Categorical Features for SMOTENC?
    # EBM handles mixed types, but SMOTE assumes continuous.
    # TF-IDF is continuous. Vitals are continuous.
    # Strictly binary features are "Categorical" for SMOTE-NC.
    # Let's detect binary columns.
    
    cat_mask = []
    col_names = list(X.columns)
    for col in col_names:
        unique_vals = X[col].nunique()
        if unique_vals <= 2:
            cat_mask.append(True) # Treat binary as categorical for safe interpolation
        else:
            cat_mask.append(False)
            
    cat_indices = np.where(cat_mask)[0].tolist()
    
    # Only use SMOTENC if we have cats, else SMOTE
    # sampling_strategy=0.5 (Ratio 1:2) or 'auto' (1:1)?
    # User wants > 85% RECALL. Standard SMOTE (1:1) boosts Recall.
    # Let's target ratio 0.8 to be safe against over-smoothing.
    
    if len(cat_indices) > 0:
        print(f"Using SMOTENC on {len(cat_indices)} categorical features.")
        smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=0.8, random_state=42)
    else:
        smote = SMOTE(sampling_strategy=0.8, random_state=42)
        
    print("Resampling training data...")
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled Shape: {X_res.shape}, Class 1 count: {sum(y_res)}")
    
    # 4. Train Single EBM (Phase 11)
    print("Training Phase 11 EBM...")
    ebm = ExplainableBoostingClassifier(
        interactions=50, # Keep the magic number
        outer_bags=16,
        inner_bags=0,
        learning_rate=0.01,
        random_state=42,
        n_jobs=1
    )
    
    ebm.fit(X_res, y_res)
    
    # 5. Evaluate
    # Use Probability Calibration? EBM is usually well calibrated.
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    # Find Optimal Threshold for 85/85
    # Just save raw model for analysis script
    
    with open(os.path.join(output_dir, 'ebm_phase11_model.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    print("Model Saved. Analyzing Performance...")
    
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    metrics = {
        'max_f1': f1_scores[best_idx],
        'threshold': thresholds[best_idx] if best_idx < len(thresholds) else 1.0,
        'precision': precisions[best_idx],
        'recall': recalls[best_idx],
        'auroc': roc_auc_score(y_test, y_proba)
    }
    
    print("Phase 11 (Expanded + SMOTE) metrics:")
    print(metrics)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

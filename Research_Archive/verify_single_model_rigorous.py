import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

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

def main():
    # 1. Load Data
    X, y = load_merged_data()
    
    # 2. Re-create Split (Verify seed usage from Phase 8)
    # Phase 8 used random_state=42
    print("Splitting data with random_state=42 (Same as Phase 8 inference)...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Test Set Size: {len(y_test)}")
    print(f"Class Distribution in Test:\n{y_test.value_counts()}")
    
    # 3. Load Model
    model_path = 'outputs/ebm_phase10_ensemble/ebm_phase10_model.pkl'
    print(f"Loading model: {model_path}")
    
    # Define class relative to EBM structure if needed
    class EBMEnsemblePhase10:
        def __init__(self, n_estimators=20, interactions=None, random_state=42):
            self.n_estimators = n_estimators
            self.interactions = interactions # List of indices [[1, 2], [3, 4]]
            self.random_state = random_state
        
        def predict_proba(self, X):
            probas = np.zeros(len(X))
            for model in self.models:
                probas += model.predict_proba(X)[:, 1]
            return probas / len(self.models)
        
        def predict(self, X):
             # Default 0.5 threshold for initial check
             return (self.predict_proba(X) >= 0.5).astype(int)
            
    # Hack to allow unpickling if it was saved as __main__.EBMEnsemble
    import __main__
    setattr(__main__, 'EBMEnsemblePhase10', EBMEnsemblePhase10)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 4. Predict
    print("Running predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 5. Rigorous Evaluation
    print("\n" + "="*50)
    print("RIGOROUS VERIFICATION REPORT")
    print("="*50)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n1. Confusion Matrix:")
    print(f"      Pred 0    Pred 1")
    print(f"Act 0   {tn:<8}  {fp:<8}")
    print(f"Act 1   {fn:<8}  {tp:<8}")
    
    print(f"\n- True Negatives (Correct Safe): {tn}")
    print(f"- False Positives (False Alarm): {fp}")
    print(f"- False Negatives (Missed Risk): {fn}")
    print(f"- True Positives (Correct Risk): {tp}")
    
    # Classification Report (Per Class)
    print("\n2. Classification Report (Per Class):")
    print(classification_report(y_test, y_pred, target_names=['Class 0 (No Readmit)', 'Class 1 (Readmit)'], digits=4))
    
    # Explicit Calculation for Class 1
    print("\n3. Minority Class (Class 1) Metrics Explicitly:")
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0
    
    print(f"   Precision (Class 1): {p1:.4f}")
    print(f"   Recall (Class 1):    {r1:.4f}")
    print(f"   F1 Score (Class 1):  {f1_1:.4f}")
    
    # Compare with sklearn default
    sk_f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"   (Sklearn f1_score check: {sk_f1:.4f})")
    
    # AUROC
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n4. AUROC: {auc:.4f}")

if __name__ == "__main__":
    main()

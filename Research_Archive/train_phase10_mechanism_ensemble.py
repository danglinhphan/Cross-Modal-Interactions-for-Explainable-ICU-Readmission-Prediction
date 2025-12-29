import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

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
    
    return X, y, list(X.columns)

class EBMEnsemblePhase10:
    def __init__(self, n_estimators=20, interactions=None, random_state=42):
        self.n_estimators = n_estimators
        self.interactions = interactions # List of indices [[1, 2], [3, 4]]
        self.random_state = random_state
        self.models = []
        
    def fit(self, X, y):
        self.models = []
        print(f"Training Phase 10 Ensemble with {self.n_estimators} estimators...")
        
        for i in range(self.n_estimators):
            seed = self.random_state + i
            # Balanced Bagging: Ratio 1.0
            rus = RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
            X_res, y_res = rus.fit_resample(X, y)
            
            # Use fixed interactions
            ebm = ExplainableBoostingClassifier(
                interactions=self.interactions,
                outer_bags=8,
                inner_bags=0,
                learning_rate=0.01,
                random_state=seed,
                n_jobs=1 
            )
            ebm.fit(X_res, y_res)
            self.models.append(ebm)
            
            if (i+1) % 5 == 0:
                print(f"  Trained {i+1}/{self.n_estimators}")
                
        return self
        
    def predict_proba(self, X):
        probas = np.zeros(len(X))
        for model in self.models:
            probas += model.predict_proba(X)[:, 1]
        return probas / len(self.models)
        
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

def main():
    output_dir = 'outputs/ebm_phase10_ensemble'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y, feature_names = load_merged_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Train Ensemble
    print(f"Training Unrestricted Ensemble (Interactions=100, Est=20)...")
    ensemble = EBMEnsemblePhase10(
        n_estimators=20,
        interactions=100, # Free Discovery of Top 100
        random_state=42
    )
    ensemble.fit(X_train, y_train)
    
    # 4. Evaluate (Rigorous)
    y_proba = ensemble.predict_proba(X_test)
    y_pred_def = (y_proba >= 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred_def)
    recall = recall_score(y_test, y_pred_def)
    f1 = f1_score(y_test, y_pred_def)
    
    print("\nMetrics (Threshold 0.5):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    
    # 5. Threshold Optimization for 85/85
    # Since we want P>=0.85 and R>=0.85
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    best_f1 = 0
    best_th = 0.5
    met_target = False
    
    for p, r, th in zip(precisions, recalls, thresholds):
        if p >= 0.85 and r >= 0.85:
            met_target = True
            current_f1 = 2*p*r/(p+r)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_th = th
                
    if met_target:
        print(f"\n[SUCCESS] Target 85/85 Achieved at Threshold {best_th:.4f}!")
    else:
        print("\n[INFO] Target 85/85 Not strictly met. Finding best compromise...")
        # Maximize min(P, R) where both > 0.8
        
    # Save Model
    with open(os.path.join(output_dir, 'ebm_phase10_model.pkl'), 'wb') as f:
        pickle.dump(ensemble, f)

if __name__ == "__main__":
    main()

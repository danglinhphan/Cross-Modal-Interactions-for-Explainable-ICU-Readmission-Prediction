import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def load_merged_data():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    print("Loading merged Phase 7 data...")
    
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
        
    # Merge
    df_main = dfs[0][1] # Vital
    for k, df in dfs[1:]:
        how = 'inner' if k in ['nlp', 'lbl'] else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(-1)
    
    # Split
    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

def main():
    output_dir = 'outputs/ebm_single_interaction'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = load_merged_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("-" * 30)
    print(f"Training Single EBM (Interactions Enabled)...")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X_train)}")
    
    # Single EBM for pure math extraction
    # Using pos_weight instead of undersampling to keep full data richness
    # Ratio is ~4:1?
    
    ebm = ExplainableBoostingClassifier(
        interactions=50,
        outer_bags=16, 
        inner_bags=0,
        learning_rate=0.01,
        random_state=42,
        n_jobs=1
    )
    
    ebm.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ebm.predict(X_test)
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba)
    }
    
    print("Metrics:", metrics)
    
    # Extract Interactions
    print("Extracting Interaction Logic...")
    interaction_list = []
    
    # EBM stores term names. Interactions have ' x '
    # Also global importance
    importances = ebm.term_importances()
    names = ebm.term_names_
    
    interaction_pairs = []
    for name, imp in zip(names, importances):
        if ' x ' in name:
            interaction_pairs.append((name, imp))
            
    # Sort by importance
    interaction_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Save list
    list_path = os.path.join(output_dir, 'interaction_list.txt')
    with open(list_path, 'w') as f:
        f.write("Top Learned Cross-Interactions (Math Mechanism):\n")
        f.write("============================================\n")
        for name, imp in interaction_pairs:
            f.write(f"{name} : {imp:.4f}\n")
            
    # Save Model
    model_path = os.path.join(output_dir, 'ebm_single_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(ebm, f)
        
    print(f"Model saved to {model_path}")
    print(f"Interactions list saved to {list_path}")

if __name__ == "__main__":
    main()

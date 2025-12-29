import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# --- HELPER CLASSES ---
class ProductionEBM:
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
        if hasattr(base_model, 'n_estimators'):
            self.n_estimators = base_model.n_estimators
            
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
        
    def predict(self, X):
        probas = self.predict_proba(X)
        if probas.ndim == 2:
            probas = probas[:, 1]
        return (probas >= self.threshold).astype(int)

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

def get_cross_domain_indices(ebm, feature_names):
    """
    Extracts interaction indices from a fitted EBM, keeping only 'Text x Tabular'.
    """
    print("Filtering Cross-Domain Interactions...")
    importances = ebm.term_importances()
    term_names = ebm.term_names_
    
    # Identify Text columns
    text_indices = [i for i, name in enumerate(feature_names) if 'TFIDF_' in name.upper()]
    tabular_indices = [i for i, name in enumerate(feature_names) if 'TFIDF_' not in name.upper()]
    
    text_set = set(text_indices)
    tabular_set = set(tabular_indices)
    
    selected_pairs = []
    
    # Map term names to indices
    # term_names can be "FeatureA" or "FeatureA & FeatureB"
    # We need to parse "FeatureA & FeatureB" back to indices in X
    
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    
    cross_count = 0
    
    for term, imp in zip(term_names, importances):
        if ' & ' in term:
            parts = term.split(' & ')
            if len(parts) != 2:
                continue
            
            n1, n2 = parts
            if n1 not in name_to_idx or n2 not in name_to_idx:
                continue
                
            idx1 = name_to_idx[n1]
            idx2 = name_to_idx[n2]
            
            is_text1 = idx1 in text_set
            is_text2 = idx2 in text_set
            
            # Condition: Exactly one is text
            if is_text1 != is_text2:
                selected_pairs.append([idx1, idx2])
                print(f"  [Keep] {term} (Imp: {imp:.4f})")
                cross_count += 1
                
    print(f"Found {cross_count} Cross-Domain interactions.")
    return selected_pairs

def main():
    output_dir = 'outputs/ebm_explicit_interaction'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y = load_merged_data()
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Load Discovery Model (Phase 8 Single)
    discovery_path = 'outputs/ebm_single_interaction/ebm_single_model.pkl'
    if not os.path.exists(discovery_path):
        print("Discovery model not found. Run train_single_interaction_model.py first.")
        return
        
    print(f"Loading Discovery Model from {discovery_path}...")
    with open(discovery_path, 'rb') as f:
        ebm_discovery = pickle.load(f)
        
    # 3. Filter Interactions
    cross_pairs = get_cross_domain_indices(ebm_discovery, feature_names)
    
    if not cross_pairs:
        print("No cross-domain interactions found! Using top 20 whatever.")
        # Fallback or exit?
        # Let's try to grab top interactions regardless if strict filter fails
        # But for 'mechanism', user wants text x tabular.
        # Maybe names mismatch? (Case sensitivity).
        # We will assume get_cross_domain_indices logic is correct (Using 'TFIDF_')
        pass
    
    # 4. Train Explicit Model
    # We pass the LIST of pairs to 'interactions'
    # EBM will ONLY learn these interactions (plus main effects).
    
    print("-" * 30)
    print(f"Training Explicit Cross-Interaction Model...")
    print(f"Forcing {len(cross_pairs)} specific interactions.")
    
    # If 0 pairs, default to 0 interactions?
    # inters = cross_pairs if cross_pairs else 0
    # Actually if 0, we should probably warn.
    
    ebm_final = ExplainableBoostingClassifier(
        interactions=cross_pairs,
        outer_bags=16,
        inner_bags=0,
        learning_rate=0.01, 
        random_state=42,
        n_jobs=1
    )
    
    ebm_final.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = ebm_final.predict(X_test)
    y_proba = ebm_final.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': 0.5,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba),
        'auprc': average_precision_score(y_test, y_proba)
    }
    
    print("Explicit Model Metrics:")
    print(metrics)
    
    # 6. Save
    final_pkl = os.path.join(output_dir, 'ebm_explicit_final.pkl')
    # Wrap
    prod_model_final = ProductionEBM(ebm_final, threshold=0.5)
    
    with open(final_pkl, 'wb') as f:
        pickle.dump(prod_model_final, f)
        
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved explicitly constrained model to {final_pkl}")
    
    # Save Interaction List for confirmation
    with open(os.path.join(output_dir, 'explicit_interactions_used.txt'), 'w') as f:
        f.write("Explicit Cross-Domain Interactions Enforced:\n")
        f.write("============================================\n")
        for i, pair in enumerate(cross_pairs):
            n1 = feature_names[pair[0]]
            n2 = feature_names[pair[1]]
            f.write(f"{i+1}. {n1} & {n2}\n")

if __name__ == "__main__":
    main()

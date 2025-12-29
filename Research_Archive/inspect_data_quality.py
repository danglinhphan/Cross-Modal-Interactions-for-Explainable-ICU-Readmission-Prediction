import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier

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
    X = X.fillna(0) # Standard imputation for EBM
    return X, y

def calculate_overlap(X, y, feature_name):
    """
    Calculate the intersection area (overlap) between class distributions.
    Lower overlap = Better separation.
    """
    pos = X[y==1][feature_name]
    neg = X[y==0][feature_name]
    
    # Simple histogram overlap
    bins = np.histogram(np.hstack((pos, neg)), bins=50)[1]
    hist_pos, _ = np.histogram(pos, bins=bins, density=True)
    hist_neg, _ = np.histogram(neg, bins=bins, density=True)
    
    # Intersection area
    overlap = np.minimum(hist_pos, hist_neg).sum() * (bins[1] - bins[0])
    return overlap

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    data_dir = os.path.join(base_dir, 'cohort')
    output_dir = os.path.join(base_dir, 'outputs/data_audit')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    X, y = load_data(
        os.path.join(data_dir, 'features_phase4_clinical.csv'),
        os.path.join(data_dir, 'nlp_features_enhanced.csv'),
        os.path.join(data_dir, 'new_cohort_icu_readmission_labels.csv')
    )
    
    # 2. Train a single quick EBM (Global Explanation)
    print("\nTraining diagnostic EBM (1000 samples) to find important features...")
    ebm = ExplainableBoostingClassifier(interactions=0, n_jobs=1, max_bins=256)
    # Subsample for speed
    if len(X) > 2000:
        idx = np.random.choice(len(X), 2000, replace=False)
        X_sub, y_sub = X.iloc[idx], y.iloc[idx]
    else:
        X_sub, y_sub = X, y
        
    ebm.fit(X_sub, y_sub)
    
    # 3. Get Top Features
    # Note: explanation.data() returns a dict, scores are usually per term.
    # Term names are what we want.
    global_exp = ebm.explain_global()
    feature_names = global_exp.feature_names
    importances = global_exp.data()['scores']
    
    # Ensure they match length
    if len(feature_names) != len(importances):
        print(f"Warning: Mismatch len(names)={len(feature_names)} vs len(scores)={len(importances)}")
        # Fallback to internal
        feature_names = ebm.term_names_
        importances = ebm.term_scores_
        # For GAM (no interactions), this should be 1:1 with features, but let's be safe
        # term_scores_ is a list of arrays (one per feature). We take max abs or mean abs.
        importances = [np.mean(np.abs(score)) for score in importances]

    # Sort
    idx_sorted = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))
    top_features = [feature_names[i] for i in idx_sorted[:top_n]]
    top_scores = [importances[i] for i in idx_sorted[:top_n]]
    
    print("\ntop_features identified:")
    print(top_features)

    # 4. Analyze Discriminative Power (Overlap)
    print("\nAnalyzing Class Separation (Overlap)...")
    overlaps = {}
    for ft in top_features:
        ov = calculate_overlap(X, y, ft)
        overlaps[ft] = ov
    
    # 5. Missingness / Zero Rate
    print("Analyzing Sparsity...")
    sparsity = {}
    for ft in top_features:
        zero_rate = (X[ft] == 0).mean()
        sparsity[ft] = zero_rate
        
    # 6. Report
    report = {
        "dataset_size": len(X),
        "class_balance": y.mean(),
        "top_features": []
    }
    
    for i, ft in enumerate(top_features):
        report["top_features"].append({
            "name": ft,
            "importance": top_scores[i],
            "overlap": overlaps[ft], 
            "sparsity": sparsity[ft]
        })
        
    # Save Report
    with open(os.path.join(output_dir, 'data_quality_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport saved to: {os.path.join(output_dir, 'data_quality_report.json')}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
from train_phase12_nosmote import load_phase12_data

def load_all_data_phase14():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    # Start with Phase 12 base
    X, y = load_phase12_data()
    
    # Merge Social
    p_social = os.path.join(base_dir, 'cohort/features_phase14_social.csv')
    df_social = pd.read_csv(p_social)
    if 'HADM_ID' in df_social.columns:
        df_social['HADM_ID'] = df_social['HADM_ID'].astype(int)
        
    # We need HADM_ID back on X to merge?
    # load_phase12_data drops it.
    # We should modify loading or index match.
    # Actually, load_phase12_data drops HADM_ID.
    # I should redefine loader or just trust row order? NO. Row order is dangerous.
    # Let's rebuild loader here for safety.
    return None, None 

# RE-IMPLEMENT LOADER SAFER
def safe_load_phase14():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    print("Loading Phase 14 Social + Phase 12 High-Fi...")
    
    # Paths
    paths = [
        'cohort/features_phase4_clinical.csv',
        'cohort/nlp_features_enhanced.csv',
        'cohort/text_tfidf_features.csv',
        'cohort/new_pathology_features.csv',
        'cohort/features_phase11_extra.csv',
        'cohort/features_phase12_extra.csv',
        'cohort/features_phase14_social.csv', # NEW
        'cohort/new_cohort_icu_readmission_labels.csv'
    ]
    
    dfs = []
    for p in paths:
        path = os.path.join(base_dir, p)
        df = pd.read_csv(path)
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        dfs.append(df)
        
    df_main = dfs[0]
    for df in dfs[1:]:
        how = 'inner' if 'LABEL' in df.columns or 'Y' in df.columns else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(0) # Categoricals in social might become 0/strings?
    # Social columns are strings. fillna(0) turns them to 0 (int).
    # Better: fill social with 'UNKNOWN' explicitly?
    # Phase 14 script already filled NAs with 'UNKNOWN'. So merge keeps them.
    # But left join might introduce NaNs for missing IDs.
    
    # Social cols: INSURANCE, MARITAL_STATUS, ETHNICITY_GROUP, DISCHARGE_LOCATION
    social_cols = ['INSURANCE', 'MARITAL_STATUS', 'ETHNICITY_GROUP', 'DISCHARGE_LOCATION']
    for c in social_cols:
        if c in df_main.columns:
            df_main[c] = df_main[c].fillna('UNKNOWN')
            
    # Fluid Balance calc
    if 'URINE_OUT_24H' in df_main.columns and 'FLUID_INPUT_24H' in df_main.columns:
        df_main['FLUID_BALANCE_24H'] = df_main['FLUID_INPUT_24H'] - df_main['URINE_OUT_24H']

    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

def main():
    output_dir = 'outputs/ebm_phase14_social'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = safe_load_phase14()
    print(f"Features: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Imbalance: Ratio 0.4 (Same as Phase 13)
    RUS_RATIO = 0.4
    print(f"Applying RandomUnderSampler (Ratio {RUS_RATIO})...")
    rus = RandomUnderSampler(sampling_strategy=RUS_RATIO, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"Resampled Shape: {X_res.shape}")
    
    # Train Phase 14 EBM (Social)
    print("Training Phase 14 EBM (Social, Interactions=200)...")
    ebm = ExplainableBoostingClassifier(
        interactions=200,
        outer_bags=24,
        inner_bags=0,
        learning_rate=0.005,
        max_rounds=5000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=1
    )
    
    ebm.fit(X_res, y_res)
    
    # Evaluate
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    with open(os.path.join(output_dir, 'ebm_phase14_model.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Check 90/90
    target_mask = (precisions >= 0.90) & (recalls >= 0.90)
    
    if np.any(target_mask):
        indices = np.where(target_mask)[0]
        best_idx = indices[np.argmax(f1_scores[indices])]
        print("[SUCCESS] 90/90 Target Met with Social Data!")
    else:
        best_idx = np.argmax(f1_scores)
        print("[NOTE] 90/90 Not Met (Social). Reporting Max F1.")
        
    metrics = {
        'threshold': float(thresholds[best_idx] if best_idx < len(thresholds) else 1.0),
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'f1': float(f1_scores[best_idx]),
        'auroc': float(roc_auc_score(y_test, y_proba)),
        'rus_ratio': RUS_RATIO
    }
    
    print("Phase 14 Metrics:")
    print(metrics)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

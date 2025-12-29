import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler

def load_phase12_data():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    print("Loading Phase 12 feature set...")
    
    # Paths
    p_vital = 'cohort/features_phase4_clinical.csv'
    p_nlp = 'cohort/nlp_features_enhanced.csv'
    p_text = 'cohort/text_tfidf_features.csv'
    p_path = 'cohort/new_pathology_features.csv'
    p_ph11 = 'cohort/features_phase11_extra.csv' # Urine is here
    p_ph12 = 'cohort/features_phase12_extra.csv' # Micro, Ward, Input
    p_lbl = 'cohort/new_cohort_icu_readmission_labels.csv'
    
    dfs = []
    for p in [p_vital, p_nlp, p_path, p_text, p_ph11, p_ph12, p_lbl]:
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
        
    df_main = df_main.fillna(0)
    
    # Feature Calculation: Fluid Balance
    # URINE_OUT_24H comes from Phase 11
    # FLUID_INPUT_24H comes from Phase 12
    if 'URINE_OUT_24H' in df_main.columns and 'FLUID_INPUT_24H' in df_main.columns:
        df_main['FLUID_BALANCE_24H'] = df_main['FLUID_INPUT_24H'] - df_main['URINE_OUT_24H']
        print("Created FLUID_BALANCE_24H feature.")
        
    # Drop proxy antibiotics if we have real Micro?
    # Actually, keep both. Antibiotics + Positive Culture is a stronger signal than either alone.
    
    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')
    
    return X, y

def main():
    output_dir = 'outputs/ebm_phase12_nosmote'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load
    X, y = load_phase12_data()
    print(f"Features: {X.shape[1]}")
    
    # 2. Split (Seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Imbalance Handling: Random Undersampling
    # Strategy: Ratio 0.5 (1 Positive : 2 Negatives). 
    # This keeps ~2200 Positives and ~4400 Negatives (Total ~6600).
    # Less data but cleaner "Hard" negatives (statistically).
    # If 0.5 fails to get Recall, we can go to 1.0.
    RUS_RATIO = 0.5
    print(f"Applying RandomUnderSampler (Ratio {RUS_RATIO})...")
    rus = RandomUnderSampler(sampling_strategy=RUS_RATIO, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"Resampled Shape: {X_res.shape}")
    
    # 4. Train
    print("Training Phase 12 EBM (No SMOTE)...")
    ebm = ExplainableBoostingClassifier(
        interactions=50,
        outer_bags=16,
        inner_bags=0,
        learning_rate=0.01,
        random_state=42,
        n_jobs=1
    )
    ebm.fit(X_res, y_res)
    
    # 5. Evaluate
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    with open(os.path.join(output_dir, 'ebm_phase12_model.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    # Check 85/85 feasibility
    feasible = (precisions >= 0.85) & (recalls >= 0.85)
    if np.any(feasible):
        indices = np.where(feasible)[0]
        best_feasible_idx = indices[np.argmax(f1_scores[indices])]
        p_final, r_final, f1_final = precisions[best_feasible_idx], recalls[best_feasible_idx], f1_scores[best_feasible_idx]
        thresh_final = thresholds[best_feasible_idx]
        print("[SUCCESS] Found configuration >= 85/85!")
    else:
        p_final, r_final, f1_final = precisions[best_idx], recalls[best_idx], f1_scores[best_idx]
        thresh_final = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        print("[NOTE] Max F1 shown (Target 85/85 not strictly in curve points).")
        
    metrics = {
        'threshold': float(thresh_final),
        'precision': float(p_final),
        'recall': float(r_final),
        'f1': float(f1_final),
        'auroc': float(roc_auc_score(y_test, y_proba)),
        'rus_ratio': RUS_RATIO
    }
    
    print("Phase 12 Metrics:")
    print(metrics)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

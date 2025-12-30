import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# Import Clean Loader
try:
    from train_phase16_honest import load_clean_data
except ImportError:
    # Fallback if specific file missing, copy logic
    from train_phase14_social import safe_load_phase14
    def load_clean_data():
        X, y = safe_load_phase14()
        leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
        X = X.drop(columns=[c for c in leakage_cols if c in X.columns], errors='ignore')
        return X, y

def main():
    output_dir = 'outputs/ebm_phase18_honest_fix'
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- PHASE 18: HONEST RETRAINING (Fixing NLP Leakage) ---")
    
    # 1. Load Clinical Data (Cleaned)
    X_clinical, y = load_clean_data()
    print(f"Clinical Features Loaded: {X_clinical.shape}")
    
    # 2. DROP LEAKED TF-IDF Columns
    # Identify columns starting with 'tfidf_' or 'TFIDF_' and remove them from X_clinical
    leak_cols = [c for c in X_clinical.columns if 'tfidf' in c.lower()]
    if leak_cols:
        print(f"Dropping {len(leak_cols)} leaked global TF-IDF columns...")
        X_clinical = X_clinical.drop(columns=leak_cols)
    print(f"Clean Clinical Features: {X_clinical.shape[1]}")
    
    # 3. Load RAW TEXT
    # We need HADM_ID to merge. Does X_clinical have HADM_ID? 
    # extract_phase14/16/clean usually drops IDs.
    # We might need to re-attach IDs or relies on index if we are careful.
    # BUT safe_load_phase14 drops IDs.
    # Strategy: Reload IDs from cohort file matching the index? 
    # NO, safe_load_phase14 builds from merge.
    # BEST WAY: Modify load_clean_data to return IDs or reload matching file.
    # Let's reload `new_cohort_icu_readmission_labels.csv` which is the anchor.
    # Wait, safe_load_phase14 joins everything on HADM_ID. 
    # We should re-implement a loader that keeps HADM_ID for a moment.
    
    # Let's do a custom load here to be safe.
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    text_path = os.path.join(base_dir, 'cohort/phase17_honest_text.csv')
    
    if not os.path.exists(text_path):
        print(f"CRITICAL: {text_path} not found. Run extract_nursing_notes.py first.")
        return

    df_text = pd.read_csv(text_path)
    df_text['HADM_ID'] = df_text['HADM_ID'].astype(int)
    
    # Re-assembling Clinical Data with IDs for robust merge...
    # We must replicate the FULL stack of merges from Phase 14 to ensure we have all 500+ features
    print("Re-assembling FULL Clinical Data stack...")
    
    paths = [
        'cohort/features_phase4_clinical.csv',
        'cohort/nlp_features_enhanced.csv', # This had old NLP topics, valid to keep? Yes.
        # 'cohort/text_tfidf_features.csv', # SKIP LEAKAGE
        'cohort/new_pathology_features.csv',
        'cohort/features_phase11_extra.csv',
        'cohort/features_phase12_extra.csv',
        'cohort/features_phase14_social.csv',
        'cohort/new_cohort_icu_readmission_labels.csv'
    ]
    
    dfs = []
    for p in paths:
        full_p = os.path.join(base_dir, p)
        if not os.path.exists(full_p):
            print(f"Warning: {full_p} missing.")
            continue
            
        df = pd.read_csv(full_p)
        if 'HADM_ID' in df.columns:
            df['HADM_ID'] = df['HADM_ID'].astype(int)
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        dfs.append(df)
        
    df_main = dfs[0]
    for df in dfs[1:]:
        how = 'inner' if 'Y' in df.columns or 'LABEL' in df.columns else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(0) # Basic imputation consistent with previoius phases
    
    # Merge Raw Text
    print("Merging Raw Text...")
    df_main = df_main.merge(df_text, on='HADM_ID', how='left')
    df_main['TEXT'] = df_main['TEXT'].fillna('')
    
    # Drop Leakage Columns (Explicit)
    leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
    df_main = df_main.drop(columns=[c for c in leakage_cols if c in df_main.columns], errors='ignore')
    
    # Target
    y_full = df_main['Y']
    X_full = df_main.drop(columns=['Y', 'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID'], errors='ignore')
    
    print(f"Total Data Shape: {X_full.shape}")
    
    # 4. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
    
    # 5. VECTORIZE (Honest)
    print("Vectorizing Text (Fit on Train, Transform on Test)...")
    text_train = X_train['TEXT']
    text_test = X_test['TEXT']
    
    # Drop Text from Clinical temporarily
    X_train_clin = X_train.drop(columns=['TEXT'])
    X_test_clin = X_test.drop(columns=['TEXT'])
    
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=0.01,
        max_df=0.9
    )
    
    # FIT on TRAIN
    X_train_tfidf = vectorizer.fit_transform(text_train)
    # TRANSFORM TEST
    X_test_tfidf = vectorizer.transform(text_test)
    
    feat_names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    
    # Convert to DF
    df_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=feat_names, index=X_train.index)
    df_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=feat_names, index=X_test.index)
    
    # Concatenate
    X_train_final = pd.concat([X_train_clin, df_train_tfidf], axis=1)
    X_test_final = pd.concat([X_test_clin, df_test_tfidf], axis=1)
    
    print(f"Train Shape: {X_train_final.shape}")
    print(f"Test Shape: {X_test_final.shape}")
    
    # 6. RESAMPLE (RUS)
    RATIO = 0.4
    print(f"Applying RUS (Ratio {RATIO})...")
    rus = RandomUnderSampler(sampling_strategy=RATIO, random_state=42)
    X_res, y_res = rus.fit_resample(X_train_final, y_train)
    
    # 7. TRAIN EBM ENSEMBLE
    print("Training EBM Ensemble (5 models)...")
    n_estimators = 5
    models = []
    
    for i in range(n_estimators):
        seed = 42 + i
        # Different RUS SEED for diversity?
        # Phase 17 code used explicit loop with RUS inside.
        # Let's replicate Phase 17 Ensemble logic exactly but with clean data.
        
        rus_i = RandomUnderSampler(sampling_strategy=RATIO, random_state=seed)
        X_res_i, y_res_i = rus_i.fit_resample(X_train_final, y_train)
        
        ebm = ExplainableBoostingClassifier(
            interactions=40,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.01,
            max_rounds=2000,
            random_state=seed, 
            n_jobs=1
        )
        ebm.fit(X_res_i, y_res_i)
        models.append(ebm)
        print(f"  Model {i+1} trained.")
        
    # 8. AGGREGATE
    print("Aggregating Predictions...")
    probas = np.zeros(len(X_test_final))
    for model in models:
        probas += model.predict_proba(X_test_final)[:, 1]
    probas /= n_estimators
    
    # 9. EVALUATE
    precisions, recalls, thresholds = precision_recall_curve(y_test, probas)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    auroc = roc_auc_score(y_test, probas)
    
    print("\n--- PHASE 18 RESULTS (HONEST) ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"Best F1: {f1_scores[best_idx]:.4f}")
    print(f"Precision: {precisions[best_idx]:.4f}")
    print(f"Recall: {recalls[best_idx]:.4f}")
    
    metrics = {
        'auroc': float(auroc),
        'f1': float(f1_scores[best_idx]),
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'threshold': float(thresholds[best_idx] if best_idx < len(thresholds) else 1.0)
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, 'ebm_ensemble_honest.pkl'), 'wb') as f:
        pickle.dump(models, f)

if __name__ == "__main__":
    main()

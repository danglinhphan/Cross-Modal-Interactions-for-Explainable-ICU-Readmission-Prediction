
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

def load_clean_data():
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
        'cohort/features_phase14_social.csv',
        'cohort/new_cohort_icu_readmission_labels.csv'
    ]
    
    dfs = []
    for p in paths:
        path = os.path.join(base_dir, p)
        if not os.path.exists(path):
            continue
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
    
    target_col = 'Y' if 'Y' in df_main.columns else 'LABEL'
    y = df_main[target_col]
    X = df_main.drop(columns=['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', target_col], errors='ignore')

    leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
    X = X.drop(columns=[c for c in leakage_cols if c in X.columns], errors='ignore')
    return X, y


def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    model_path = 'outputs/ebm_phase19_strict/ebm_ensemble_strict.pkl'
    metrics_path = 'outputs/ebm_phase19_strict/metrics.json'
    
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        print("Model or Metrics not found.")
        return
        
    # Load Threshold
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        threshold = metrics['threshold']
    print(f"Loaded Optimal Threshold: {threshold}")

    # --- DATA LOADING (Must match train_phase19_strict.py EXACTLY) ---
    # 1. Load Raw Text
    text_path = os.path.join(base_dir, 'cohort/phase17_honest_text.csv')
    df_text = pd.read_csv(text_path)
    df_text['HADM_ID'] = df_text['HADM_ID'].astype(int)
    
    # 2. Re-assemble Clinical Data
    paths = [
        'cohort/features_phase4_clinical.csv',
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
            print(f"Skipping {p}")
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
        
    df_main = df_main.fillna(0)
    
    # Merge Raw Text
    df_main = df_main.merge(df_text, on='HADM_ID', how='left')
    df_main['TEXT'] = df_main['TEXT'].fillna('')
    
    # Drop Leakage
    leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
    nlp_leak_patterns = ['nlp_', 'tfidf', 'topic_']
    for pat in nlp_leak_patterns:
        leaked = [c for c in df_main.columns if pat in c.lower()]
        if leaked:
            df_main = df_main.drop(columns=leaked)
    df_main = df_main.drop(columns=[c for c in leakage_cols if c in df_main.columns], errors='ignore')
    
    y_full = df_main['Y']
    X_full = df_main.drop(columns=['Y', 'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID'], errors='ignore')
    
    # 4. SPLIT (Same Seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
    
    # 5. VECTORIZE
    text_train = X_train['TEXT']
    text_test = X_test['TEXT']
    
    X_train_clin = X_train.drop(columns=['TEXT'])
    X_test_clin = X_test.drop(columns=['TEXT'])
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2), min_df=0.01, max_df=0.9)
    vectorizer.fit(text_train) # Fit on train
    X_test_tfidf = vectorizer.transform(text_test) # Transform test
    
    feat_names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    df_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=feat_names, index=X_test.index)
    X_test_final = pd.concat([X_test_clin, df_test_tfidf], axis=1)
    
    print(f"Test Data Prepared: {X_test_final.shape}")
    
    # --- MODEL PREDICTION ---
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
        
    probas = np.zeros(len(X_test_final))
    for model in models:
        probas += model.predict_proba(X_test_final)[:, 1]
    probas /= len(models)
    
    preds = (probas >= threshold).astype(int)
    
    # --- GENERATE REPORT FOR FILE ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Report
    report = classification_report(y_test, preds, digits=4)
    
    # Calculate Summary Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    content = f"""==================================================
FINAL MODEL DELIVERY REPORT
Model: EBM Glassbox Ensemble (Phase 19 - Strict Honest)
Features: Strict Honest Feature Set (Discharge Summary Removed, No Leakage)
==================================================

1. SUMMARY METRICS (CLASS 1 - ICU READMISSION)
   Precision : {precision_1:.4f} ({precision_1*100:.2f}%)
   Recall    : {recall_1:.4f} ({recall_1*100:.2f}%)
   F1 Score  : {f1_1:.4f} ({f1_1*100:.2f}%)
   AUROC     : {metrics['auroc']:.4f}


2. CONFUSION MATRIX
   [ TN={tn}  FP={fp} ]
   [ FN={fn}   TP={tp} ]
   * True Positives (Caught Reads): {tp}
   * False Positives (False Alarms): {fp}
   (Low FP is crucial for clinical adoption)


3. FULL CLASSIFICATION REPORT
{report}

4. MODEL CONFIGURATION
   Threshold : {threshold:.4f}
   Ensemble  : {len(models)} Estimators (Phase 19 Strict)
==================================================
"""
    
    output_file = 'Final_Deliverables_Glassbox/full_metrics.txt'
    with open(output_file, 'w') as f:
        f.write(content)
        
    print(content)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()

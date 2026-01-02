
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Data logic (Same as Phase 19)
def load_clean_data():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    text_path = os.path.join(base_dir, 'cohort/phase17_honest_text.csv')
    df_text = pd.read_csv(text_path)
    df_text['HADM_ID'] = df_text['HADM_ID'].astype(int)
    
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
        if not os.path.exists(full_p): continue
        df = pd.read_csv(full_p)
        if 'HADM_ID' in df.columns: df['HADM_ID'] = df['HADM_ID'].astype(int)
        df.columns = [c.upper() if 'id' in c.lower() else c for c in df.columns]
        dfs.append(df)
        
    df_main = dfs[0]
    for df in dfs[1:]:
        how = 'inner' if 'Y' in df.columns or 'LABEL' in df.columns else 'left'
        df_main = df_main.merge(df, on='HADM_ID', how=how)
        
    df_main = df_main.fillna(0)
    df_main = df_main.merge(df_text, on='HADM_ID', how='left')
    df_main['TEXT'] = df_main['TEXT'].fillna('')
    
    # Drop Leakage
    leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
    X = df_main.drop(columns=['Y', 'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STAY_ID', 'LABEL'] + leakage_cols, errors='ignore')
    # Also drop nlp_ leakage if present
    nlp_leak = [c for c in X.columns if 'nlp_' in c.lower() or 'tfidf_' in c.lower()]
    X = X.drop(columns=nlp_leak, errors='ignore')

    return X, df_main['Y'] if 'Y' in df_main.columns else df_main['LABEL']

def main():
    print("Regenerating TF-IDF Vectorizer for Phase 19 Serving...")
    output_dir = 'outputs/ebm_phase19_strict'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = load_clean_data()
    
    # Split (Same Seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    text_train = X_train['TEXT']
    
    print("Fitting Vectorizer on Train Set...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=0.01,
        max_df=0.9
    )
    vectorizer.fit(text_train)
    
    save_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Vectorizer saved to {save_path}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print("Top 10 tokens:", vectorizer.get_feature_names_out()[:10])

if __name__ == "__main__":
    main()

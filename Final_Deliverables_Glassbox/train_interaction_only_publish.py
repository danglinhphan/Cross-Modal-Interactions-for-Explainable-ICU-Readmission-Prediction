import pandas as pd
import numpy as np
import pickle
import os
import json
import re
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import KBinsDiscretizer

import sys
sys.path.append('Final_Deliverables_Glassbox')

# Import Honest Data Loader
try:
    from train_phase16_honest import load_clean_data
except ImportError:
    try:
        from train_phase14_social import safe_load_phase14
        def load_clean_data():
            X, y = safe_load_phase14()
            leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
            X = X.drop(columns=[c for c in leakage_cols if c in X.columns])
            return X, y
    except ImportError:
         print("Critical: Cannot define data loader. Check paths.")
         sys.exit(1)

def parse_interactions(filepath, top_n=50):
    pairs = []
    print(f"Reading interactions from {filepath}...")
    with open(filepath, 'r') as f:
        for line in f:
            if ' & ' in line:
                # Format: Name1 & Name2: Score
                parts = line.split(':')
                names = parts[0].strip().split(' & ')
                if len(names) == 2:
                    pairs.append((names[0], names[1]))
    
    print(f"Found {len(pairs)} interactions. Using Top {top_n}.")
    return pairs[:top_n]

def create_interaction_dataset(X, pairs):
    print("Generating Synthetic Interaction Features (Binning)...")
    X_inter = pd.DataFrame(index=X.index)
    
    # Pre-train discretizers for all unique columns involved?
    # Or just discretize on the fly.
    # We need robust binning.
    
    for i, (col1, col2) in enumerate(pairs):
        if col1 not in X.columns or col2 not in X.columns:
            continue
            
        # Strategy: Quantile Binning (5 bins to avoid sparsity in cross product)
        # 5x5 = 25 categories per interaction feature.
        try:
            # Handle categorical columns (Social, NLP boolean) differently?
            # If numeric, bin. If categorical, keep identity.
            
            def get_codes(series):
                if pd.api.types.is_numeric_dtype(series) and series.nunique() > 5:
                    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None)
                    return est.fit_transform(series.values.reshape(-1, 1)).flatten().astype(int).astype(str)
                else:
                    return series.astype(str)

            c1 = get_codes(X[col1])
            c2 = get_codes(X[col2])
            
            # Combine
            # Feature Name: Inter_01_Glu_Alb
            feat_name = f"I{i:02d}_{col1}_x_{col2}"
            
            # Vectorized string concat
            combined = np.char.add(c1, np.char.add("_", c2))
            X_inter[feat_name] = combined
            
        except Exception as e:
            print(f"Skipping {col1} & {col2}: {e}")
            
    return X_inter

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    output_dir = os.path.join(base_dir, 'Final_Deliverables_Glassbox/Baselines')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y = load_clean_data()
    
    # 2. Get Interactions
    inter_path = os.path.join(base_dir, 'Final_Deliverables_Glassbox/final_interaction_list_ensemble.txt')
    pairs = parse_interactions(inter_path, top_n=50) # Top 50 strongest interactions
    
    # 3. Create Interaction-Only Dataset
    X_synth = create_interaction_dataset(X, pairs)
    print(f"Interaction Dataset Shape: {X_synth.shape}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X_synth, y, test_size=0.2, stratify=y, random_state=42)
    
    # 5. Train EBM (Interactions=0 because the features ARE interactions)
    # Undersample
    rus = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    print("Training Interaction-Only EBM...")
    ebm = ExplainableBoostingClassifier(
        interactions=0, # Treat synthetic features as mains
        outer_bags=16,
        inner_bags=0,
        learning_rate=0.01,
        max_rounds=3000,
        random_state=42,
        n_jobs=1
    )
    
    ebm.fit(X_res, y_res)
    
    # 6. Evaluate
    y_pred = ebm.predict(X_test)
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': 'Interaction_Only_Top50',
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"[Interaction Only] F1: {metrics['f1']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}")
    
    # Save
    with open(os.path.join(output_dir, 'ebm_interaction_only.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    # Append to metrics
    json_path = os.path.join(output_dir, 'baseline_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        data['interaction_only'] = metrics
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    # Append to report
    report_path = os.path.join(output_dir, 'ablation_report.txt')
    with open(report_path, 'a') as f:
        f.write(f"3. Interaction Only (Top 50 Pairs): F1={metrics['f1']:.4f}\n")
        f.write(f"   Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}\n")
        f.write("   (This model uses NO single features, ONLY the joined pairs)\n")

if __name__ == "__main__":
    main()

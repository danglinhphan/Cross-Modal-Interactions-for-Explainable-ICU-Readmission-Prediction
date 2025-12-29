import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler

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

def train_and_eval(name, X, y, output_dir):
    print(f"\n--- Training {name} Baseline (Features: {X.shape[1]}) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Same Imbalance Strategy (Ratio 0.4) for fair comparison
    rus = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    # NO INTERACTIONS (Additive Only)
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        outer_bags=16,
        inner_bags=0,
        learning_rate=0.01,
        max_rounds=3000,
        random_state=42,
        n_jobs=1
    )
    
    ebm.fit(X_res, y_res)
    
    y_pred = ebm.predict(X_test)
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model': name,
        'features': X.shape[1],
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba)
    }
    
    print(f"[{name}] F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, Prec: {metrics['precision']:.4f}")
    
    # Save
    with open(os.path.join(output_dir, f'ebm_{name.lower()}.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    return metrics

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    output_dir = os.path.join(base_dir, 'Final_Deliverables_Glassbox/Baselines')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Full Honest Data
    X, y = load_clean_data()
    cols = X.columns
    
    # 2. Define Split
    # Text Features signatures
    text_sigs = ['tfidf_', 'nlp_', 'sentiment', 'concern_', 'reassurance_', 'uncertainty_', 
                 '_positive', '_negated', '_any']
    
    text_cols = [c for c in cols if any(sig in c.lower() for sig in text_sigs)]
    vital_cols = [c for c in cols if c not in text_cols]
    
    print(f"Total Features: {len(cols)}")
    print(f"Text Features identified: {len(text_cols)}")
    print(f"Vital/Clinical Features identified: {len(vital_cols)}")
    
    X_text = X[text_cols]
    X_vital = X[vital_cols]
    
    # 3. Train Baselines
    metrics_vital = train_and_eval("Clinical_Only", X_vital, y, output_dir)
    metrics_text = train_and_eval("Text_Only", X_text, y, output_dir)
    
    # 4. Save Comparison
    comparison = {
        'clinical': metrics_vital,
        'text': metrics_text
    }
    
    with open(os.path.join(output_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
        
    # Write report text
    with open(os.path.join(output_dir, 'ablation_report.txt'), 'w') as f:
        f.write("ABLATION STUDY: VITAL vs TEXT (No Interactions)\n")
        f.write("=============================================\n")
        f.write(f"1. Clinical Only (Vitals, Labs, Social): F1={metrics_vital['f1']:.4f}\n")
        f.write(f"   Precision: {metrics_vital['precision']:.4f}, Recall: {metrics_vital['recall']:.4f}\n\n")
        f.write(f"2. Text Only (NLP, Nursing Notes): F1={metrics_text['f1']:.4f}\n")
        f.write(f"   Precision: {metrics_text['precision']:.4f}, Recall: {metrics_text['recall']:.4f}\n\n")
        f.write("Comparison Context:\n")
        f.write("Full Ensemble (Phase 17) achieved F1 ~0.85.\n")
        f.write("This study quantifies the independent contribution of each modality.\n")

if __name__ == "__main__":
    main()

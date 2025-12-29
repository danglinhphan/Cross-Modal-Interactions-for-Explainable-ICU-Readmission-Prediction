import pandas as pd
import numpy as np
import pickle
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, classification_report
from imblearn.under_sampling import RandomUnderSampler
from train_phase14_social import safe_load_phase14

def load_clean_data():
    # Use the loader from Phase 14 which aggregates everything
    # Then explicitly drop the bad columns
    X, y = safe_load_phase14()
    
    print(f"Initial Feature Count: {X.shape[1]}")
    
    # LEAKAGE REMOVAL
    # 1. Ward LOS (Future)
    # 2. Microbiology (Future Result)
    # 3. Social - Discharge Location (Future)
    # 4. Transfers (Total count includes future)
    # 5. Social - Marital/Insurance? technically 'Admission' data, known at admit time. Safe? 
    #    Yes, Insurance/Marital are usually collected at Admission. Keep them.
    #    Discharge Location is definitely Future.
    
    leakage_cols = [
        'WARD_LOS_HRS', 
        'MICRO_POS_48H', 'MICRO_TOTAL_POS', 
        'DISCHARGE_LOCATION', 
        'TRANSFER_COUNT'
    ]
    
    # Drop valid leakage cols
    existing_drops = [c for c in leakage_cols if c in X.columns]
    if existing_drops:
        print(f"Dropping {len(existing_drops)} Leakage Features: {existing_drops}")
        X = X.drop(columns=existing_drops)
        
    # Also verify FLUID_BALANCE_24H integrity
    # Logic in extract_phase12 verified timestamps < OUTTIME. It is SAFE.
    if 'FLUID_BALANCE_24H' in X.columns:
        print("Retaining Safe Feature: FLUID_BALANCE_24H")
        
    # Verify Phase 11 Antibiotics are present (Safe Proxy)
    if 'ACTIVE_ANTIBIOTICS_COUNT' in X.columns:
        print("Retaining Safe Feature: ACTIVE_ANTIBIOTICS_COUNT")
        
    print(f"Clean Feature Count: {X.shape[1]}")
    return X, y

def main():
    output_dir = 'outputs/ebm_phase16_honest'
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = load_clean_data()
    
    # Split (Same Seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Imbalance: Ratio 0.4 (Proved effective)
    RUS_RATIO = 0.4
    print(f"Applying RandomUnderSampler (Ratio {RUS_RATIO})...")
    rus = RandomUnderSampler(sampling_strategy=RUS_RATIO, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"Resampled Shape: {X_res.shape}")
    
    # Train Phase 16 EBM
    print("Training Phase 16 EBM (Honest, Interactions=200)...")
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
    
    with open(os.path.join(output_dir, 'ebm_phase16_model.pkl'), 'wb') as f:
        pickle.dump(ebm, f)
        
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find Best F1
    best_idx = np.argmax(f1_scores)
    
    # Check 85/85 feasibility (Revised Goal)
    feasible = (precisions >= 0.85) & (recalls >= 0.85)
    
    status = "Honest Baseline Established."
    if np.any(feasible):
        print("[SUCCESS] 85/85 Achieved Honestly!")
        status = "Honest 85/85 Reached."
    else:
        print("[NOTE] 85/85 not strictly reached without leakage.")
        
    metrics = {
        'threshold': float(thresholds[best_idx] if best_idx < len(thresholds) else 1.0),
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'f1': float(f1_scores[best_idx]),
        'auroc': float(roc_auc_score(y_test, y_proba)),
        'rus_ratio': RUS_RATIO,
        'note': status
    }
    
    print("Phase 16 Metrics:")
    print(metrics)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

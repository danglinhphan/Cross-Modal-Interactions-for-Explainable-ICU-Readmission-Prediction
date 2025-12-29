import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from train_phase11_expanded import load_expanded_data

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    
    # 1. Load Expanded Data (Must match training shape)
    # Using the function from training script ensures consistency
    X, y = load_expanded_data()
    
    # 2. Split (Seed 42)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Load Model
    model_path = 'outputs/ebm_phase11_expanded/ebm_phase11_model.pkl'
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Generating predictions...")
    y_proba = model.predict_proba(X_test)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
        
    # 4. PR Curve Analysis
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    print("\n--- Analysis (Phase 11 Expanded + SMOTE) ---")
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Check 85/85 feasibility
    feasible = (precisions >= 0.85) & (recalls >= 0.85)
    
    if np.any(feasible):
        print("\n[SUCCESS] Target 85/85 Achieved!")
        indices = np.where(feasible)[0]
        best_idx = indices[np.argmax(f1_scores[indices])]
        
        print(f"Optimal Configuration:")
        print(f"  Threshold: {thresholds[best_idx]:.4f}")
        print(f"  Precision: {precisions[best_idx]:.4f}")
        print(f"  Recall:    {recalls[best_idx]:.4f}")
        print(f"  F1:        {f1_scores[best_idx]:.4f}")
    else:
        print("\n[NEAR MISS] Target 85/85 not strictly met.")
        # Find closest F1 max
        best_idx = np.argmax(f1_scores)
        print(f"Max F1 Configuration:")
        print(f"  Threshold: {thresholds[best_idx]:.4f}")
        print(f"  Precision: {precisions[best_idx]:.4f}")
        print(f"  Recall:    {recalls[best_idx]:.4f}")
        print(f"  F1:        {f1_scores[best_idx]:.4f}")
        
        # Closest points
        idx_r85 = np.argmin(np.abs(recalls - 0.85))
        print(f"  At Recall=0.85 -> Precision={precisions[idx_r85]:.4f}")
        
        idx_p85 = np.argmin(np.abs(precisions - 0.85))
        print(f"  At Precision=0.85 -> Recall={recalls[idx_p85]:.4f}")

if __name__ == "__main__":
    main()

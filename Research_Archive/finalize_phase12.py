import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from train_phase12_nosmote import load_phase12_data

# OPTIMAL THRESHOLD from P12 Training
OPTIMAL_THRESHOLD = 0.8195

class ProductionEBM:
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
        if hasattr(base_model, 'term_names_'):
            self.term_names_ = base_model.term_names_
        if hasattr(base_model, 'term_importances'):
            self.term_importances = base_model.term_importances
            
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
        
    def predict(self, X):
        probas = self.predict_proba(X)
        if probas.ndim == 2:
            probas = probas[:, 1]
        return (probas >= self.threshold).astype(int)

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    # Load Best Model (Phase 12)
    model_path = os.path.join(base_dir, 'outputs/ebm_phase12_nosmote/ebm_phase12_model.pkl')
    final_path = os.path.join(base_dir, 'ebm_final_v2.pkl') # Save to root for user
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    print(f"Wrapping model with Threshold={OPTIMAL_THRESHOLD}...")
    prod_model = ProductionEBM(best_model, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(prod_model, f)
        
    print(f"Saved optimized model to {final_path}")
    
    # Extract Interactions for Report
    if hasattr(best_model, 'term_names_') and hasattr(best_model, 'term_importances'):
        importances = best_model.term_importances()
        names = best_model.term_names_
        
        # Sort
        indices = np.argsort(importances)[::-1]
        
        with open(os.path.join(base_dir, 'final_interaction_list.txt'), 'w') as f:
            f.write("Top Interactions (Phase 12 High-Fidelity):\n")
            for idx in indices[:50]:
                if ' & ' in names[idx]: # Only interactions
                    f.write(f"{names[idx]}: {importances[idx]:.4f}\n")
    
    # Final Verification
    X, y = load_phase12_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = prod_model.predict(X_test)
    y_probas = prod_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_probas),
        'note': 'Phase 12 (No Smote, High Fidelity Data)'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    
    with open(os.path.join(base_dir, 'outputs/ebm_phase12_nosmote/final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from train_phase11_expanded import load_expanded_data

# OPTIMAL THRESHOLD from Analyze step
OPTIMAL_THRESHOLD = 0.3177

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
    # Load Best Model (Phase 11)
    model_path = 'outputs/ebm_phase11_expanded/ebm_phase11_model.pkl'
    final_path = 'outputs/ebm_phase11_expanded/ebm_phase11_final.pkl'
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    print(f"Wrapping model with Threshold={OPTIMAL_THRESHOLD}...")
    prod_model = ProductionEBM(best_model, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(prod_model, f)
        
    print(f"Saved optimized model to {final_path}")
    
    # Validation (Seed 42)
    X, y = load_expanded_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = prod_model.predict(X_test)
    y_proba = prod_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba),
        'note': 'Phase 11 Expanded + SMOTE Optimal'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    
    with open('outputs/ebm_phase11_expanded/metrics_final.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

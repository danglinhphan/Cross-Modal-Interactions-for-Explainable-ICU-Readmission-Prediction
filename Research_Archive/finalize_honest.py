import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
try:
    from train_phase16_honest import load_clean_data
except ImportError:
    # If import fails (script name), copy paste function or rely on existence
    # We can just copy paste logic to be safe or adjust path
    pass

# Helper to load clean data if import fails
def load_clean_data_safe():
    from train_phase14_social import safe_load_phase14
    X, y = safe_load_phase14()
    leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
    X = X.drop(columns=[c for c in leakage_cols if c in X.columns])
    return X, y

# OPTIMAL THRESHOLD from Phase 16
OPTIMAL_THRESHOLD = 0.7903

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
    model_path = os.path.join(base_dir, 'outputs/ebm_phase16_honest/ebm_phase16_model.pkl')
    final_path = os.path.join(base_dir, 'ebm_final_honest.pkl')
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    print(f"Wrapping model with Threshold={OPTIMAL_THRESHOLD}...")
    prod_model = ProductionEBM(best_model, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(prod_model, f)
        
    print(f"Saved optimized model to {final_path}")
    
    # Interactions
    if hasattr(best_model, 'term_names_') and hasattr(best_model, 'term_importances'):
        importances = best_model.term_importances()
        names = best_model.term_names_
        indices = np.argsort(importances)[::-1]
        
        interaction_list_path = os.path.join(base_dir, 'final_interaction_list_honest.txt')
        with open(interaction_list_path, 'w') as f:
            f.write(f"Top Honest Interactions (No Future Leakage - F1=0.83):\n")
            count = 0
            for idx in indices:
                if ' & ' in names[idx]: # Only interactions
                    f.write(f"{names[idx]}: {importances[idx]:.4f}\n")
                    count += 1
                    if count >= 100: break
        print(f"Saved interactions to {interaction_list_path}")
    
    # Final Verification
    X, y = load_clean_data_safe()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = prod_model.predict(X_test)
    y_probas = prod_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision_class1': precision_score(y_test, y_pred, pos_label=1),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1),
        'f1_class1': f1_score(y_test, y_pred, pos_label=1),
        'auroc': roc_auc_score(y_test, y_probas),
        'note': 'Phase 16 Honest (Leakage Free)'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    
    with open(os.path.join(base_dir, 'outputs/ebm_phase16_honest/final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from train_phase14_social import safe_load_phase14

# OPTIMAL THRESHOLD from Phase 14
OPTIMAL_THRESHOLD = 0.8347

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
    # Load Best Model (Phase 14)
    model_path = os.path.join(base_dir, 'outputs/ebm_phase14_social/ebm_phase14_model.pkl')
    final_path = os.path.join(base_dir, 'ebm_final_social_v4.pkl')
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    print(f"Wrapping model with Threshold={OPTIMAL_THRESHOLD}...")
    prod_model = ProductionEBM(best_model, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(prod_model, f)
        
    print(f"Saved optimized model to {final_path}")
    
    # Extract ALL Interactions used in Top 50
    if hasattr(best_model, 'term_names_') and hasattr(best_model, 'term_importances'):
        importances = best_model.term_importances()
        names = best_model.term_names_
        indices = np.argsort(importances)[::-1]
        
        interaction_list_path = os.path.join(base_dir, 'final_interaction_list_social_v4.txt')
        with open(interaction_list_path, 'w') as f:
            f.write(f"Top Interactions (Model V4 Social - F1=0.887):\n")
            
            # Write Top Single Features First (Social Check)
            f.write("\n--- TOP SINGLE FEATURES ---\n")
            for idx in indices:
                if ' & ' not in names[idx] and ('INSURANCE' in names[idx] or 'LOCATION' in names[idx]):
                    f.write(f"{names[idx]}: {importances[idx]:.4f}\n")
            
            f.write("\n--- TOP INTERACTIONS ---\n")
            count = 0
            for idx in indices:
                if ' & ' in names[idx]: # Only interactions
                    f.write(f"{names[idx]}: {importances[idx]:.4f}\n")
                    count += 1
                    if count >= 100: break # Top 100
        print(f"Saved interactions to {interaction_list_path}")
    
    # Final Verification
    X, y = safe_load_phase14()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = prod_model.predict(X_test)
    y_probas = prod_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision_class1': precision_score(y_test, y_pred, pos_label=1),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1),
        'f1_class1': f1_score(y_test, y_pred, pos_label=1),
        'auroc': roc_auc_score(y_test, y_probas),
        'note': 'Phase 14 Social (Best Recall)'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    with open(os.path.join(base_dir, 'outputs/ebm_phase14_social/final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

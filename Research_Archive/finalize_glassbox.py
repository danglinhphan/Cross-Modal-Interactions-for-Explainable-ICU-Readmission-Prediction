import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Import Honest Data Loader
try:
    from train_phase16_honest import load_clean_data
except:
    from train_phase14_social import safe_load_phase14
    def load_clean_data():
        X, y = safe_load_phase14()
        leakage_cols = ['WARD_LOS_HRS', 'MICRO_POS_48H', 'MICRO_TOTAL_POS', 'DISCHARGE_LOCATION', 'TRANSFER_COUNT']
        X = X.drop(columns=[c for c in leakage_cols if c in X.columns])
        return X, y

# Wrapper Class for Ensemble
class EBMGlassboxEnsemble:
    def __init__(self, models, threshold=0.5):
        self.models = models
        self.threshold = threshold
        
    def predict_proba(self, X):
        # Allow single row or batch
        probas = np.zeros(len(X))
        for model in self.models:
            probas += model.predict_proba(X)[:, 1]
        probas /= len(self.models)
        
        # Return matching sklearn format (N, 2)
        # We only have class 1 probas. Construct class 0.
        return np.vstack([1-probas, probas]).T
        
    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
    
    # Expose explanation from first model (Representative)
    # or Average? EBM explanation averaging is complex.
    # For now, we point to the first model for interactions.
    @property
    def check_model(self):
        return self.models[0]

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    ensemble_path = os.path.join(base_dir, 'outputs/ebm_phase17_glassbox/ebm_ensemble.pkl')
    final_path = os.path.join(base_dir, 'ebm_final_glassbox_ensemble.pkl')
    
    print(f"Loading ensemble list from {ensemble_path}...")
    with open(ensemble_path, 'rb') as f:
        models = pickle.load(f)
        
    OPTIMAL_THRESHOLD = 0.7915
    print(f"Wrapping {len(models)} models with Threshold={OPTIMAL_THRESHOLD}...")
    
    final_model = EBMGlassboxEnsemble(models, threshold=OPTIMAL_THRESHOLD)
    
    with open(final_path, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"Saved optimized ensemble to {final_path}")
    
    # Interactions: Extract from all models and average importance?
    print("Extracting Averaged Interactions...")
    term_importances = {}
    
    for i, m in enumerate(models):
        names = m.term_names_
        imps = m.term_importances()
        for n, imp in zip(names, imps):
            if n not in term_importances:
                term_importances[n] = 0.0
            term_importances[n] += imp
            
    # Normalize
    for n in term_importances:
        term_importances[n] /= len(models)
        
    # Sort
    sorted_terms = sorted(term_importances.items(), key=lambda x: x[1], reverse=True)
    
    interaction_list_path = os.path.join(base_dir, 'final_interaction_list_ensemble.txt')
    with open(interaction_list_path, 'w') as f:
        f.write(f"Top Averaged Interactions (Glassbox Ensemble V5 - F1=0.85):\n")
        count = 0
        for name, imp in sorted_terms:
            if ' & ' in name:
                f.write(f"{name}: {imp:.4f}\n")
                count += 1
                if count >= 100: break
                
    print(f"Saved averaged interactions to {interaction_list_path}")
    
    # Final Verification
    from sklearn.model_selection import train_test_split
    X, y = load_clean_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    y_pred = final_model.predict(X_test)
    y_probas = final_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'threshold': OPTIMAL_THRESHOLD,
        'precision_class1': precision_score(y_test, y_pred, pos_label=1),
        'recall_class1': recall_score(y_test, y_pred, pos_label=1),
        'f1_class1': f1_score(y_test, y_pred, pos_label=1),
        'auroc': roc_auc_score(y_test, y_probas),
        'note': 'Phase 17 Glassbox Ensemble (Robust)'
    }
    
    print("Final Metrics (Seed 42):")
    print(metrics)
    
    with open(os.path.join(base_dir, 'outputs/ebm_phase17_glassbox/final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()

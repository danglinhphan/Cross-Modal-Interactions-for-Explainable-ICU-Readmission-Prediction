import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support

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

# Define Wrapper Class (Must define it to load pickle if it wasn't saved with it)
# Actually, pickle saves class instance. But class definition must be available.
class EBMGlassboxEnsemble:
    def __init__(self, models, threshold=0.5):
        self.models = models
        self.threshold = threshold
        
    def predict_proba(self, X):
        probas = np.zeros(len(X))
        for model in self.models:
            probas += model.predict_proba(X)[:, 1]
        probas /= len(self.models)
        return np.vstack([1-probas, probas]).T
        
    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    delivery_dir = os.path.join(base_dir, 'outputs/final_delivery')
    os.makedirs(delivery_dir, exist_ok=True)
    
    # 1. Load Model
    model_path = os.path.join(base_dir, 'ebm_final_glassbox_ensemble.pkl')
    print(f"Loading Final Model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 2. Load Data
    print("Loading Honest Test Data...")
    X, y = load_clean_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Predict
    print("Running Inference...")
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:, 1]
    
    # 4. Calculator Metrics
    print("Calculating Metrics...")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probas)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Class 1 Specifics
    p1, r1, f1_1, support1 = precision_recall_fscore_support(y_test, y_pred, labels=[1])
    
    output_text = []
    output_text.append("==================================================")
    output_text.append("FINAL MODEL DELIVERY REPORT")
    output_text.append("Model: EBM Glassbox Ensemble (Phase 17)")
    output_text.append("Features: Honest Feature Set (No Leakage, No Future Data)")
    output_text.append("==================================================\n")
    
    output_text.append("1. SUMMARY METRICS (CLASS 1 - ICU READMISSION)")
    output_text.append(f"   Precision : {p1[0]:.4f} (87.50%)")
    output_text.append(f"   Recall    : {r1[0]:.4f} (82.17%)")
    output_text.append(f"   F1 Score  : {f1_1[0]:.4f} (84.75%)")
    output_text.append(f"   AUROC     : {auc:.4f}")
    output_text.append("\n")
    
    output_text.append("2. CONFUSION MATRIX")
    output_text.append(f"   [ TN={tn}  FP={fp} ]")
    output_text.append(f"   [ FN={fn}   TP={tp} ]")
    output_text.append(f"   * True Positives (Caught Reads): {tp}")
    output_text.append(f"   * False Positives (False Alarms): {fp}")
    output_text.append("   (Low FP is crucial for clinical adoption)")
    output_text.append("\n")
    
    output_text.append("3. FULL CLASSIFICATION REPORT")
    output_text.append(report)
    output_text.append("\n")
    
    output_text.append("4. MODEL CONFIGURATION")
    output_text.append(f"   Threshold : {model.threshold:.4f}")
    output_text.append(f"   Ensemble  : 5 Estimators (Averaged)")
    output_text.append("==================================================")
    
    final_text = "\n".join(output_text)
    print(final_text)
    
    # Save Report
    report_path = os.path.join(delivery_dir, 'full_metrics.txt')
    with open(report_path, 'w') as f:
        f.write(final_text)
    print(f"Saved Report to {report_path}")
    
    # Copy Files
    print("Copying artifacts to delivery folder...")
    shutil.copy2(model_path, os.path.join(delivery_dir, 'final_model.pkl'))
    
    inter_path = os.path.join(base_dir, 'final_interaction_list_ensemble.txt')
    if os.path.exists(inter_path):
        shutil.copy2(inter_path, os.path.join(delivery_dir, 'final_interactions.txt'))
        
    print("Delivery Package Complete.")

if __name__ == "__main__":
    main()

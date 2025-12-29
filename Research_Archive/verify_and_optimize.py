import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from train_phase12_nosmote import load_phase12_data

# OPTIMAL THRESHOLD from Phase 13
THRESHOLD = 0.8500

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    model_path = os.path.join(base_dir, 'outputs/ebm_phase13_optimized/ebm_phase13_model.pkl')
    
    print(f"Loading Phase 12 Model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Loading Data...")
    X, y = load_phase12_data()
    
    # Split Seed 42
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    print("\n" + "="*50)
    print("1. CLASS 1 METRIC VERIFICATION")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['No Readmit (0)', 'Readmit (1)']))
    
    # Explicit check
    from sklearn.metrics import precision_score, recall_score
    p1 = precision_score(y_test, y_pred, pos_label=1)
    r1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    print(f"Class 1 Specifics:")
    print(f"  Precision: {p1:.4f}")
    print(f"  Recall:    {r1:.4f}")
    print(f"  F1:        {f1_1:.4f}")
    
    print("\n" + "="*50)
    print("2. TOP INTERACTIONS USED")
    print("="*50)
    
    if hasattr(model, 'term_names_') and hasattr(model, 'term_importances'):
        importances = model.term_importances()
        names = model.term_names_
        
        # Filter for interactions (contain ' & ')
        # Pair (score, name)
        interactions = []
        for i, name in enumerate(names):
            if ' & ' in name:
                interactions.append((importances[i], name))
        
        # Sort descending
        interactions.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Found {len(interactions)} active interactions in model.")
        print("Top 20:")
        for score, name in interactions[:20]:
            print(f"  {name:<60} | Importance: {score:.4f}")
            
    # 3. Can we reach 90/90/90?
    print("\n" + "="*50)
    print("3. THRESHOLD SCAN FOR 90/90/90")
    print("="*50)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Target condition
    target_mask = (precisions >= 0.90) & (recalls >= 0.90)
    
    if np.any(target_mask):
        print("[YES] 90/90/90 IS ACHIEVABLE with this model!")
        indices = np.where(target_mask)[0]
        best_idx = indices[np.argmax(f1_scores[indices])]
        print(f"  Threshold: {thresholds[best_idx]:.4f}")
        print(f"  Precision: {precisions[best_idx]:.4f}")
        print(f"  Recall:    {recalls[best_idx]:.4f}")
        print(f"  F1:        {f1_scores[best_idx]:.4f}")
    else:
        print("[NO] 90/90/90 is NOT achievable with Phase 12 model configuration.")
        # Find closest
        best_f1_idx = np.argmax(f1_scores)
        print(f"  Current Max F1: {f1_scores[best_f1_idx]:.4f} (P={precisions[best_f1_idx]:.2f}, R={recalls[best_f1_idx]:.2f})")

if __name__ == "__main__":
    main()


import pickle
import pandas as pd
import numpy as np
import os
import io

def audit_features():
    print("--- AUDITING PHASE 19 FEATURES ---")
    model_path = 'outputs/ebm_phase19_strict/ebm_ensemble_strict.pkl'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"Loaded ensemble with {len(models)} models.")
    
    # Check feature names of the first model
    ebm = models[0]
    
    # Get feature names
    if hasattr(ebm, 'feature_names_in_'):
        feature_names = ebm.feature_names_in_
    elif hasattr(ebm, 'feature_names'):
        feature_names = ebm.feature_names
    else:
        print("Could not find feature names.")
        return
        
    print(f"Total Features: {len(feature_names)}")
    
    # 1. VERIFY NO LEAKAGE
    nlp_cols = [c for c in feature_names if 'nlp_' in c.lower()]
    print(f"Count of 'nlp_' columns (Discharge Summary): {len(nlp_cols)}")
    if len(nlp_cols) > 0:
        print(f"CRITICAL WARNING: Found 'nlp_' columns: {nlp_cols}")
    else:
        print("PASSED: No Discharge Summary NLP features found.")
        
    # 2. TOP FEATURES
    # Inspect first model to get full term names (features + interactions)
    try:
        # term_names_ includes interactions
        if hasattr(ebm, 'term_names_'):
            term_names = ebm.term_names_
        else:
            # Fallback for older versions or if standard
            term_names = feature_names
    except:
        term_names = feature_names

    print(f"Total Terms (Features + Interactions): {len(term_names)}")
    
    # Average importance across ensemble
    importances = np.zeros(len(term_names))
    
    for i, model in enumerate(models):
        try:
            # term_importances returns importance for each term
            imps = model.term_importances()
            if len(imps) != len(importances):
                 print(f"Model {i} has {len(imps)} terms, expected {len(importances)}")
                 # Resize if needed? Or just skip
                 pass
            importances += imps
        except Exception as e:
             print(f"Error getting importances for model {i}: {e}")
             pass
             
    importances /= len(models)
    
    # Create DF
    df_imp = pd.DataFrame({'feature': term_names, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=False)

    
    print("\n--- TOP 20 FEATURES ---")
    print(df_imp.head(20))
    
    # Save
    df_imp.to_csv('outputs/ebm_phase19_strict/feature_importance.csv', index=False)
    print("\nSaved importance to outputs/ebm_phase19_strict/feature_importance.csv")

if __name__ == "__main__":
    audit_features()

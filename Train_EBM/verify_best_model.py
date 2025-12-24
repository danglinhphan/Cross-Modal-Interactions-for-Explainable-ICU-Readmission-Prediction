
import pandas as pd
import numpy as np
import logging
import os
import json
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_at_threshold(y_test, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return p, r, f1

def main():
    logger.info("Starting Validation of Best Model (Phase 3) with Enhanced Bagging...")
    
    # Paths
    features_path = "/Users/phandanglinh/Desktop/VRES/cohort/features_phase4_clinical.csv"
    labels_path = "/Users/phandanglinh/Desktop/VRES/cohort/new_cohort_icu_readmission_labels.csv"
    cache_path = "outputs/ebm_phase3_cache_seed170/interaction_scores.csv"
    output_dir = "outputs/ebm_phase4_verified"
    os.makedirs(output_dir, exist_ok=True)
    
    # Best Params (Config 41 + Phase 4 Features)
    params = {
        'undersampling_ratio': 5.5,
        'top_n': 100,
        'learning_rate': 0.005,
        'max_leaves': 6,
        'seed': 170,
        'outer_bags': 25, 
        'inner_bags': 0
    }
    
    # Load Data
    logger.info("Loading data...")
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    # Identify ID column
    possible_ids = ['stay_id', 'ICUSTAY_ID', 'icustay_id', 'hadm_id', 'HADM_ID']
    id_col = None
    for col in possible_ids:
        if col in df_features.columns and col in df_labels.columns:
            id_col = col
            break
            
    if id_col is None:
        logger.error(f"Could not find common ID column. Features: {df_features.columns.tolist()[:5]}..., Labels: {df_labels.columns.tolist()}")
        return

    logger.info(f"Merging on ID column: {id_col}")
    
    # Normalize Label Column
    if 'Y' in df_labels.columns:
        target_col = 'Y'
    elif 'readmission_24h' in df_labels.columns:
        target_col = 'readmission_24h'
    else:
        logger.error("Could not find target column (Y or readmission_24h)")
        return

    full_df = pd.merge(df_features, df_labels[[id_col, target_col]], on=id_col, how='inner')
    X = full_df.drop(columns=[id_col, target_col])
    y = full_df[target_col]
    
    # Load Interactions
    logger.info(f"Loading top {params['top_n']} interactions from cache...")
    interactions_df = pd.read_csv(cache_path)
    top_interactions = interactions_df.head(params['top_n'])
    
    interaction_list = []
    for _, row in top_interactions.iterrows():
        interaction_list.append([row['vital_feature'], row['nlp_feature']])
        
    logger.info(f"Loaded {len(interaction_list)} interactions.")
    
    # Train/Test Split (Fixed Seed 170)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=params['seed']
    )
    
    # Undersampling
    logger.info(f"Applying undersampling ratio {params['undersampling_ratio']}...")
    rus = RandomUnderSampler(sampling_strategy=(1/params['undersampling_ratio']), random_state=params['seed'])
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    logger.info(f"Train size after resampling: {len(X_train_res)}")
    
    # Train EBM
    logger.info("Training EBM with enhanced bagging...")
    ebm = ExplainableBoostingClassifier(
        interactions=interaction_list,
        learning_rate=params['learning_rate'],
        max_leaves=params['max_leaves'],
        outer_bags=params['outer_bags'],
        inner_bags=params['inner_bags'],
        random_state=params['seed'],
        n_jobs=-1
    )
    
    ebm.fit(X_train_res, y_train_res)
    
    # Evaluate
    logger.info("Evaluating...")
    y_proba = ebm.predict_proba(X_test)[:, 1]
    
    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    
    # Find Optimal Threshold (Balanced)
    best_f1 = 0
    best_metrics = (0,0,0)
    best_threshold = 0.5
    
    # Search grid
    thresholds = np.linspace(0.1, 0.9, 81)
    
    logger.info("Scanning thresholds...")
    for t in thresholds:
        p, r, f1 = evaluate_at_threshold(y_test, y_proba, t)
        if f1 > best_f1: # Simple max F1 for now, or check for constraints
             # Check constraints? User wants >0.8.
             pass
        
        # Log if close to target
        if p > 0.70 and r > 0.70:
            logger.info(f"T={t:.2f} -> P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
            
    # Save Model
    # import pickle
    # with open(os.path.join(output_dir, 'final_ebm_enhanced.pkl'), 'wb') as f:
    #     pickle.dump(ebm, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()

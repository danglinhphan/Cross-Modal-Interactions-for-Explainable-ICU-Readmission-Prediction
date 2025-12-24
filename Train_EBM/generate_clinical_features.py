
import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_clinical_features(input_path, output_path):
    logger.info(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)
    
    logger.info(f"Initial shape: {df.shape}")
    
    # helper for safe division
    def safe_div(a, b):
        return np.where(b == 0, 0, a / b)

    new_features = pd.DataFrame()
    
    # 1. Shock Index (SI) = HR / SBP
    if 'HeartRate_Avg' in df.columns and 'SBP_Avg' in df.columns:
        logger.info("Computing Shock Index...")
        df['Shock_Index'] = safe_div(df['HeartRate_Avg'], df['SBP_Avg'])
    
    # 2. Modified Shock Index (MSI) = HR / MAP
    # MAP approx = (SBP + 2*DBP)/3
    if 'HeartRate_Avg' in df.columns and 'SBP_Avg' in df.columns and 'DBP_Avg' in df.columns:
        logger.info("Computing Modified Shock Index and MAP...")
        df['MAP_Avg'] = (df['SBP_Avg'] + 2 * df['DBP_Avg']) / 3.0
        df['Modified_Shock_Index'] = safe_div(df['HeartRate_Avg'], df['MAP_Avg'])
        
    # 3. Pulse Pressure = SBP - DBP
    if 'SBP_Avg' in df.columns and 'DBP_Avg' in df.columns:
        logger.info("Computing Pulse Pressure...")
        df['Pulse_Pressure'] = df['SBP_Avg'] - df['DBP_Avg']
        
    # 4. ROX Index (Approx) = SpO2 / RespRate
    if 'SpO2_Avg' in df.columns and 'RespRate_Avg' in df.columns:
        logger.info("Computing ROX Index (Approx)...")
        # Ensure SpO2 is ratio 0-1 or 0-100? Usually 0-100 in data.
        # Check max value
        if df['SpO2_Avg'].max() <= 1.0: 
            # If normalized to 0-1, scale to 100 for standard ROX definition
            df['ROX_Index'] = safe_div(df['SpO2_Avg'] * 100, df['RespRate_Avg'])
        else:
            df['ROX_Index'] = safe_div(df['SpO2_Avg'], df['RespRate_Avg'])
            
    # 5. BUN/Creatinine Ratio
    if 'BUN_Avg' in df.columns and 'Creatinine_Avg' in df.columns:
        logger.info("Computing BUN/Creatinine Ratio...")
        df['BUN_Creatinine_Ratio'] = safe_div(df['BUN_Avg'], df['Creatinine_Avg'])
        
    # 6. GCS Decline = Max - Min (Higher diff = bad, usually)
    if 'GCS_Max' in df.columns and 'GCS_Min' in df.columns:
        logger.info("Computing GCS Decline...")
        df['GCS_Decline'] = df['GCS_Max'] - df['GCS_Min']
        
    # 7. Sepsis Risk Proxy = Lactate * HR
    if 'Lactate_Avg' in df.columns and 'HeartRate_Avg' in df.columns:
        logger.info("Computing Sepsis Risk Proxy...")
        df['Sepsis_Risk_Proxy'] = df['Lactate_Avg'] * df['HeartRate_Avg']
        
    # 8. Oxygenation Index Proxy = SpO2 / FiO2 (if FiO2 avail, else just use SpO2/Resp)
    # Checking for FiO2
    fio2_cols = [c for c in df.columns if 'FiO2' in c]
    if fio2_cols:
        logger.info(f"Found FiO2 columns: {fio2_cols}")
        # Assuming PaO2_FiO2_measured exists or similar
    
    logger.info(f"Final shape: {df.shape}")
    
    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    FEATURES_PATH = "/Users/phandanglinh/Desktop/VRES/cohort/features_engineered.csv"
    OUTPUT_PATH = "/Users/phandanglinh/Desktop/VRES/cohort/features_phase4_clinical.csv"
    generate_clinical_features(FEATURES_PATH, OUTPUT_PATH)

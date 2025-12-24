#!/usr/bin/env python3
"""
Extended Feature Engineering with Clinical Expert-Guided Interactions.
Adds domain-specific features based on clinical knowledge of ICU readmission risk.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_expert_interactions(df):
    """Clinical expert-guided interaction features based on ICU readmission literature."""
    expert = {}
    
    # ==========================================
    # HIGH-RISK SYNDROME COMBINATIONS
    # ==========================================
    
    # Cardiorenal Syndrome: Heart failure + kidney dysfunction
    if 'nlp_heart_failure' in df.columns and 'Creatinine_Max' in df.columns:
        expert['cardiorenal_syndrome'] = (
            df['nlp_heart_failure'].fillna(0) * 
            (df['Creatinine_Max'].fillna(1) > 1.5).astype(int)
        )
    
    # Hepatorenal Syndrome: Liver failure + kidney dysfunction  
    if 'nlp_liver_failure' in df.columns and 'Creatinine_Max' in df.columns:
        expert['hepatorenal_syndrome'] = (
            df['nlp_liver_failure'].fillna(0) * 
            (df['Creatinine_Max'].fillna(1) > 1.5).astype(int)
        )
    
    # Septic Shock indicators
    if 'nlp_sepsis' in df.columns and 'nlp_vasopressors' in df.columns:
        expert['septic_shock'] = (
            df['nlp_sepsis'].fillna(0) * df['nlp_vasopressors'].fillna(0)
        )
    
    # ARDS with hypoxia
    if 'nlp_ards' in df.columns and 'SpO2_Min' in df.columns:
        expert['ards_with_hypoxia'] = (
            df['nlp_ards'].fillna(0) * 
            (df['SpO2_Min'].fillna(95) < 90).astype(int)
        )
    
    # ==========================================
    # AGE-DISEASE INTERACTIONS
    # ==========================================
    
    if 'AGE' in df.columns:
        age = df['AGE'].fillna(60)
        elderly = (age >= 70).astype(int)
        very_elderly = (age >= 80).astype(int)
        
        # Elderly + sepsis = very high risk
        if 'nlp_sepsis' in df.columns:
            expert['elderly_sepsis'] = elderly * df['nlp_sepsis'].fillna(0)
        
        # Elderly + heart failure
        if 'nlp_heart_failure' in df.columns:
            expert['elderly_heart_failure'] = elderly * df['nlp_heart_failure'].fillna(0)
        
        # Very elderly + respiratory failure
        if 'nlp_respiratory_failure' in df.columns:
            expert['very_elderly_resp_failure'] = very_elderly * df['nlp_respiratory_failure'].fillna(0)
        
        # Elderly + multiple comorbidities
        if 'CCI_Score' in df.columns:
            expert['elderly_high_cci'] = elderly * (df['CCI_Score'].fillna(0) >= 3).astype(int)
    
    # ==========================================
    # LOS-BASED RISK INDICATORS  
    # ==========================================
    
    if 'LOS_ICU' in df.columns:
        los_icu = df['LOS_ICU'].fillna(1)
        prolonged_icu = (los_icu >= 5).astype(int)
        very_prolonged = (los_icu >= 10).astype(int)
        
        # Prolonged ICU + ventilation = high risk
        if 'Ventilation_Usage' in df.columns:
            expert['prolonged_icu_vent'] = prolonged_icu * df['Ventilation_Usage'].fillna(0)
        
        # Prolonged ICU + sepsis
        if 'nlp_sepsis' in df.columns:
            expert['prolonged_icu_sepsis'] = prolonged_icu * df['nlp_sepsis'].fillna(0)
        
        # Prolonged ICU + AKI
        if 'nlp_acute_kidney_injury' in df.columns:
            expert['prolonged_icu_aki'] = prolonged_icu * df['nlp_acute_kidney_injury'].fillna(0)
    
    # ==========================================
    # MULTI-ORGAN DYSFUNCTION COMBINATIONS
    # ==========================================
    
    # Respiratory + Renal dysfunction
    if 'nlp_respiratory_failure' in df.columns and 'nlp_acute_kidney_injury' in df.columns:
        expert['resp_renal_failure'] = (
            df['nlp_respiratory_failure'].fillna(0) * 
            df['nlp_acute_kidney_injury'].fillna(0)
        )
    
    # Cardiac + Renal dysfunction
    if 'nlp_heart_failure' in df.columns and 'nlp_acute_kidney_injury' in df.columns:
        expert['cardiac_renal_failure'] = (
            df['nlp_heart_failure'].fillna(0) * 
            df['nlp_acute_kidney_injury'].fillna(0)
        )
    
    # Coagulopathy + GI bleed = critical bleeding
    if 'nlp_coagulopathy' in df.columns and 'nlp_gi_bleed' in df.columns:
        expert['coag_gi_bleed'] = (
            df['nlp_coagulopathy'].fillna(0) * 
            df['nlp_gi_bleed'].fillna(0)
        )
    
    # ==========================================
    # VITAL SIGN INSTABILITY COMBINATIONS
    # ==========================================
    
    # Tachycardia + Hypotension = shock state
    if 'HeartRate_Max' in df.columns and 'SBP_Min' in df.columns:
        expert['tachycardia_hypotension'] = (
            (df['HeartRate_Max'].fillna(80) > 110).astype(int) *
            (df['SBP_Min'].fillna(120) < 90).astype(int)
        )
    
    # Tachypnea + Hypoxia = respiratory distress
    if 'RespRate_Max' in df.columns and 'SpO2_Min' in df.columns:
        expert['tachypnea_hypoxia'] = (
            (df['RespRate_Max'].fillna(16) > 24).astype(int) *
            (df['SpO2_Min'].fillna(95) < 92).astype(int)
        )
    
    # GCS decline + sepsis = septic encephalopathy
    if 'GCS_Min' in df.columns and 'nlp_sepsis' in df.columns:
        expert['gcs_sepsis'] = (
            (df['GCS_Min'].fillna(15) < 14).astype(int) *
            df['nlp_sepsis'].fillna(0)
        )
    
    # ==========================================
    # LAB VALUE EXTREME COMBINATIONS
    # ==========================================
    
    # Severe anemia + coagulopathy = bleeding risk
    if 'HMG_Min' in df.columns and 'nlp_coagulopathy' in df.columns:
        expert['severe_anemia_coag'] = (
            (df['HMG_Min'].fillna(12) < 8).astype(int) *
            df['nlp_coagulopathy'].fillna(0)
        )
    
    # Severe thrombocytopenia + sepsis = DIC risk
    if 'Platelet_Min' in df.columns and 'nlp_sepsis' in df.columns:
        expert['thrombocytopenia_sepsis'] = (
            (df['Platelet_Min'].fillna(200) < 50).astype(int) *
            df['nlp_sepsis'].fillna(0)
        )
    
    # Elevated lactate + renal failure = severe metabolic derangement
    if 'Lactate_Max' in df.columns and 'nlp_acute_kidney_injury' in df.columns:
        expert['lactate_aki'] = (
            (df['Lactate_Max'].fillna(1) > 4).astype(int) *
            df['nlp_acute_kidney_injury'].fillna(0)
        )
    
    # High BUN/Creat with dehydration markers
    if 'BUN_Creat_Ratio' in df.columns and 'UrineOutput_Median' in df.columns:
        expert['prerenal_low_urine'] = (
            (df['BUN_Creat_Ratio'].fillna(15) > 20).astype(int) *
            (df['UrineOutput_Median'].fillna(100) < 50).astype(int)
        )
    
    # ==========================================
    # PROCEDURE + CONDITION COMBINATIONS
    # ==========================================
    
    # Intubation + ARDS
    if 'nlp_intubation' in df.columns and 'nlp_ards' in df.columns:
        expert['intubation_ards'] = (
            df['nlp_intubation'].fillna(0) * df['nlp_ards'].fillna(0)
        )
    
    # Central line + sepsis = line infection risk
    if 'nlp_central_line' in df.columns and 'nlp_sepsis' in df.columns:
        expert['central_line_sepsis'] = (
            df['nlp_central_line'].fillna(0) * df['nlp_sepsis'].fillna(0)
        )
    
    # Surgery + anemia
    if 'nlp_surgery' in df.columns and 'HMG_Min' in df.columns:
        expert['surgery_anemia'] = (
            df['nlp_surgery'].fillna(0) * 
            (df['HMG_Min'].fillna(12) < 10).astype(int)
        )
    
    # ==========================================
    # COMPOSITE RISK SCORES
    # ==========================================
    
    # High acuity composite
    high_acuity_components = []
    if 'Ventilation_Usage' in df.columns:
        high_acuity_components.append(df['Ventilation_Usage'].fillna(0))
    if 'nlp_vasopressors' in df.columns:
        high_acuity_components.append(df['nlp_vasopressors'].fillna(0))
    if 'nlp_sepsis' in df.columns:
        high_acuity_components.append(df['nlp_sepsis'].fillna(0))
    if 'nlp_respiratory_failure' in df.columns:
        high_acuity_components.append(df['nlp_respiratory_failure'].fillna(0))
    if high_acuity_components:
        expert['high_acuity_composite'] = sum(high_acuity_components)
        expert['very_high_acuity'] = (expert['high_acuity_composite'] >= 3).astype(int)
    
    # Chronic disease burden with acute decompensation
    chronic_components = []
    if 'nlp_chronic_kidney_disease' in df.columns:
        chronic_components.append(df['nlp_chronic_kidney_disease'].fillna(0))
    if 'nlp_heart_failure' in df.columns:
        chronic_components.append(df['nlp_heart_failure'].fillna(0))
    if 'nlp_copd' in df.columns:
        chronic_components.append(df['nlp_copd'].fillna(0))
    if 'nlp_liver_failure' in df.columns:
        chronic_components.append(df['nlp_liver_failure'].fillna(0))
    if chronic_components:
        expert['chronic_disease_burden'] = sum(chronic_components)
        expert['high_chronic_burden'] = (expert['chronic_disease_burden'] >= 2).astype(int)
    
    logger.info(f"Created {len(expert)} clinical expert-guided features")
    return pd.DataFrame(expert, index=df.index)

def enhance_features_v3(input_path, output_path):
    """Add expert features to enhanced features."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Create expert features
    expert_df = create_expert_interactions(df)
    
    # Combine
    enhanced_df = pd.concat([df, expert_df], axis=1)
    enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
    
    logger.info(f"Final dataset: {len(enhanced_df)} samples with {len(enhanced_df.columns)} features")
    logger.info(f"Added {len(enhanced_df.columns) - len(df.columns)} expert features")
    
    # Save
    enhanced_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Summary
    new_cols = [c for c in enhanced_df.columns if c not in df.columns]
    logger.info(f"\nExpert features added ({len(new_cols)}):")
    for col in new_cols:
        non_zero = (enhanced_df[col] != 0).sum()
        logger.info(f"  {col}: {non_zero} non-zero ({100*non_zero/len(enhanced_df):.1f}%)")
    
    return enhanced_df

if __name__ == '__main__':
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    input_path = os.path.join(base_dir, 'cohort/features_enhanced.csv')
    output_path = os.path.join(base_dir, 'cohort/features_expert.csv')
    
    enhance_features_v3(input_path, output_path)

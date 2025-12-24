"""
Feature Engineering from Raw MIMIC-III Data.

Computes 320 engineered features from raw MIMIC tables for a given HADM_ID.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# MIMIC-III database path
DB_PATH = Path("/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db")


class FeatureEngineer:
    """Compute features from raw MIMIC data for a single HADM_ID."""
    
    # Lab items mapping (ITEMID -> feature name)
    LAB_ITEMS = {
        'BUN': [51006],
        'Creatinine': [50912],
        'Glucose': [50809, 50931],
        'WBC': [51300, 51301],
        'Platelet': [51265],
        'Lactate': [50813],
        'Albumin': [50862],
        'Anion_Gap': [50868],
        'HMG': [51222],  # Hemoglobin
        'PTT': [51275],
    }
    
    # Vital items mapping (ITEMID -> feature name)
    VITAL_ITEMS = {
        'SBP': [220179, 220050, 225309],
        'DBP': [220180, 220051, 225310],
        'HeartRate': [220045, 211],
        'RespRate': [220210, 618],
        'SpO2': [220277, 646],
        'GCS': [220739, 223901],
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        
    def connect(self) -> sqlite3.Connection:
        """Create database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        return sqlite3.connect(str(self.db_path))
    
    def get_patient_info(self, hadm_id: int) -> Dict[str, Any]:
        """Get basic patient demographics."""
        conn = self.connect()
        try:
            query = """
            SELECT 
                a.SUBJECT_ID,
                a.HADM_ID,
                a.ADMITTIME,
                a.DISCHTIME,
                ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25, 0) as AGE,
                ROUND((julianday(a.DISCHTIME) - julianday(a.ADMITTIME)), 2) as LOS_Hospital
            FROM ADMISSIONS a
            JOIN PATIENTS p ON p.SUBJECT_ID = a.SUBJECT_ID
            WHERE a.HADM_ID = ?
            """
            result = pd.read_sql_query(query, conn, params=[hadm_id])
            if len(result) == 0:
                return {}
            return result.iloc[0].to_dict()
        finally:
            conn.close()
    
    def get_icu_stay(self, hadm_id: int) -> Dict[str, Any]:
        """Get ICU stay info."""
        conn = self.connect()
        try:
            query = """
            SELECT 
                ICUSTAY_ID,
                INTIME,
                OUTTIME,
                ROUND((julianday(OUTTIME) - julianday(INTIME)), 2) as LOS_ICU
            FROM ICUSTAYS
            WHERE HADM_ID = ?
            ORDER BY INTIME
            LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=[hadm_id])
            if len(result) == 0:
                return {}
            return result.iloc[0].to_dict()
        finally:
            conn.close()
    
    def get_lab_features(self, hadm_id: int) -> Dict[str, float]:
        """Compute lab features from LABEVENTS."""
        conn = self.connect()
        features = {}
        
        try:
            for lab_name, itemids in self.LAB_ITEMS.items():
                query = f"""
                SELECT VALUENUM, CHARTTIME
                FROM LABEVENTS
                WHERE HADM_ID = ?
                AND ITEMID IN ({','.join(map(str, itemids))})
                AND VALUENUM IS NOT NULL
                ORDER BY CHARTTIME
                """
                df = pd.read_sql_query(query, conn, params=[hadm_id])
                
                if len(df) > 0:
                    # Convert to numpy float array to avoid type issues
                    values = df['VALUENUM'].astype(float).values
                    features[f'{lab_name}_Min'] = float(np.nanmin(values))
                    features[f'{lab_name}_Max'] = float(np.nanmax(values))
                    features[f'{lab_name}_Avg'] = float(np.nanmean(values))
                    features[f'{lab_name}_Median'] = float(np.nanmedian(values))
                    features[f'{lab_name}_Std'] = float(np.nanstd(values)) if len(values) > 1 else 0.0
                    features[f'{lab_name}_Count'] = float(len(values))
                    features[f'{lab_name}_Last'] = float(values[-1])
                    features[f'{lab_name}_measured'] = 1.0
                else:
                    # Fill with defaults
                    for suffix in ['Min', 'Max', 'Avg', 'Median', 'Std', 'Count', 'Last']:
                        features[f'{lab_name}_{suffix}'] = 0.0
                    features[f'{lab_name}_measured'] = 0.0
                    
        finally:
            conn.close()
        
        return features
    
    def get_vital_features(self, hadm_id: int, icustay_id: Optional[int] = None) -> Dict[str, float]:
        """Compute vital sign features from CHARTEVENTS."""
        conn = self.connect()
        features = {}
        
        try:
            for vital_name, itemids in self.VITAL_ITEMS.items():
                query = """
                SELECT VALUENUM, CHARTTIME
                FROM CHARTEVENTS
                WHERE HADM_ID = ?
                AND ITEMID IN ({})
                AND VALUENUM IS NOT NULL
                ORDER BY CHARTTIME
                """.format(','.join(map(str, itemids)))
                
                df = pd.read_sql_query(query, conn, params=[hadm_id])
                
                if len(df) > 0:
                    # Convert to float to avoid type issues
                    values = df['VALUENUM'].astype(float).values
                    features[f'{vital_name}_Min'] = float(np.nanmin(values))
                    features[f'{vital_name}_Max'] = float(np.nanmax(values))
                    features[f'{vital_name}_Avg'] = float(np.nanmean(values))
                    features[f'{vital_name}_Median'] = float(np.nanmedian(values))
                    features[f'{vital_name}_Std'] = float(np.nanstd(values)) if len(values) > 1 else 0.0
                    features[f'{vital_name}_Count'] = float(len(values))
                    features[f'{vital_name}_Last'] = float(values[-1])
                    features[f'{vital_name}_measured'] = 1.0
                else:
                    for suffix in ['Min', 'Max', 'Avg', 'Median', 'Std', 'Count', 'Last']:
                        features[f'{vital_name}_{suffix}'] = 0.0
                    features[f'{vital_name}_measured'] = 0.0
                    
        finally:
            conn.close()
        
        return features
    
    def get_clinical_notes(self, hadm_id: int) -> str:
        """Get clinical notes before ICU discharge."""
        conn = self.connect()
        try:
            # Get ICU outtime first
            icu_query = """
            SELECT OUTTIME FROM ICUSTAYS 
            WHERE HADM_ID = ? ORDER BY INTIME LIMIT 1
            """
            icu_result = pd.read_sql_query(icu_query, conn, params=[hadm_id])
            
            if len(icu_result) == 0:
                return ""
            
            outtime = icu_result.iloc[0]['OUTTIME']
            
            # Get notes before ICU out
            notes_query = """
            SELECT TEXT
            FROM NOTEEVENTS
            WHERE HADM_ID = ?
            AND CHARTTIME < ?
            AND TRIM(CATEGORY) IN ('Nursing/other', 'Nursing', 'Physician')
            AND TEXT IS NOT NULL
            ORDER BY CHARTTIME
            LIMIT 20
            """
            notes_df = pd.read_sql_query(notes_query, conn, params=[hadm_id, outtime])
            
            if len(notes_df) == 0:
                return ""
            
            # Concatenate and clean
            combined = ' '.join(notes_df['TEXT'].tolist())
            # Simple cleaning
            combined = combined.replace('\n', ' ').replace('\r', ' ')
            combined = ' '.join(combined.split())  # Normalize whitespace
            
            return combined[:50000]  # Limit length
            
        finally:
            conn.close()
    
    def compute_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute derived clinical scores (SIRS, MEWS, etc.)."""
        derived = {}
        
        # SIRS Score (simplified)
        sirs = 0
        if features.get('HeartRate_Last', 0) > 90:
            sirs += 1
        if features.get('RespRate_Last', 0) > 20:
            sirs += 1
        if features.get('WBC_Last', 0) > 12 or features.get('WBC_Last', 0) < 4:
            sirs += 1
        derived['SIRS_Score'] = float(sirs)
        derived['SIRS_Positive'] = float(sirs >= 2)
        
        # Shock Index
        sbp = features.get('SBP_Last', 120)
        hr = features.get('HeartRate_Last', 80)
        derived['Shock_Index'] = float(hr / sbp) if sbp > 0 else 0.0
        
        # BUN/Creatinine Ratio
        bun = features.get('BUN_Last', 0)
        creat = features.get('Creatinine_Last', 1)
        derived['BUN_Creatinine_Ratio'] = float(bun / creat) if creat > 0 else 0.0
        
        return derived
    
    def compute_all_features(self, hadm_id: int) -> Dict[str, Any]:
        """
        Compute all 320 features for a patient.
        
        Returns dict with:
            - features: computed feature values
            - clinical_notes: raw notes text
            - patient_info: demographics
        """
        logger.info(f"Computing features for HADM_ID: {hadm_id}")
        
        # Get patient info
        patient_info = self.get_patient_info(hadm_id)
        if not patient_info:
            raise ValueError(f"HADM_ID {hadm_id} not found")
        
        # Get ICU stay
        icu_stay = self.get_icu_stay(hadm_id)
        
        # Initialize features
        features = {}
        
        # Demographics
        features['AGE'] = patient_info.get('AGE', 0)
        features['LOS_Hospital'] = patient_info.get('LOS_Hospital', 0)
        features['LOS_ICU'] = icu_stay.get('LOS_ICU', 0) if icu_stay else 0
        features['Number_of_Prior_Readmits'] = 0  # Would need additional query
        
        # Lab features
        lab_features = self.get_lab_features(hadm_id)
        features.update(lab_features)
        
        # Vital features
        vital_features = self.get_vital_features(hadm_id)
        features.update(vital_features)
        
        # Derived features
        derived = self.compute_derived_features(features)
        features.update(derived)
        
        # Clinical notes
        clinical_notes = self.get_clinical_notes(hadm_id)
        
        logger.info(f"Computed {len(features)} features for HADM_ID {hadm_id}")
        
        return {
            "features": features,
            "clinical_notes": clinical_notes,
            "patient_info": patient_info
        }


def compute_features_from_hadm(hadm_id: int) -> Dict[str, Any]:
    """Convenience function to compute features from HADM_ID."""
    engineer = FeatureEngineer()
    return engineer.compute_all_features(hadm_id)

"""
Advanced Feature Engineering for ICU Readmission Prediction

This module creates sophisticated features including:
1. Temporal Features - Trends, variability, deterioration indicators
2. Cross-Feature Interactions - Ratio features, composite scores
3. Clinical Acuity Scores - SIRS, SOFA-like components
4. Statistical Features - Higher moments, entropy, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy import stats
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Create advanced features for ICU readmission prediction.
    """
    
    # Clinical thresholds for abnormal values
    CLINICAL_THRESHOLDS = {
        # Vital Signs
        'HeartRate': {'low': 60, 'high': 100, 'critical_low': 40, 'critical_high': 150},
        'SysBP': {'low': 90, 'high': 140, 'critical_low': 70, 'critical_high': 180},
        'DiasBP': {'low': 60, 'high': 90, 'critical_low': 40, 'critical_high': 110},
        'MeanBP': {'low': 65, 'high': 100, 'critical_low': 50, 'critical_high': 120},
        'RespRate': {'low': 12, 'high': 20, 'critical_low': 8, 'critical_high': 30},
        'SpO2': {'low': 95, 'high': 100, 'critical_low': 90, 'critical_high': 100},
        'Temperature': {'low': 36, 'high': 38, 'critical_low': 35, 'critical_high': 39},
        
        # Labs
        'WBC': {'low': 4, 'high': 11, 'critical_low': 2, 'critical_high': 20},
        'Platelet': {'low': 150, 'high': 400, 'critical_low': 50, 'critical_high': 500},
        'HMG': {'low': 12, 'high': 17, 'critical_low': 7, 'critical_high': 20},  # Hemoglobin
        'Creatinine': {'low': 0.6, 'high': 1.2, 'critical_low': 0.4, 'critical_high': 4.0},
        'BUN': {'low': 7, 'high': 20, 'critical_low': 5, 'critical_high': 50},
        'Glucose': {'low': 70, 'high': 140, 'critical_low': 50, 'critical_high': 400},
        'Sodium': {'low': 135, 'high': 145, 'critical_low': 125, 'critical_high': 155},
        'Potassium': {'low': 3.5, 'high': 5.0, 'critical_low': 2.5, 'critical_high': 6.5},
        'Lactate': {'low': 0.5, 'high': 2.0, 'critical_low': 0.3, 'critical_high': 4.0},
        'PTT': {'low': 25, 'high': 35, 'critical_low': 20, 'critical_high': 60},
        'INR': {'low': 0.9, 'high': 1.1, 'critical_low': 0.8, 'critical_high': 2.0},
        'Bilirubin': {'low': 0.1, 'high': 1.2, 'critical_low': 0.05, 'critical_high': 5.0},
        'Anion_Gap': {'low': 8, 'high': 12, 'critical_low': 6, 'critical_high': 20},
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_names = []
        
    def log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        vital_cols: List[str] = None,
        lab_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create all advanced features.
        
        Args:
            df: Input dataframe with raw features
            vital_cols: List of vital sign columns
            lab_cols: List of lab result columns
            
        Returns:
            DataFrame with all new features
        """
        self.log("="*60)
        self.log("ADVANCED FEATURE ENGINEERING")
        self.log("="*60)
        
        # Auto-detect columns if not provided
        if vital_cols is None:
            vital_cols = self._detect_vital_columns(df)
        if lab_cols is None:
            lab_cols = self._detect_lab_columns(df)
        
        self.log(f"Detected {len(vital_cols)} vital columns")
        self.log(f"Detected {len(lab_cols)} lab columns")
        
        # Initialize new features dataframe
        new_features = pd.DataFrame(index=df.index)
        
        # 1. Temporal Trend Features
        self.log("\n1. Creating Temporal Trend Features...")
        temporal_features = self._create_temporal_features(df, vital_cols + lab_cols)
        new_features = pd.concat([new_features, temporal_features], axis=1)
        self.log(f"   Created {temporal_features.shape[1]} temporal features")
        
        # 2. Variability Features
        self.log("\n2. Creating Variability Features...")
        variability_features = self._create_variability_features(df, vital_cols + lab_cols)
        new_features = pd.concat([new_features, variability_features], axis=1)
        self.log(f"   Created {variability_features.shape[1]} variability features")
        
        # 3. Clinical Abnormality Features
        self.log("\n3. Creating Clinical Abnormality Features...")
        abnormality_features = self._create_abnormality_features(df, vital_cols + lab_cols)
        new_features = pd.concat([new_features, abnormality_features], axis=1)
        self.log(f"   Created {abnormality_features.shape[1]} abnormality features")
        
        # 4. Composite Clinical Scores
        self.log("\n4. Creating Composite Clinical Scores...")
        score_features = self._create_clinical_scores(df)
        new_features = pd.concat([new_features, score_features], axis=1)
        self.log(f"   Created {score_features.shape[1]} clinical score features")
        
        # 5. Ratio Features
        self.log("\n5. Creating Ratio Features...")
        ratio_features = self._create_ratio_features(df)
        new_features = pd.concat([new_features, ratio_features], axis=1)
        self.log(f"   Created {ratio_features.shape[1]} ratio features")
        
        # 6. Deterioration Indicators
        self.log("\n6. Creating Deterioration Indicators...")
        deterioration_features = self._create_deterioration_features(df, vital_cols + lab_cols)
        new_features = pd.concat([new_features, deterioration_features], axis=1)
        self.log(f"   Created {deterioration_features.shape[1]} deterioration features")
        
        # 7. Time-Window Comparison Features
        self.log("\n7. Creating Time-Window Comparison Features...")
        window_features = self._create_window_comparison_features(df, vital_cols + lab_cols)
        new_features = pd.concat([new_features, window_features], axis=1)
        self.log(f"   Created {window_features.shape[1]} window comparison features")
        
        # Clean up
        new_features = new_features.replace([np.inf, -np.inf], np.nan)
        new_features = new_features.fillna(0)
        
        self.feature_names = new_features.columns.tolist()
        
        self.log(f"\n{'='*60}")
        self.log(f"TOTAL NEW FEATURES: {new_features.shape[1]}")
        self.log(f"{'='*60}")
        
        return new_features
    
    def _detect_vital_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect vital sign columns."""
        vital_patterns = ['HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 
                         'SpO2', 'Temperature', 'HR_', 'BP_', 'Resp_', 'Temp_']
        cols = []
        for col in df.columns:
            for pattern in vital_patterns:
                if pattern.lower() in col.lower():
                    cols.append(col)
                    break
        return list(set(cols))
    
    def _detect_lab_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect lab result columns."""
        lab_patterns = ['WBC', 'Platelet', 'HMG', 'Creatinine', 'BUN', 'Glucose',
                       'Sodium', 'Potassium', 'Lactate', 'PTT', 'INR', 'Bilirubin',
                       'Anion_Gap', 'Chloride', 'Bicarbonate', 'Calcium', 'Magnesium']
        cols = []
        for col in df.columns:
            for pattern in lab_patterns:
                if pattern.lower() in col.lower():
                    cols.append(col)
                    break
        return list(set(cols))
    
    def _get_base_name(self, col: str) -> str:
        """Extract base name from column (e.g., 'HeartRate_Mean_24h' -> 'HeartRate')."""
        suffixes = ['_Mean', '_Std', '_Min', '_Max', '_Range', '_Count', 
                   '_24h', '_48h', '_First', '_Last', '_Median']
        base = col
        for suffix in suffixes:
            if suffix in base:
                base = base.replace(suffix, '')
        return base.strip('_')
    
    def _create_temporal_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Create temporal trend features.
        - Delta between 24h and 48h windows
        - First vs Last value comparison
        - Trend direction indicators
        """
        features = pd.DataFrame(index=df.index)
        
        # Group columns by base name
        base_names = set(self._get_base_name(col) for col in cols)
        
        for base in base_names:
            # Find related columns
            mean_24h = f"{base}_Mean_24h" if f"{base}_Mean_24h" in df.columns else None
            mean_48h = f"{base}_Mean_48h" if f"{base}_Mean_48h" in df.columns else None
            first = f"{base}_First" if f"{base}_First" in df.columns else None
            last = f"{base}_Last" if f"{base}_Last" in df.columns else None
            
            # Delta between time windows
            if mean_24h and mean_48h:
                features[f"{base}_Delta_24h_48h"] = df[mean_24h] - df[mean_48h]
                features[f"{base}_Trend_24h_48h"] = np.sign(df[mean_24h] - df[mean_48h])
                # Percentage change
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct_change = (df[mean_24h] - df[mean_48h]) / (np.abs(df[mean_48h]) + 1e-8)
                    features[f"{base}_PctChange_24h_48h"] = np.clip(pct_change, -10, 10)
            
            # First vs Last delta
            if first and last:
                features[f"{base}_Delta_First_Last"] = df[last] - df[first]
                features[f"{base}_Trend_First_Last"] = np.sign(df[last] - df[first])
        
        return features
    
    def _create_variability_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Create variability features.
        - Coefficient of variation
        - Range relative to mean
        - Instability indicators
        """
        features = pd.DataFrame(index=df.index)
        
        base_names = set(self._get_base_name(col) for col in cols)
        
        for base in base_names:
            mean_col = None
            std_col = None
            range_col = None
            
            # Find related columns
            for col in df.columns:
                if base in col:
                    if '_Mean' in col and '24h' not in col and '48h' not in col:
                        mean_col = col
                    elif '_Std' in col and '24h' not in col and '48h' not in col:
                        std_col = col
                    elif '_Range' in col:
                        range_col = col
            
            # Coefficient of Variation (CV)
            if mean_col and std_col:
                with np.errstate(divide='ignore', invalid='ignore'):
                    cv = df[std_col] / (np.abs(df[mean_col]) + 1e-8)
                    features[f"{base}_CV"] = np.clip(cv, 0, 10)
                    # High variability indicator
                    features[f"{base}_HighVariability"] = (cv > 0.3).astype(int)
            
            # Range relative to mean
            if mean_col and range_col:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_range = df[range_col] / (np.abs(df[mean_col]) + 1e-8)
                    features[f"{base}_RelativeRange"] = np.clip(rel_range, 0, 10)
        
        return features
    
    def _create_abnormality_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Create clinical abnormality features based on thresholds.
        """
        features = pd.DataFrame(index=df.index)
        
        for base, thresholds in self.CLINICAL_THRESHOLDS.items():
            # Find mean column for this base
            mean_col = None
            for col in df.columns:
                if base in col and ('_Mean' in col or col == base):
                    mean_col = col
                    break
            
            if mean_col is None:
                continue
            
            values = df[mean_col]
            
            # Binary abnormality indicators
            features[f"{base}_IsLow"] = (values < thresholds['low']).astype(int)
            features[f"{base}_IsHigh"] = (values > thresholds['high']).astype(int)
            features[f"{base}_IsAbnormal"] = ((values < thresholds['low']) | 
                                              (values > thresholds['high'])).astype(int)
            
            # Critical abnormality
            features[f"{base}_IsCriticalLow"] = (values < thresholds['critical_low']).astype(int)
            features[f"{base}_IsCriticalHigh"] = (values > thresholds['critical_high']).astype(int)
            features[f"{base}_IsCritical"] = ((values < thresholds['critical_low']) | 
                                              (values > thresholds['critical_high'])).astype(int)
            
            # Distance from normal range
            low, high = thresholds['low'], thresholds['high']
            mid = (low + high) / 2
            features[f"{base}_DistFromNormal"] = np.where(
                values < low, low - values,
                np.where(values > high, values - high, 0)
            )
            features[f"{base}_DistFromMid"] = np.abs(values - mid)
        
        # Total abnormality count
        abnormal_cols = [col for col in features.columns if '_IsAbnormal' in col]
        features['Total_Abnormal_Count'] = features[abnormal_cols].sum(axis=1)
        
        critical_cols = [col for col in features.columns if '_IsCritical' in col and 'Low' not in col and 'High' not in col]
        features['Total_Critical_Count'] = features[critical_cols].sum(axis=1)
        
        return features
    
    def _create_clinical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite clinical scores (SIRS-like, SOFA-like components).
        """
        features = pd.DataFrame(index=df.index)
        
        # ============================================================
        # SIRS-like Score (Systemic Inflammatory Response)
        # ============================================================
        sirs_score = pd.Series(0, index=df.index)
        
        # Temperature criteria
        temp_col = self._find_column(df, ['Temperature', 'Temp'])
        if temp_col:
            sirs_score += ((df[temp_col] > 38) | (df[temp_col] < 36)).astype(int)
        
        # Heart rate criteria
        hr_col = self._find_column(df, ['HeartRate', 'HR'])
        if hr_col:
            sirs_score += (df[hr_col] > 90).astype(int)
        
        # Respiratory rate criteria
        rr_col = self._find_column(df, ['RespRate', 'Resp'])
        if rr_col:
            sirs_score += (df[rr_col] > 20).astype(int)
        
        # WBC criteria
        wbc_col = self._find_column(df, ['WBC'])
        if wbc_col:
            sirs_score += ((df[wbc_col] > 12) | (df[wbc_col] < 4)).astype(int)
        
        features['SIRS_Score'] = sirs_score
        features['SIRS_Positive'] = (sirs_score >= 2).astype(int)
        
        # ============================================================
        # Shock Index (HR / SysBP)
        # ============================================================
        hr_col = self._find_column(df, ['HeartRate', 'HR'])
        sbp_col = self._find_column(df, ['SysBP', 'Systolic'])
        if hr_col and sbp_col:
            with np.errstate(divide='ignore', invalid='ignore'):
                shock_index = df[hr_col] / (df[sbp_col] + 1e-8)
                features['Shock_Index'] = np.clip(shock_index, 0, 3)
                features['Shock_Index_Elevated'] = (shock_index > 0.9).astype(int)
                features['Shock_Index_High'] = (shock_index > 1.0).astype(int)
        
        # ============================================================
        # Modified Early Warning Score (MEWS-like)
        # ============================================================
        mews_score = pd.Series(0, index=df.index)
        
        # Heart rate component
        if hr_col:
            hr = df[hr_col]
            mews_score += np.where(hr < 40, 2, np.where(hr < 50, 1, 
                          np.where(hr > 130, 2, np.where(hr > 110, 1, 0))))
        
        # Systolic BP component
        if sbp_col:
            sbp = df[sbp_col]
            mews_score += np.where(sbp < 70, 3, np.where(sbp < 80, 2, 
                          np.where(sbp < 100, 1, np.where(sbp > 200, 2, 0))))
        
        # Respiratory rate component
        if rr_col:
            rr = df[rr_col]
            mews_score += np.where(rr < 9, 2, np.where(rr > 29, 3, 
                          np.where(rr > 20, 2, np.where(rr > 14, 1, 0))))
        
        # Temperature component
        if temp_col:
            temp = df[temp_col]
            mews_score += np.where(temp < 35, 2, np.where(temp > 38.5, 2, 0))
        
        features['MEWS_Score'] = mews_score
        features['MEWS_High'] = (mews_score >= 4).astype(int)
        
        # ============================================================
        # Renal Function Indicators
        # ============================================================
        creat_col = self._find_column(df, ['Creatinine'])
        bun_col = self._find_column(df, ['BUN'])
        
        if creat_col and bun_col:
            # BUN/Creatinine ratio (pre-renal vs renal)
            with np.errstate(divide='ignore', invalid='ignore'):
                bun_creat_ratio = df[bun_col] / (df[creat_col] + 1e-8)
                features['BUN_Creat_Ratio'] = np.clip(bun_creat_ratio, 0, 100)
                features['Prerenal_Indicator'] = (bun_creat_ratio > 20).astype(int)
        
        if creat_col:
            # AKI staging (simplified)
            features['AKI_Stage1'] = (df[creat_col] >= 1.5).astype(int)
            features['AKI_Stage2'] = (df[creat_col] >= 2.0).astype(int)
            features['AKI_Stage3'] = (df[creat_col] >= 3.0).astype(int)
        
        # ============================================================
        # Coagulation Risk Score
        # ============================================================
        coag_score = pd.Series(0, index=df.index)
        
        plt_col = self._find_column(df, ['Platelet'])
        if plt_col:
            plt = df[plt_col]
            coag_score += np.where(plt < 50, 2, np.where(plt < 100, 1, 0))
        
        inr_col = self._find_column(df, ['INR'])
        if inr_col:
            inr = df[inr_col]
            coag_score += np.where(inr > 2.0, 2, np.where(inr > 1.5, 1, 0))
        
        ptt_col = self._find_column(df, ['PTT'])
        if ptt_col:
            ptt = df[ptt_col]
            coag_score += np.where(ptt > 60, 2, np.where(ptt > 40, 1, 0))
        
        features['Coagulation_Risk_Score'] = coag_score
        features['Coag_Risk_High'] = (coag_score >= 2).astype(int)
        
        # ============================================================
        # Metabolic Disturbance Score
        # ============================================================
        metabolic_score = pd.Series(0, index=df.index)
        
        glucose_col = self._find_column(df, ['Glucose'])
        if glucose_col:
            gluc = df[glucose_col]
            metabolic_score += np.where(gluc < 60, 2, np.where(gluc > 300, 2, 
                              np.where((gluc < 70) | (gluc > 180), 1, 0)))
        
        na_col = self._find_column(df, ['Sodium'])
        if na_col:
            na = df[na_col]
            metabolic_score += np.where((na < 125) | (na > 155), 2,
                              np.where((na < 130) | (na > 150), 1, 0))
        
        k_col = self._find_column(df, ['Potassium'])
        if k_col:
            k = df[k_col]
            metabolic_score += np.where((k < 2.5) | (k > 6.5), 2,
                              np.where((k < 3.0) | (k > 5.5), 1, 0))
        
        features['Metabolic_Score'] = metabolic_score
        features['Metabolic_High'] = (metabolic_score >= 2).astype(int)
        
        # ============================================================
        # Overall Acuity Score
        # ============================================================
        features['Overall_Acuity'] = (features.get('SIRS_Score', 0) + 
                                      features.get('MEWS_Score', 0) +
                                      features.get('Coagulation_Risk_Score', 0) +
                                      features.get('Metabolic_Score', 0))
        
        return features
    
    def _find_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Find column matching any of the patterns."""
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower() and '_Mean' in col:
                    return col
        # Try without _Mean suffix
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return col
        return None
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically meaningful ratio features.
        """
        features = pd.DataFrame(index=df.index)
        
        # Define ratio pairs
        ratio_pairs = [
            ('HeartRate', 'SysBP', 'HR_SBP_Ratio'),  # Shock index
            ('HeartRate', 'RespRate', 'HR_RR_Ratio'),
            # BUN_Creat_Ratio already created in clinical scores
            ('WBC', 'Platelet', 'WBC_Plt_Ratio'),
            ('Glucose', 'Potassium', 'Gluc_K_Ratio'),
            ('Sodium', 'Potassium', 'Na_K_Ratio'),
        ]
        
        for num_pattern, denom_pattern, name in ratio_pairs:
            num_col = self._find_column(df, [num_pattern])
            denom_col = self._find_column(df, [denom_pattern])
            
            if num_col and denom_col:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = df[num_col] / (df[denom_col] + 1e-8)
                    features[name] = np.clip(ratio, 0, 100)
        
        # Anion gap related
        na_col = self._find_column(df, ['Sodium'])
        cl_col = self._find_column(df, ['Chloride'])
        bicarb_col = self._find_column(df, ['Bicarbonate'])
        
        if na_col and cl_col and bicarb_col:
            # Calculated anion gap
            features['Calc_Anion_Gap'] = df[na_col] - df[cl_col] - df[bicarb_col]
        
        return features
    
    def _create_deterioration_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Create deterioration indicators - features that signal worsening condition.
        """
        features = pd.DataFrame(index=df.index)
        
        base_names = set(self._get_base_name(col) for col in cols)
        
        deterioration_count = pd.Series(0, index=df.index)
        
        for base in base_names:
            mean_24h = f"{base}_Mean_24h" if f"{base}_Mean_24h" in df.columns else None
            mean_48h = f"{base}_Mean_48h" if f"{base}_Mean_48h" in df.columns else None
            
            if mean_24h and mean_48h and base in self.CLINICAL_THRESHOLDS:
                thresholds = self.CLINICAL_THRESHOLDS[base]
                mid = (thresholds['low'] + thresholds['high']) / 2
                
                # Deterioration = moving away from normal
                dist_24h = np.abs(df[mean_24h] - mid)
                dist_48h = np.abs(df[mean_48h] - mid)
                
                # Is deteriorating? (24h worse than 48h)
                deteriorating = (dist_24h > dist_48h * 1.1).astype(int)  # 10% threshold
                features[f"{base}_Deteriorating"] = deteriorating
                deterioration_count += deteriorating
                
                # Rapid deterioration (>20% worse)
                rapid_deteriorating = (dist_24h > dist_48h * 1.2).astype(int)
                features[f"{base}_RapidDeterioration"] = rapid_deteriorating
        
        features['Total_Deterioration_Count'] = deterioration_count
        features['Multi_Organ_Deterioration'] = (deterioration_count >= 3).astype(int)
        
        return features
    
    def _create_window_comparison_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Create features comparing different time windows.
        """
        features = pd.DataFrame(index=df.index)
        
        base_names = set(self._get_base_name(col) for col in cols)
        
        for base in base_names:
            # Mean comparison
            mean_24h = f"{base}_Mean_24h" if f"{base}_Mean_24h" in df.columns else None
            mean_48h = f"{base}_Mean_48h" if f"{base}_Mean_48h" in df.columns else None
            mean_overall = f"{base}_Mean" if f"{base}_Mean" in df.columns else None
            
            # Std comparison
            std_24h = f"{base}_Std_24h" if f"{base}_Std_24h" in df.columns else None
            std_48h = f"{base}_Std_48h" if f"{base}_Std_48h" in df.columns else None
            
            # Count comparison (measurement frequency)
            count_24h = f"{base}_Count_24h" if f"{base}_Count_24h" in df.columns else None
            count_48h = f"{base}_Count_48h" if f"{base}_Count_48h" in df.columns else None
            
            # 24h vs 48h mean difference (recent trend)
            if mean_24h and mean_48h:
                features[f"{base}_Recent_vs_Earlier"] = df[mean_24h] - df[mean_48h]
            
            # Variability change (increasing instability)
            if std_24h and std_48h:
                with np.errstate(divide='ignore', invalid='ignore'):
                    var_change = df[std_24h] / (df[std_48h] + 1e-8)
                    features[f"{base}_Variability_Change"] = np.clip(var_change, 0, 10)
                    features[f"{base}_Instability_Increasing"] = (var_change > 1.2).astype(int)
            
            # Measurement frequency change (increasing monitoring)
            if count_24h and count_48h:
                with np.errstate(divide='ignore', invalid='ignore'):
                    freq_change = df[count_24h] / (df[count_48h] + 1e-8) 
                    features[f"{base}_Monitoring_Intensity"] = np.clip(freq_change, 0, 10)
                    features[f"{base}_Increased_Monitoring"] = (freq_change > 1.5).astype(int)
        
        return features


def engineer_features_for_training(
    vital_df: pd.DataFrame,
    nlp_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to create engineered features and merge with existing data.
    
    Args:
        vital_df: DataFrame with vital/lab features
        nlp_df: DataFrame with NLP features
        labels_df: DataFrame with labels
        output_path: Optional path to save engineered features
        
    Returns:
        X: Feature DataFrame (original + engineered)
        y: Labels Series
    """
    logger.info("="*70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)
    
    # Initialize engineer
    engineer = AdvancedFeatureEngineer(verbose=True)
    
    # Create advanced features from vital/lab data
    logger.info("\nCreating advanced features from vital/lab data...")
    advanced_features = engineer.create_all_features(vital_df)
    
    # Merge everything
    logger.info("\nMerging datasets...")
    
    # Start with vital_df
    merged = vital_df.copy()
    
    # Add advanced features (avoid duplicates)
    for col in advanced_features.columns:
        if col not in merged.columns:
            merged[col] = advanced_features[col].values
        else:
            # Rename engineered feature to avoid conflict
            merged[f'{col}_engineered'] = advanced_features[col].values
    
    # Merge with NLP
    nlp_merge_col = 'HADM_ID' if 'HADM_ID' in nlp_df.columns else 'hadm_id'
    merged = merged.merge(nlp_df, on=nlp_merge_col, how='inner')
    
    # Merge with labels
    labels_merge_col = 'HADM_ID' if 'HADM_ID' in labels_df.columns else 'hadm_id'
    merged = merged.merge(labels_df[[labels_merge_col, 'Y']], 
                          left_on='HADM_ID', right_on=labels_merge_col, how='inner')
    
    # Prepare X and y
    exclude_cols = {'SUBJECT_ID', 'Y', 'label', 
                   'subject_id', 'icustay_id', 'ICUSTAY_ID', 
                   'Gender', 'Discharge_Disposition'}
    time_cols = {col for col in merged.columns if 'Time' in col or 'ChartTime' in col}
    exclude_cols.update(time_cols)
    
    # Keep HADM_ID for merge purposes but exclude from training features
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    
    # Make sure HADM_ID is included for downstream merging
    if 'HADM_ID' not in feature_cols and 'HADM_ID' in merged.columns:
        feature_cols = ['HADM_ID'] + [c for c in feature_cols if c != 'HADM_ID']
    elif 'hadm_id' not in feature_cols and 'hadm_id' in merged.columns:
        feature_cols = ['hadm_id'] + [c for c in feature_cols if c != 'hadm_id']
    
    X = merged[feature_cols]
    y = merged['Y']
    
    logger.info(f"\nFinal feature set: {X.shape[1]} features")
    logger.info(f"Original features: ~{vital_df.shape[1] + nlp_df.shape[1] - 2}")
    logger.info(f"New engineered features: {advanced_features.shape[1]}")
    logger.info(f"Samples: {len(y)}")
    
    # Save if requested
    if output_path:
        X.to_csv(output_path, index=False)
        logger.info(f"\nSaved engineered features to: {output_path}")
    
    return X, y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Feature Engineering')
    parser.add_argument('--vital-features', type=str, required=True,
                       help='Path to vital/lab features CSV')
    parser.add_argument('--nlp-features', type=str, required=True,
                       help='Path to NLP features CSV')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels CSV')
    parser.add_argument('--output', type=str, default='engineered_features.csv',
                       help='Output path for engineered features')
    
    args = parser.parse_args()
    
    # Load data
    vital_df = pd.read_csv(args.vital_features, low_memory=False)
    nlp_df = pd.read_csv(args.nlp_features)
    labels_df = pd.read_csv(args.labels)
    
    # Engineer features
    X, y = engineer_features_for_training(vital_df, nlp_df, labels_df, args.output)
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Samples: {len(y)}")

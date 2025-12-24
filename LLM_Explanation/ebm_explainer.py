"""
EBM Explainer Module.
Extracts local explanations from EBM cross-interaction model.
"""
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import LLMConfig, ExplanationResult, MODULE_DIR

logger = logging.getLogger(__name__)


class EBMExplainer:
    """Extract explanations from EBM cross-interaction model."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.feature_names = None
        self.cross_interactions = None
        self.feature_mappings = None
        self._load_resources()
    
    def _load_resources(self):
        """Load model and supporting resources."""
        # Load EBM model
        model_path = Path(self.config.ebm_model_path)
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded EBM model from {model_path}")
        else:
            logger.warning(f"EBM model not found at {model_path}")
        
        # Load feature names
        feature_path = Path(self.config.feature_names_path)
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                feature_data = json.load(f)
            # Handle both old format (list) and new format (dict with X_V_features + X_T_features)
            if isinstance(feature_data, list):
                self.feature_names = feature_data
            elif isinstance(feature_data, dict):
                # New format from ebm_optimized_final
                self.feature_names = feature_data.get('X_V_features', []) + feature_data.get('X_T_features', [])
            else:
                self.feature_names = []
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load cross-interactions
        interactions_path = Path(self.config.cross_interactions_path)
        if interactions_path.exists():
            with open(interactions_path, 'r') as f:
                data = json.load(f)
            # Handle both formats: list or dict with 'forced_interactions' key
            if isinstance(data, list):
                self.cross_interactions = data
            elif isinstance(data, dict) and 'forced_interactions' in data:
                self.cross_interactions = data['forced_interactions']
            else:
                self.cross_interactions = []
            logger.info(f"Loaded {len(self.cross_interactions)} cross-interactions")
        
        # Load feature mappings (clinical descriptions)
        mappings_path = MODULE_DIR / "feature_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                self.feature_mappings = json.load(f)
            logger.info("Loaded feature mappings")
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for a feature."""
        if self.feature_mappings is None:
            return feature_name
        
        # Check vital features
        if feature_name in self.feature_mappings.get("vital_features", {}):
            return self.feature_mappings["vital_features"][feature_name]
        
        # Check NLP features
        if feature_name in self.feature_mappings.get("nlp_features", {}):
            return self.feature_mappings["nlp_features"][feature_name]
        
        # Fallback: clean up feature name
        return feature_name.replace("_", " ").title()
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to human-readable level."""
        if score < 0.3:
            return "Low"
        elif score < 0.5:
            return "Moderate"
        elif score < 0.7:
            return "Elevated"
        else:
            return "High"
    
    def explain(
        self, 
        patient_data: Dict[str, Any],
        clinical_notes: Optional[str] = None
    ) -> ExplanationResult:
        """
        Generate explanation for a single patient.
        
        Args:
            patient_data: Dictionary of feature values for the patient
            clinical_notes: Optional raw clinical notes (if provided, NLP features 
                           will be auto-extracted and override any in patient_data)
            
        Returns:
            ExplanationResult with risk score, level, top factors, and interactions
        """
        if self.model is None:
            raise RuntimeError("EBM model not loaded. Check model path.")
        
        # Auto-extract NLP features from clinical notes if provided
        if clinical_notes:
            from .nlp_extractor import extract_nlp_features
            nlp_features = extract_nlp_features(clinical_notes)
            patient_data = {**patient_data, **nlp_features}
            logger.info(f"Extracted {sum(nlp_features.values())} active NLP features from notes")
        
        # Create DataFrame for prediction
        df = pd.DataFrame([patient_data])
        
        # Ensure all required features exist (create missing as NaN)
        if self.feature_names:
            missing_cols = {feat: np.nan for feat in self.feature_names if feat not in df.columns}
            if missing_cols:
                df = pd.concat([df, pd.DataFrame(missing_cols, index=df.index)], axis=1)
            df = df[self.feature_names]
        
        # Convert all columns to numeric, fill NaN with 0 for EBM
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        
        # Get prediction probability
        proba = self.model.predict_proba(df)[:, 1][0]
        risk_level = self._get_risk_level(proba)
        
        # Extract local explanation
        top_factors = self._extract_top_factors(df, patient_data)
        active_interactions = self._extract_active_interactions(patient_data)
        
        return ExplanationResult(
            risk_score=float(proba),
            risk_level=risk_level,
            top_factors=top_factors,
            cross_interactions=active_interactions
        )
    
    def _extract_top_factors(
        self, 
        df: pd.DataFrame,
        patient_data: Dict[str, Any]
    ) -> List[Dict]:
        """Extract top contributing factors using EBM explain_local."""
        factors = []
        
        try:
            # Use EBM's built-in explain_local
            explanation = self.model.explain_local(df)
            data = explanation.data(0)  # Get first (only) row
            
            if data is not None:
                names = data.get('names', [])
                scores = data.get('scores', [])
                
                # Sort by absolute contribution
                indexed = [(i, abs(s)) for i, s in enumerate(scores)]
                indexed.sort(key=lambda x: x[1], reverse=True)
                
                # Take top k
                for idx, _ in indexed[:self.config.top_k_factors]:
                    if idx < len(names):
                        feat_name = names[idx]
                        contribution = scores[idx]
                        direction = "increases" if contribution > 0 else "decreases"
                        
                        factors.append({
                            "name": feat_name,
                            "description": self._get_feature_description(feat_name),
                            "contribution": float(contribution),
                            "direction": direction,
                            "value": patient_data.get(feat_name)
                        })
                        
        except Exception as e:
            logger.warning(f"Could not extract local explanation: {e}")
            # Fallback: use feature names with high values
            if self.feature_names:
                for feat in self.feature_names[:self.config.top_k_factors]:
                    if feat in patient_data:
                        factors.append({
                            "name": feat,
                            "description": self._get_feature_description(feat),
                            "contribution": 0.0,
                            "direction": "unknown",
                            "value": patient_data.get(feat)
                        })
        
        return factors
    
    def _extract_active_interactions(
        self, 
        patient_data: Dict[str, Any]
    ) -> List[Dict]:
        """Extract active cross-interactions for this patient."""
        interactions = []
        
        if not self.cross_interactions:
            return interactions
        
        for interaction in self.cross_interactions[:self.config.top_k_interactions]:
            vital = interaction.get("vital", "")
            nlp = interaction.get("nlp", "")
            score = interaction.get("score", 0)
            
            # Check if both features are present/active
            vital_value = patient_data.get(vital)
            nlp_value = patient_data.get(nlp)
            
            if nlp_value == 1:  # NLP feature is active
                vital_desc = self._get_feature_description(vital)
                nlp_desc = self._get_feature_description(nlp)
                
                interactions.append({
                    "vital": vital,
                    "nlp": nlp,
                    "vital_description": vital_desc,
                    "nlp_description": nlp_desc,
                    "description": f"When {nlp_desc.lower()}, {vital_desc.lower()} becomes more important",
                    "importance": float(score)
                })
        
        return interactions
    
    def explain_batch(
        self, 
        patients_data: List[Dict[str, Any]]
    ) -> List[ExplanationResult]:
        """Generate explanations for multiple patients."""
        return [self.explain(p) for p in patients_data]

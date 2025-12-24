"""
NLP Feature Extractor for Clinical Notes.
Extracts 41 NLP features from doctor's notes to feed into EBM model.
"""
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Mapping of NLP feature names to regex patterns
# These patterns match clinical concepts in discharge notes
NLP_PATTERNS = {
    # Infections
    "nlp_sepsis": r"\bsepsis\b|\bseptic\b|\bsirs\b|\bbacteremia\b",
    "nlp_pneumonia": r"\bpneumonia\b|\bpna\b|\bcap\b|\bhap\b|\bvap\b",
    "nlp_cellulitis": r"\bcellulitis\b|\bskin infection\b|\berysipelas\b",
    "nlp_uti": r"\buti\b|\burinary tract infection\b|\bpyelonephritis\b",
    "nlp_fever": r"\bfever\b|\bfebrile\b|\btemperature\s*>\s*38\b",
    
    # Cardiovascular
    "nlp_heart_failure": r"\bheart failure\b|\bchf\b|\bhfref\b|\bhfpef\b|\bef\s*[<>]?\s*\d+%",
    "nlp_arrhythmia": r"\barrhythmia\b|\bafib\b|\batrial fibrillation\b|\bvt\b|\bvfib\b",
    "nlp_hypotension": r"\bhypotension\b|\bhypotensive\b|\bmap\s*<\s*65\b|\bsbp\s*<\s*90\b",
    "nlp_hypertension": r"\bhypertension\b|\bhtn\b|\bhypertensive\b",
    "nlp_cardiac_arrest": r"\bcardiac arrest\b|\bcode blue\b|\brosc\b|\bcpr\b",
    "nlp_myocardial_infarction": r"\bmyocardial infarction\b|\bmi\b|\bnstemi\b|\bstemi\b|\btroponin\b",
    "nlp_stroke": r"\bstroke\b|\bcva\b|\bcerebrovascular\b|\btia\b",
    "nlp_dvt_pe": r"\bdvt\b|\bpe\b|\bpulmonary embolism\b|\bdeep vein thrombosis\b",
    
    # Respiratory
    "nlp_respiratory_failure": r"\brespiratory failure\b|\bhypoxia\b|\bhypoxic\b|\bards\b",
    "nlp_ards": r"\bards\b|\bacute respiratory distress\b",
    "nlp_copd": r"\bcopd\b|\bchronic obstructive\b|\bemphysema\b",
    "nlp_pleural_effusion": r"\bpleural effusion\b|\bpneumothorax\b|\bchest tube\b",
    "nlp_intubation": r"\bintubat\w*\b|\bventilat\w*\b|\bett\b|\bextubat\w*\b",
    
    # Renal
    "nlp_chronic_kidney_disease": r"\bckd\b|\bchronic kidney\b|\besrd\b|\bdialysis\b",
    "nlp_acute_kidney_injury": r"\baki\b|\bacute kidney injury\b|\bacute renal\b|\bcreatinine\s*(rising|elevated)\b",
    
    # Hepatic
    "nlp_liver_failure": r"\bliver failure\b|\bhepatic failure\b|\bcirrhosis\b|\bhepatic encephalopathy\b",
    
    # Hematologic
    "nlp_thrombocytopenia": r"\bthrombocytopenia\b|\blow platelet\b|\bplt\s*<\s*100\b",
    "nlp_anemia": r"\banemia\b|\bhgb\s*<\s*[78]\b|\btransfusion\b|\bprbc\b",
    "nlp_coagulopathy": r"\bcoagulopathy\b|\bdic\b|\binr\s*>\s*[23]\b|\bbleeding\b",
    "nlp_gi_bleed": r"\bgi bleed\b|\bgastrointestinal bleed\b|\bhematemesis\b|\bmelena\b|\bhematochezia\b",
    
    # Metabolic
    "nlp_electrolyte_imbalance": r"\belectrolyte\b|\bhyponatremia\b|\bhyperkalemia\b|\bhypokalemia\b",
    "nlp_acidosis": r"\bacidosis\b|\bmetabolic acidosis\b|\bph\s*<\s*7\.3\b|\blactate\s*>\s*[24]\b",
    "nlp_diabetes": r"\bdiabetes\b|\bdm\b|\bhyperglycemia\b|\bdiabetic\b|\binsulin\b",
    "nlp_pancreatitis": r"\bpancreatitis\b|\blipase\b|\bamylase elevated\b",
    
    # Neurologic
    "nlp_altered_mental_status": r"\baltered mental\b|\bconfusion\b|\bdelirium\b|\bencephalopathy\b|\boriented\s*x\s*[012]\b",
    "nlp_seizure": r"\bseizure\b|\bepilepsy\b|\bconvulsion\b|\bstatus epilepticus\b",
    
    # Procedures / Interventions
    "nlp_surgery": r"\bsurgery\b|\bpost[- ]?op\b|\boperative\b|\bor\b|\bprocedure\b",
    "nlp_central_line": r"\bcentral line\b|\bpicc\b|\bcvc\b|\bcentral venous\b",
    "nlp_vasopressors": r"\bvasopressor\b|\bnorepinephrine\b|\blevophed\b|\bdopamine\b|\bphenylephrine\b",
    
    # Other Clinical Conditions
    "nlp_unstable": r"\bunstable\b|\bdecompensating\b|\bdeteriorating\b|\bworsening\b",
    "nlp_edema": r"\bedema\b|\bswelling\b|\banasarca\b|\bfluid overload\b",
    "nlp_pressure_ulcer": r"\bpressure ulcer\b|\bdecubitus\b|\bbed sore\b|\bstage [1-4] ulcer\b",
    "nlp_falls": r"\bfall\b|\bfell\b|\bfound on floor\b",
    "nlp_pain": r"\bpain\b|\bdiscomfort\b|\bnrs\s*[7-9]|10\b",
    "nlp_code_status": r"\bcode status\b|\bdnr\b|\bdni\b|\bfull code\b|\bcomfort care\b",
    "nlp_icu_admission": r"\bicu\b|\bintensive care\b|\bccu\b|\bmicu\b|\bsicu\b",
}


class NLPExtractor:
    """Extract NLP features from clinical notes."""
    
    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        """
        Initialize with custom patterns or use defaults.
        
        Args:
            patterns: Optional dict mapping feature names to regex patterns
        """
        self.patterns = patterns or NLP_PATTERNS
        self._compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        logger.info(f"NLPExtractor initialized with {len(self.patterns)} patterns")
    
    def extract(self, clinical_notes: str) -> Dict[str, int]:
        """
        Extract NLP features from clinical notes text.
        
        Args:
            clinical_notes: Raw text from doctor's notes
            
        Returns:
            Dict with feature names as keys and 0/1 as values
        """
        if not clinical_notes or not isinstance(clinical_notes, str):
            # Return all zeros if no notes
            return {name: 0 for name in self.patterns}
        
        text = clinical_notes.lower().strip()
        
        features = {}
        for name, pattern in self._compiled.items():
            match = pattern.search(text)
            features[name] = 1 if match else 0
        
        # Log extracted features
        active = [k for k, v in features.items() if v == 1]
        if active:
            logger.info(f"Extracted {len(active)} NLP features: {active[:5]}{'...' if len(active) > 5 else ''}")
        
        return features
    
    def extract_with_positions(self, clinical_notes: str) -> dict:
        """
        Extract NLP features with positions for highlighting.
        
        Args:
            clinical_notes: Raw clinical notes text
            
        Returns:
            Dict with:
                - features: {feature_name: 0/1}
                - highlights: [{feature, text, start, end}]
        """
        if not clinical_notes or not isinstance(clinical_notes, str):
            return {
                "features": {name: 0 for name in self.patterns},
                "highlights": []
            }
        
        text_lower = clinical_notes.lower()
        features = {}
        highlights = []
        
        for name, pattern in self._compiled.items():
            matches = list(pattern.finditer(text_lower))
            features[name] = 1 if matches else 0
            
            # Collect all match positions
            for match in matches[:5]:  # Limit to 5 matches per feature
                highlights.append({
                    "feature": name,
                    "text": clinical_notes[match.start():match.end()],  # Original case
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort by position
        highlights.sort(key=lambda x: x["start"])
        
        # Log extracted features
        active = [k for k, v in features.items() if v == 1]
        if active:
            logger.info(f"Extracted {len(active)} NLP features with {len(highlights)} highlights")
        
        return {
            "features": features,
            "highlights": highlights
        }
    
    def extract_with_matches(self, clinical_notes: str) -> Dict[str, dict]:
        """
        Extract NLP features with matched text for transparency.
        
        Returns:
            Dict with feature info including matched text
        """
        if not clinical_notes:
            return {name: {"value": 0, "matches": []} for name in self.patterns}
        
        text = clinical_notes.lower()
        
        features = {}
        for name, pattern in self._compiled.items():
            matches = pattern.findall(text)
            features[name] = {
                "value": 1 if matches else 0,
                "matches": matches[:3]  # Keep first 3 matches
            }
        
        return features
    
    def get_feature_names(self) -> list:
        """Get list of all NLP feature names."""
        return list(self.patterns.keys())


def extract_nlp_features(clinical_notes: str) -> Dict[str, int]:
    """
    Convenience function to extract NLP features.
    
    Args:
        clinical_notes: Raw clinical notes text
        
    Returns:
        Dict with 41 NLP features (0 or 1)
    """
    extractor = NLPExtractor()
    return extractor.extract(clinical_notes)

"""
Clinical NLP Pipeline using spaCy with Negation Detection.

Pipeline steps:
1. Text Cleaning: Remove de-identification tokens, headers, noise
2. Sentence Segmentation: Using spaCy dependency parser
3. NER: Extract medical entities (Problems, Treatments, Tests)
4. Negation Detection: Flag negated entities using negspacy
5. Concept Mapping: Map entities to standardized concepts
6. Bag-of-Concepts: Create binary features for each concept

Output: Binary features indicating presence/absence of clinical concepts
"""

import argparse
import csv
import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# spaCy imports
import spacy
from spacy.tokens import Doc
from negspacy.negation import Negex

# Sklearn for feature selection
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# CONSTANTS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / 'cohort'

# De-identification pattern
DEID_RE = re.compile(r"\[\*\*.*?\*\*\]", flags=re.DOTALL)

# Common section headers to remove
SECTION_HEADERS = [
    r"admission date:.*?\n",
    r"discharge date:.*?\n", 
    r"date of birth:.*?\n",
    r"sex:.*?\n",
    r"service:.*?\n",
    r"chief complaint:.*?\n",
    r"allergies:.*?\n",
    r"attending:.*?\n",
    r"dictated by:.*?\n",
    r"completed by:.*?\n",
]

# Clinical concept categories (ICD-like groupings)
CLINICAL_CONCEPT_GROUPS = {
    # Cardiovascular
    'cardiac_arrest': ['cardiac arrest', 'asystole', 'pulseless', 'code blue'],
    'heart_failure': ['heart failure', 'chf', 'congestive heart failure', 'hfref', 'hfpef', 'cardiomyopathy'],
    'myocardial_infarction': ['myocardial infarction', 'mi', 'stemi', 'nstemi', 'heart attack', 'troponin elevated'],
    'arrhythmia': ['arrhythmia', 'atrial fibrillation', 'afib', 'a-fib', 'ventricular tachycardia', 'vtach', 'bradycardia', 'tachycardia'],
    'hypertension': ['hypertension', 'htn', 'high blood pressure', 'hypertensive'],
    'hypotension': ['hypotension', 'low blood pressure', 'hypotensive', 'shock'],
    
    # Respiratory
    'respiratory_failure': ['respiratory failure', 'hypoxic', 'hypoxia', 'hypoxemic', 'desaturation'],
    'pneumonia': ['pneumonia', 'pna', 'lung infection', 'consolidation'],
    'ards': ['ards', 'acute respiratory distress', 'ali'],
    'copd': ['copd', 'chronic obstructive', 'emphysema'],
    'intubation': ['intubated', 'intubation', 'mechanical ventilation', 'ventilator', 'extubated', 'extubation'],
    'pleural_effusion': ['pleural effusion', 'effusion'],
    
    # Neurological
    'altered_mental_status': ['altered mental status', 'ams', 'confusion', 'confused', 'delirium', 'encephalopathy', 'obtunded'],
    'stroke': ['stroke', 'cva', 'cerebrovascular', 'tia', 'ischemic stroke', 'hemorrhagic stroke'],
    'seizure': ['seizure', 'sz', 'epilepsy', 'convulsion'],
    
    # Renal
    'acute_kidney_injury': ['acute kidney injury', 'aki', 'acute renal failure', 'arf', 'creatinine elevated'],
    'chronic_kidney_disease': ['chronic kidney disease', 'ckd', 'esrd', 'end stage renal', 'dialysis', 'hemodialysis'],
    
    # Infectious
    'sepsis': ['sepsis', 'septic', 'sirs', 'bacteremia', 'septic shock'],
    'uti': ['urinary tract infection', 'uti', 'pyelonephritis'],
    'cellulitis': ['cellulitis', 'skin infection'],
    
    # GI
    'gi_bleed': ['gi bleed', 'gastrointestinal bleed', 'melena', 'hematochezia', 'hematemesis', 'upper gi bleed', 'lower gi bleed'],
    'liver_failure': ['liver failure', 'hepatic failure', 'cirrhosis', 'hepatic encephalopathy'],
    'pancreatitis': ['pancreatitis', 'lipase elevated'],
    
    # Hematologic
    'anemia': ['anemia', 'low hemoglobin', 'low hgb', 'transfusion'],
    'coagulopathy': ['coagulopathy', 'dic', 'inr elevated', 'bleeding'],
    'thrombocytopenia': ['thrombocytopenia', 'low platelets', 'plt low'],
    
    # Metabolic/Endocrine
    'diabetes': ['diabetes', 'dm', 'hyperglycemia', 'dka', 'diabetic ketoacidosis', 'insulin'],
    'electrolyte_imbalance': ['hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia', 'hypocalcemia', 'hypercalcemia'],
    'acidosis': ['acidosis', 'metabolic acidosis', 'respiratory acidosis', 'lactic acidosis'],
    
    # Procedures/Interventions
    'surgery': ['surgery', 'surgical', 'operation', 'operative', 'post-op', 'postoperative'],
    'central_line': ['central line', 'central venous catheter', 'cvc', 'picc', 'port'],
    'vasopressors': ['vasopressor', 'levophed', 'norepinephrine', 'dopamine', 'vasopressin', 'phenylephrine', 'epinephrine'],
    
    # General Severity Indicators
    'icu_admission': ['icu', 'intensive care', 'micu', 'sicu', 'ccu', 'critical care'],
    'unstable': ['unstable', 'deteriorating', 'worsening', 'decompensated'],
    'code_status': ['dnr', 'dni', 'comfort care', 'cmo', 'palliative', 'hospice'],
    
    # Additional common conditions
    'falls': ['fall', 'fell', 'mechanical fall'],
    'pressure_ulcer': ['pressure ulcer', 'decubitus', 'bedsore', 'pressure injury'],
    'dvt_pe': ['dvt', 'deep vein thrombosis', 'pe', 'pulmonary embolism', 'vte'],
    'pain': ['pain', 'painful'],
    'fever': ['fever', 'febrile', 'temperature elevated'],
    'edema': ['edema', 'swelling', 'fluid overload', 'volume overload'],
}


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_clinical_text(text: str) -> str:
    """
    Clean clinical note text:
    - Remove de-identification tokens [** ... **]
    - Remove section headers with dates
    - Normalize whitespace
    """
    if not text:
        return ""
    
    # Remove de-identification tokens
    text = DEID_RE.sub(" ", text)
    
    # Remove section headers
    for pattern in SECTION_HEADERS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# ============================================================================
# CONCEPT EXTRACTION
# ============================================================================

class ClinicalConceptExtractor:
    """Extract clinical concepts from text using pattern matching and negation detection."""
    
    def __init__(self, use_negation: bool = True):
        """Initialize the extractor with spaCy and negation detection."""
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Add negation detection
        if use_negation:
            print("Adding negation detection...")
            # negspacy requires entity recognition, add simple pattern matching
            self.nlp.add_pipe("negex")
        
        self.use_negation = use_negation
        
        # Compile concept patterns
        self.concept_patterns = {}
        for concept, keywords in CLINICAL_CONCEPT_GROUPS.items():
            # Create regex pattern for each concept
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            self.concept_patterns[concept] = re.compile(pattern, re.IGNORECASE)
    
    def extract_concepts(self, text: str) -> Dict[str, int]:
        """
        Extract clinical concepts from text.
        
        Returns:
            Dict mapping concept name to:
            - 1: concept present and NOT negated
            - 0: concept absent or negated
        """
        if not text:
            return {}
        
        # Clean text first
        text = clean_clinical_text(text)
        
        # Initialize all concepts as absent
        concepts = {concept: 0 for concept in self.concept_patterns}
        
        # Process with spaCy for sentence segmentation
        doc = self.nlp(text[:100000])  # Limit text length for memory
        
        # Check each sentence for concepts
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check negation for this sentence
            sent_negated = False
            if self.use_negation:
                # Check if sentence contains negation markers
                for token in sent:
                    if hasattr(token._, 'negex') and token._.negex:
                        sent_negated = True
                        break
            
            # Check for each concept
            for concept, pattern in self.concept_patterns.items():
                if pattern.search(sent_text):
                    # Check if concept mention is negated
                    if self._is_concept_negated(sent_text, pattern):
                        # Negated - don't set to 1
                        pass
                    else:
                        concepts[concept] = 1
        
        return concepts
    
    def _is_concept_negated(self, sent_text: str, concept_pattern: re.Pattern) -> bool:
        """Check if concept mention in sentence is negated."""
        # Common negation patterns
        negation_patterns = [
            r'no\s+(?:evidence|signs?|symptoms?|history)\s+of',
            r'denies',
            r'negative\s+for',
            r'ruled\s+out',
            r'without',
            r'not\s+(?:have|show|demonstrate)',
            r'absence\s+of',
            r'free\s+of',
        ]
        
        # Find the concept match
        match = concept_pattern.search(sent_text)
        if not match:
            return False
        
        # Check for negation before the concept
        text_before = sent_text[:match.start()]
        
        for neg_pattern in negation_patterns:
            if re.search(neg_pattern, text_before):
                return True
        
        return False
    
    def process_batch(self, texts: List[str], hadm_ids: List[str]) -> pd.DataFrame:
        """
        Process a batch of texts and return DataFrame with concept features.
        
        Args:
            texts: List of clinical note texts
            hadm_ids: List of HADM_IDs corresponding to texts
        
        Returns:
            DataFrame with HADM_ID and binary concept columns
        """
        results = []
        
        for hadm_id, text in tqdm(zip(hadm_ids, texts), total=len(texts), desc="Extracting concepts"):
            concepts = self.extract_concepts(text)
            concepts['HADM_ID'] = hadm_id
            results.append(concepts)
        
        df = pd.DataFrame(results)
        
        # Reorder columns with HADM_ID first
        cols = ['HADM_ID'] + [c for c in df.columns if c != 'HADM_ID']
        df = df[cols]
        
        return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_top_concepts(
    concept_df: pd.DataFrame,
    labels: pd.Series,
    method: str = 'chi2',
    min_frequency: float = 0.01,
    top_k: int = 200
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top-K most predictive concepts.
    
    Args:
        concept_df: DataFrame with binary concept features
        labels: Target labels (0/1)
        method: 'chi2' or 'mutual_info'
        min_frequency: Minimum frequency threshold (e.g., 0.01 = 1%)
        top_k: Number of top concepts to select
    
    Returns:
        Tuple of (selected_concept_names, scores_dataframe)
    """
    # Get concept columns (exclude HADM_ID)
    concept_cols = [c for c in concept_df.columns if c != 'HADM_ID']
    
    # Filter by minimum frequency
    frequencies = concept_df[concept_cols].mean()
    valid_concepts = frequencies[frequencies >= min_frequency].index.tolist()
    print(f"Concepts meeting {min_frequency*100:.1f}% frequency threshold: {len(valid_concepts)}/{len(concept_cols)}")
    
    if not valid_concepts:
        print("Warning: No concepts meet frequency threshold!")
        return [], pd.DataFrame()
    
    # Prepare feature matrix
    X = concept_df[valid_concepts].values
    y = labels.values
    
    # Calculate scores
    if method == 'chi2':
        scores, pvalues = chi2(X, y)
        score_df = pd.DataFrame({
            'concept': valid_concepts,
            'chi2_score': scores,
            'pvalue': pvalues,
            'frequency': frequencies[valid_concepts].values
        })
        score_df = score_df.sort_values('chi2_score', ascending=False)
    else:  # mutual_info
        scores = mutual_info_classif(X, y, random_state=42)
        score_df = pd.DataFrame({
            'concept': valid_concepts,
            'mi_score': scores,
            'frequency': frequencies[valid_concepts].values
        })
        score_df = score_df.sort_values('mi_score', ascending=False)
    
    # Select top-K
    selected = score_df.head(top_k)['concept'].tolist()
    print(f"Selected top {len(selected)} concepts")
    
    return selected, score_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    notes_csv: str,
    labels_csv: str,
    output_dir: str,
    min_frequency: float = 0.01,
    top_k: int = 200,
    feature_selection_method: str = 'chi2'
):
    """
    Run the full NLP pipeline.
    
    Args:
        notes_csv: Path to CSV with HADM_ID, CLEAN_TEXT columns
        labels_csv: Path to CSV with HADM_ID, Y columns
        output_dir: Directory for output files
        min_frequency: Minimum concept frequency (default 1%)
        top_k: Number of top concepts to select
        feature_selection_method: 'chi2' or 'mutual_info'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load notes
    print(f"\n{'='*60}")
    print("STEP 1: Loading clinical notes")
    print('='*60)
    notes_df = pd.read_csv(notes_csv)
    print(f"Loaded {len(notes_df)} notes")
    
    # Load labels
    print(f"\n{'='*60}")
    print("STEP 2: Loading labels")
    print('='*60)
    labels_df = pd.read_csv(labels_csv)
    print(f"Loaded {len(labels_df)} labels")
    print(f"Label distribution: {labels_df['Y'].value_counts().to_dict()}")
    
    # Merge to ensure alignment
    merged = notes_df.merge(labels_df[['HADM_ID', 'Y']], on='HADM_ID', how='inner')
    print(f"Matched {len(merged)} records")
    
    # Extract concepts
    print(f"\n{'='*60}")
    print("STEP 3: Extracting clinical concepts")
    print('='*60)
    extractor = ClinicalConceptExtractor(use_negation=True)
    
    concept_df = extractor.process_batch(
        merged['CLEAN_TEXT'].tolist(),
        merged['HADM_ID'].tolist()
    )
    
    # Save all concepts
    all_concepts_path = os.path.join(output_dir, 'all_concepts.csv')
    concept_df.to_csv(all_concepts_path, index=False)
    print(f"Saved all concepts to {all_concepts_path}")
    
    # Feature selection
    print(f"\n{'='*60}")
    print("STEP 4: Feature selection")
    print('='*60)
    
    # Align labels with concept_df
    concept_with_labels = concept_df.merge(merged[['HADM_ID', 'Y']], on='HADM_ID')
    
    selected_concepts, score_df = select_top_concepts(
        concept_with_labels.drop(columns=['Y']),
        concept_with_labels['Y'],
        method=feature_selection_method,
        min_frequency=min_frequency,
        top_k=top_k
    )
    
    # Save scores
    scores_path = os.path.join(output_dir, 'concept_scores.csv')
    score_df.to_csv(scores_path, index=False)
    print(f"Saved concept scores to {scores_path}")
    
    # Create final feature set
    print(f"\n{'='*60}")
    print("STEP 5: Creating final feature set")
    print('='*60)
    
    if selected_concepts:
        final_features = concept_df[['HADM_ID'] + selected_concepts].copy()
        
        # Rename columns with nlp_ prefix for clarity
        rename_map = {c: f'nlp_{c}' for c in selected_concepts}
        final_features = final_features.rename(columns=rename_map)
    else:
        # Use all concepts if selection failed
        concept_cols = [c for c in concept_df.columns if c != 'HADM_ID']
        final_features = concept_df.copy()
        rename_map = {c: f'nlp_{c}' for c in concept_cols}
        final_features = final_features.rename(columns=rename_map)
    
    # Save final features
    features_path = os.path.join(output_dir, 'nlp_features_bag_of_concepts.csv')
    final_features.to_csv(features_path, index=False)
    print(f"Saved {len(final_features)} records with {len(final_features.columns)-1} NLP features")
    print(f"Output: {features_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total patients: {len(final_features)}")
    print(f"NLP features: {len(final_features.columns) - 1}")
    
    # Concept prevalence
    nlp_cols = [c for c in final_features.columns if c.startswith('nlp_')]
    prevalence = final_features[nlp_cols].mean().sort_values(ascending=False)
    print(f"\nTop 10 most common concepts:")
    for concept, prev in prevalence.head(10).items():
        print(f"  {concept}: {prev*100:.1f}%")
    
    # Save metadata
    metadata = {
        'notes_csv': notes_csv,
        'labels_csv': labels_csv,
        'n_patients': len(final_features),
        'n_features': len(final_features.columns) - 1,
        'min_frequency': min_frequency,
        'top_k': top_k,
        'feature_selection_method': feature_selection_method,
        'selected_concepts': selected_concepts
    }
    metadata_path = os.path.join(output_dir, 'pipeline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")
    
    return final_features


def main():
    parser = argparse.ArgumentParser(
        description='Clinical NLP Pipeline: Extract Bag-of-Concepts features'
    )
    parser.add_argument(
        '--notes', '-n',
        default=str(COHORT_DIR / 'notes_before_icu_out.csv'),
        help='Path to notes CSV (HADM_ID, CLEAN_TEXT)'
    )
    parser.add_argument(
        '--labels', '-l',
        default=str(COHORT_DIR / 'new_cohort_icu_readmission_labels.csv'),
        help='Path to labels CSV (HADM_ID, Y)'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(COHORT_DIR / 'nlp_features_safe'),
        help='Output directory'
    )
    parser.add_argument(
        '--min-freq', type=float, default=0.01,
        help='Minimum concept frequency (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--top-k', type=int, default=200,
        help='Number of top concepts to select (default: 200)'
    )
    parser.add_argument(
        '--method', choices=['chi2', 'mutual_info'], default='chi2',
        help='Feature selection method (default: chi2)'
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        notes_csv=args.notes,
        labels_csv=args.labels,
        output_dir=args.output,
        min_frequency=args.min_freq,
        top_k=args.top_k,
        feature_selection_method=args.method
    )


if __name__ == '__main__':
    main()

"""
Full Clinical NLP Pipeline for ICU Readmission Prediction
=========================================================

Pipeline Steps (theo phương pháp nghiên cứu):
1. Text cleaning - Làm sạch văn bản
2. Sentence segmentation - Phân đoạn câu  
3. NER extraction - Trích xuất thực thể y tế (scispaCy en_core_sci_lg)
4. Negation detection - Phát hiện phủ định (negspacy)
5. Concept aggregation - Tổng hợp concepts không phủ định
6. Frequency filtering - Lọc theo tần suất (≥1-2%)
7. Binary encoding - Mã hóa nhị phân cho Bag-of-Concepts

KHÔNG sử dụng Discharge Summary (DATA LEAKAGE!)
Chỉ dùng notes TRƯỚC ICU OUTTIME (Nursing, Nursing/other, Physician)

Author: VRES Project
"""

import pandas as pd
import numpy as np
import re
import spacy
from negspacy.negation import Negex
from collections import Counter, defaultdict
from tqdm import tqdm
import logging
import os
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalNLPPipeline:
    """
    Full clinical NLP pipeline for extracting Bag-of-Concepts features
    from clinical notes.
    """
    
    def __init__(self, min_frequency_pct=1.0, max_features=500):
        """
        Initialize the pipeline.
        
        Args:
            min_frequency_pct: Minimum frequency (%) for a concept to be included
            max_features: Maximum number of features to keep
        """
        self.min_frequency_pct = min_frequency_pct
        self.max_features = max_features
        self.nlp = None
        self.selected_concepts = None
        
    def load_model(self):
        """Load scispaCy model with NegEx"""
        logger.info("Loading scispaCy model (en_core_sci_lg)...")
        
        try:
            self.nlp = spacy.load("en_core_sci_lg")
            logger.info(f"Model loaded: {self.nlp.meta['name']}")
            
            # Add NegEx for negation detection
            # Clinical negation terms
            from negspacy.termsets import termset
            ts = termset("en_clinical")
            
            # Add negex to pipeline
            self.nlp.add_pipe(
                "negex",
                config={
                    "neg_termset": ts.get_patterns(),
                    "extension_name": "negex"
                }
            )
            logger.info("Added NegEx for negation detection")
            
            logger.info(f"Pipeline components: {self.nlp.pipe_names}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def clean_text(self, text):
        """
        Clean clinical text while preserving medical information.
        
        Steps:
        1. Remove de-identification markers [**...**]
        2. Normalize whitespace
        3. Remove excessive punctuation
        4. Keep medical abbreviations intact
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Remove de-identification markers
        text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)
        
        # Remove headers/footers common in MIMIC
        text = re.sub(r'^Name:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Unit No:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Admission Date:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Discharge Date:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Date of Birth:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Sex:.*$', '', text, flags=re.MULTILINE)
        
        # Remove section headers but keep content
        # Common patterns: CHIEF COMPLAINT:, HISTORY:, etc.
        text = re.sub(r'^[A-Z][A-Z\s]+:\s*$', '', text, flags=re.MULTILINE)
        
        # Normalize numbers (keep numerical values for vitals)
        # text = re.sub(r'\b\d+\.?\d*\b', 'NUM', text)  # Optional: normalize numbers
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    def extract_entities_from_doc(self, doc):
        """
        Extract medical entities from a spaCy doc, excluding negated ones.
        
        Returns:
            list of (entity_text_normalized, entity_label) tuples
        """
        entities = []
        
        for ent in doc.ents:
            # Check if entity is negated
            if hasattr(ent._, 'negex') and ent._.negex:
                continue  # Skip negated entities
                
            # Normalize entity text (lowercase, strip)
            ent_text = ent.text.lower().strip()
            
            # Skip very short or very long entities
            if len(ent_text) < 2 or len(ent_text) > 100:
                continue
                
            # Skip numeric-only entities
            if re.match(r'^[\d\s\.\,]+$', ent_text):
                continue
                
            entities.append((ent_text, ent.label_))
            
        return entities
        
    def process_notes(self, df, text_column='CLEAN_TEXT', id_column='HADM_ID', batch_size=100):
        """
        Process all notes and extract entities.
        
        Args:
            df: DataFrame with clinical notes
            text_column: Column containing text
            id_column: Column with patient/admission ID
            batch_size: Batch size for spaCy processing
            
        Returns:
            Dict mapping HADM_ID to list of (entity, label) tuples
        """
        logger.info(f"Processing {len(df)} notes...")
        
        # Group notes by HADM_ID (concatenate if multiple notes per admission)
        notes_by_hadm = df.groupby(id_column)[text_column].apply(
            lambda x: ' '.join(x.dropna().astype(str))
        ).to_dict()
        
        logger.info(f"Unique HADM_IDs: {len(notes_by_hadm)}")
        
        # Process each HADM_ID
        hadm_entities = {}
        hadm_ids = list(notes_by_hadm.keys())
        texts = [self.clean_text(notes_by_hadm[hid]) for hid in hadm_ids]
        
        # Process in batches using spaCy's pipe
        logger.info("Extracting entities with NER + Negation detection...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = hadm_ids[i:i+batch_size]
            
            # Process batch
            docs = list(self.nlp.pipe(batch_texts, disable=['tagger']))
            
            for hadm_id, doc in zip(batch_ids, docs):
                entities = self.extract_entities_from_doc(doc)
                hadm_entities[hadm_id] = entities
                
        return hadm_entities
        
    def compute_concept_frequencies(self, hadm_entities):
        """
        Compute frequency of each concept across all admissions.
        
        Returns:
            Counter with concept -> count
            Dict with concept -> percentage
        """
        concept_counts = Counter()
        total_hadms = len(hadm_entities)
        
        # Count how many HADMs have each concept (not total occurrences)
        concept_presence = defaultdict(set)
        
        for hadm_id, entities in hadm_entities.items():
            seen_concepts = set()
            for ent_text, ent_label in entities:
                # Use entity text as concept (could also use label)
                concept = ent_text
                if concept not in seen_concepts:
                    concept_presence[concept].add(hadm_id)
                    seen_concepts.add(concept)
                    
        # Compute frequencies
        for concept, hadm_set in concept_presence.items():
            concept_counts[concept] = len(hadm_set)
            
        # Compute percentages
        concept_pcts = {
            concept: (count / total_hadms) * 100 
            for concept, count in concept_counts.items()
        }
        
        return concept_counts, concept_pcts
        
    def select_features(self, concept_pcts, concept_counts):
        """
        Select features based on frequency threshold.
        
        Args:
            concept_pcts: Dict of concept -> percentage
            concept_counts: Counter of concept -> count
            
        Returns:
            List of selected concept names
        """
        logger.info(f"Total unique concepts: {len(concept_pcts)}")
        
        # Filter by minimum frequency
        filtered = {
            c: pct for c, pct in concept_pcts.items() 
            if pct >= self.min_frequency_pct
        }
        
        logger.info(f"Concepts with ≥{self.min_frequency_pct}% frequency: {len(filtered)}")
        
        # Sort by frequency (descending) and take top N
        sorted_concepts = sorted(
            filtered.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_features]
        
        selected = [c[0] for c in sorted_concepts]
        
        logger.info(f"Selected {len(selected)} concepts as features")
        
        return selected
        
    def create_binary_features(self, hadm_entities, selected_concepts):
        """
        Create binary feature matrix (Bag-of-Concepts).
        
        Args:
            hadm_entities: Dict of HADM_ID -> list of entities
            selected_concepts: List of selected concept names
            
        Returns:
            DataFrame with HADM_ID and binary features
        """
        logger.info("Creating binary feature matrix...")
        
        # Create feature names (sanitized for column names)
        feature_names = []
        concept_to_feature = {}
        
        for concept in selected_concepts:
            # Sanitize feature name
            feat_name = f"NLP_{re.sub(r'[^a-zA-Z0-9]', '_', concept)[:50]}"
            # Handle duplicates
            if feat_name in feature_names:
                feat_name = f"{feat_name}_{len(feature_names)}"
            feature_names.append(feat_name)
            concept_to_feature[concept] = feat_name
            
        # Create matrix
        rows = []
        for hadm_id, entities in tqdm(hadm_entities.items(), desc="Creating features"):
            row = {'HADM_ID': hadm_id}
            
            # Get concepts present for this HADM
            present_concepts = set(ent[0] for ent in entities)
            
            # Set binary values
            for concept, feat_name in concept_to_feature.items():
                row[feat_name] = 1 if concept in present_concepts else 0
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Ensure HADM_ID is first column
        cols = ['HADM_ID'] + [c for c in df.columns if c != 'HADM_ID']
        df = df[cols]
        
        return df
        
    def fit_transform(self, df, text_column='CLEAN_TEXT', id_column='HADM_ID'):
        """
        Fit the pipeline on data and transform to features.
        
        Args:
            df: DataFrame with clinical notes
            text_column: Column containing text
            id_column: Column with patient/admission ID
            
        Returns:
            DataFrame with binary features
        """
        # Load model if not loaded
        if self.nlp is None:
            self.load_model()
            
        # Process notes
        hadm_entities = self.process_notes(df, text_column, id_column)
        
        # Compute frequencies
        concept_counts, concept_pcts = self.compute_concept_frequencies(hadm_entities)
        
        # Select features
        self.selected_concepts = self.select_features(concept_pcts, concept_counts)
        
        # Create binary features
        feature_df = self.create_binary_features(hadm_entities, self.selected_concepts)
        
        # Save concept statistics
        self.concept_stats = {
            'counts': concept_counts,
            'percentages': concept_pcts,
            'selected': self.selected_concepts
        }
        
        return feature_df
        
    def transform(self, df, text_column='CLEAN_TEXT', id_column='HADM_ID'):
        """
        Transform new data using fitted concepts.
        
        Args:
            df: DataFrame with clinical notes
            text_column: Column containing text  
            id_column: Column with patient/admission ID
            
        Returns:
            DataFrame with binary features
        """
        if self.selected_concepts is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
            
        if self.nlp is None:
            self.load_model()
            
        hadm_entities = self.process_notes(df, text_column, id_column)
        feature_df = self.create_binary_features(hadm_entities, self.selected_concepts)
        
        return feature_df
        
    def save(self, path):
        """Save fitted pipeline state"""
        state = {
            'selected_concepts': self.selected_concepts,
            'min_frequency_pct': self.min_frequency_pct,
            'max_features': self.max_features,
            'concept_stats': getattr(self, 'concept_stats', None)
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Pipeline saved to {path}")
        
    def load(self, path):
        """Load fitted pipeline state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.selected_concepts = state['selected_concepts']
        self.min_frequency_pct = state['min_frequency_pct']
        self.max_features = state['max_features']
        self.concept_stats = state.get('concept_stats')
        logger.info(f"Pipeline loaded from {path}")


def main():
    """Main function to run the pipeline"""
    
    # Paths
    BASE_DIR = "/Users/phandanglinh/Desktop/VRES"
    NOTES_PATH = os.path.join(BASE_DIR, "cohort/notes_before_icu_out.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "cohort/nlp_features_full")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load notes (safe - before ICU OUTTIME)
    logger.info(f"Loading notes from {NOTES_PATH}")
    notes_df = pd.read_csv(NOTES_PATH)
    logger.info(f"Loaded {len(notes_df)} notes, {notes_df['HADM_ID'].nunique()} unique HADM_IDs")
    
    # Check for CLEAN_TEXT column
    if 'CLEAN_TEXT' not in notes_df.columns:
        if 'TEXT' in notes_df.columns:
            notes_df['CLEAN_TEXT'] = notes_df['TEXT']
        else:
            raise ValueError("No text column found in notes!")
    
    # Initialize and run pipeline
    pipeline = ClinicalNLPPipeline(
        min_frequency_pct=1.0,  # At least 1% of patients
        max_features=300        # Top 300 concepts
    )
    
    # Fit and transform
    feature_df = pipeline.fit_transform(
        notes_df, 
        text_column='CLEAN_TEXT',
        id_column='HADM_ID'
    )
    
    # Save outputs
    feature_path = os.path.join(OUTPUT_DIR, "nlp_features_bag_of_concepts.csv")
    feature_df.to_csv(feature_path, index=False)
    logger.info(f"Saved features to {feature_path}")
    logger.info(f"Feature matrix shape: {feature_df.shape}")
    
    # Save pipeline for later use
    pipeline_path = os.path.join(OUTPUT_DIR, "nlp_pipeline.pkl")
    pipeline.save(pipeline_path)
    
    # Save concept statistics
    if hasattr(pipeline, 'concept_stats'):
        stats_path = os.path.join(OUTPUT_DIR, "concept_statistics.csv")
        stats_df = pd.DataFrame([
            {
                'concept': c,
                'count': pipeline.concept_stats['counts'][c],
                'percentage': pipeline.concept_stats['percentages'][c],
                'selected': c in pipeline.concept_stats['selected']
            }
            for c in pipeline.concept_stats['percentages']
        ]).sort_values('count', ascending=False)
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved concept statistics to {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("NLP PIPELINE SUMMARY")
    print("="*60)
    print(f"Input notes: {len(notes_df)}")
    print(f"Unique HADM_IDs: {notes_df['HADM_ID'].nunique()}")
    print(f"Output features: {feature_df.shape[1] - 1}")  # -1 for HADM_ID
    print(f"Output samples: {len(feature_df)}")
    print(f"Min frequency threshold: {pipeline.min_frequency_pct}%")
    print("="*60)
    
    # Show top concepts
    if hasattr(pipeline, 'concept_stats'):
        print("\nTop 20 most frequent concepts:")
        top_concepts = sorted(
            pipeline.concept_stats['percentages'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        for i, (concept, pct) in enumerate(top_concepts, 1):
            count = pipeline.concept_stats['counts'][concept]
            print(f"  {i:2d}. {concept[:40]:<40} - {pct:.1f}% ({count} patients)")
    
    return feature_df


if __name__ == "__main__":
    main()

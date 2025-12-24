"""
Clinical NLP Pipeline for MIMIC-III Discharge Summaries

This script processes discharge summaries from MIMIC-III NOTEEVENTS table:
1. Text Cleaning - Remove noise, normalize text
2. Sentence Segmentation - Use spaCy dependency parser
3. Named Entity Recognition (NER) - Extract Problem, Treatment, Test entities
4. Negation Detection - Use medspaCy/negspaCy to detect negated entities
5. UMLS Concept Mapping - Map entities to UMLS CUIs

Author: VRES Project
Date: December 2025
"""

import os
import re
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import spacy
from spacy.tokens import Doc, Span

# Clinical NLP libraries
try:
    import medspacy
    from medspacy.ner import TargetRule
    from medspacy.context import ConTextComponent
    MEDSPACY_AVAILABLE = True
except ImportError:
    MEDSPACY_AVAILABLE = False
    print("Warning: medspaCy not installed. Install with: pip install medspacy")

# UMLS mapping (using scispacy for entity linking)
try:
    import scispacy
    from scispacy.linking import EntityLinker
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    print("Warning: scispaCy not installed. Install with: pip install scispacy")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClinicalEntity:
    """Represents an extracted clinical entity."""
    text: str
    label: str  # PROBLEM, TREATMENT, TEST
    start: int
    end: int
    is_negated: bool = False
    is_hypothetical: bool = False
    is_historical: bool = False
    is_family: bool = False
    umls_cui: Optional[str] = None
    umls_name: Optional[str] = None
    confidence: float = 1.0
    sentence: Optional[str] = None


@dataclass 
class ProcessedNote:
    """Represents a processed clinical note."""
    row_id: str
    subject_id: str
    hadm_id: str
    original_text: str
    cleaned_text: str
    sentences: List[str] = field(default_factory=list)
    entities: List[ClinicalEntity] = field(default_factory=list)
    

class TextCleaner:
    """
    Clean and normalize clinical text.
    """
    
    def __init__(self):
        # Patterns for cleaning
        self.patterns = {
            # Remove de-identification placeholders
            'deid': re.compile(r'\[\*\*.*?\*\*\]', re.DOTALL),
            # Remove multiple spaces
            'multi_space': re.compile(r'\s+'),
            # Remove multiple newlines
            'multi_newline': re.compile(r'\n{3,}'),
            # Remove special characters but keep medical symbols
            'special_chars': re.compile(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\/\%\+\=\<\>\'\"\°\±]'),
            # Normalize numbers with units
            'units': re.compile(r'(\d+)\s*(mg|ml|mcg|g|kg|cm|mm|L|mL|mmHg|bpm)', re.IGNORECASE),
        }
        
        # Common abbreviation expansions
        self.abbreviations = {
            'pt': 'patient',
            'pts': 'patients',
            'hx': 'history',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'sx': 'symptoms',
            'c/o': 'complains of',
            'w/': 'with',
            'w/o': 'without',
            's/p': 'status post',
            'b/l': 'bilateral',
            'h/o': 'history of',
            'f/u': 'follow up',
            'prn': 'as needed',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qd': 'once daily',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'npo': 'nothing by mouth',
        }
    
    def clean(self, text: str, expand_abbreviations: bool = False) -> str:
        """
        Clean clinical text.
        
        Args:
            text: Raw clinical text
            expand_abbreviations: Whether to expand common abbreviations
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Remove de-identification markers [** **]
        text = self.patterns['deid'].sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = self.patterns['multi_newline'].sub('\n\n', text)
        
        # Normalize whitespace
        text = self.patterns['multi_space'].sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Optionally expand abbreviations
        if expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            # Match whole words only
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text


class SentenceSegmenter:
    """
    Segment clinical text into sentences using spaCy dependency parser.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize sentence segmenter.
        
        Args:
            model_name: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Customize sentence boundaries for clinical text
        self._customize_sentencizer()
    
    def _customize_sentencizer(self):
        """Add custom sentence boundary rules for clinical text."""
        # Note: Custom sentencizer disabled to avoid conflicts with parser
        # The default spaCy sentence segmentation works well for clinical text
        pass
    
    def segment(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def segment_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Segment text into sentences with character offsets.
        
        Args:
            text: Input text
            
        Returns:
            List of (sentence, start, end) tuples
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        sentences = [
            (sent.text.strip(), sent.start_char, sent.end_char) 
            for sent in doc.sents if sent.text.strip()
        ]
        return sentences


class ClinicalNER:
    """
    Named Entity Recognition for clinical text.
    Extracts Problem, Treatment, and Test entities.
    """
    
    def __init__(self, use_medspacy: bool = True):
        """
        Initialize NER component.
        
        Args:
            use_medspacy: Whether to use medspaCy for clinical NER
        """
        self.use_medspacy = use_medspacy and MEDSPACY_AVAILABLE
        
        if self.use_medspacy:
            self._init_medspacy()
        else:
            self._init_scispacy()
    
    def _init_medspacy(self):
        """Initialize medspaCy clinical NLP pipeline."""
        logger.info("Initializing medspaCy NER pipeline...")
        
        # Load medspaCy with clinical components
        self.nlp = medspacy.load(enable=["tokenizer", "sentencizer"])
        
        # Add target matcher for clinical entities
        from medspacy.ner import TargetMatcher
        target_matcher = self.nlp.add_pipe("medspacy_target_matcher")
        
        # Define target rules for Problems, Treatments, Tests
        target_rules = self._get_clinical_target_rules()
        target_matcher.add(target_rules)
        
        logger.info("medspaCy NER pipeline initialized")
    
    def _init_scispacy(self):
        """Initialize scispaCy for biomedical NER."""
        logger.info("Initializing scispaCy NER pipeline...")
        
        try:
            # Try to load biomedical model
            self.nlp = spacy.load("en_core_sci_sm")
        except OSError:
            logger.warning("en_core_sci_sm not found. Using en_core_web_sm...")
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info("scispaCy NER pipeline initialized")
    
    def _get_clinical_target_rules(self) -> List[TargetRule]:
        """Define rules for extracting clinical entities."""
        rules = []
        
        # Problems (diseases, symptoms, conditions)
        problem_terms = [
            "pneumonia", "diabetes", "hypertension", "infection", "fever",
            "pain", "cough", "dyspnea", "edema", "sepsis", "stroke",
            "heart failure", "renal failure", "respiratory failure",
            "myocardial infarction", "atrial fibrillation", "copd",
            "asthma", "cancer", "tumor", "fracture", "bleeding",
            "anemia", "hypotension", "tachycardia", "bradycardia",
            "anxiety", "depression", "delirium", "encephalopathy",
            "nausea", "vomiting", "diarrhea", "constipation",
            "headache", "dizziness", "weakness", "fatigue"
        ]
        
        for term in problem_terms:
            rules.append(TargetRule(
                literal=term,
                category="PROBLEM",
                pattern=None
            ))
        
        # Treatments (medications, procedures, therapies)
        treatment_terms = [
            "aspirin", "metformin", "insulin", "lisinopril", "atorvastatin",
            "metoprolol", "furosemide", "heparin", "warfarin", "vancomycin",
            "ceftriaxone", "morphine", "fentanyl", "acetaminophen", "ibuprofen",
            "surgery", "intubation", "ventilation", "dialysis", "transfusion",
            "chemotherapy", "radiation", "physical therapy", "oxygen therapy",
            "antibiotics", "diuretics", "anticoagulation", "vasopressors"
        ]
        
        for term in treatment_terms:
            rules.append(TargetRule(
                literal=term,
                category="TREATMENT",
                pattern=None
            ))
        
        # Tests (lab tests, imaging, procedures)
        test_terms = [
            "ct scan", "mri", "x-ray", "ultrasound", "echocardiogram",
            "ekg", "ecg", "cbc", "bmp", "cmp", "lfts", "troponin",
            "bnp", "creatinine", "bun", "hemoglobin", "hematocrit",
            "wbc", "platelet", "inr", "ptt", "blood culture",
            "urine culture", "lumbar puncture", "biopsy", "endoscopy",
            "colonoscopy", "bronchoscopy", "blood gas", "lactate"
        ]
        
        for term in test_terms:
            rules.append(TargetRule(
                literal=term,
                category="TEST",
                pattern=None
            ))
        
        return rules
    
    def extract_entities(self, text: str) -> List[ClinicalEntity]:
        """
        Extract clinical entities from text.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of ClinicalEntity objects
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map entity labels to our categories
            label = self._map_entity_label(ent.label_)
            
            entity = ClinicalEntity(
                text=ent.text,
                label=label,
                start=ent.start_char,
                end=ent.end_char
            )
            entities.append(entity)
        
        return entities
    
    def _map_entity_label(self, label: str) -> str:
        """Map spaCy/medspaCy entity labels to our categories."""
        label_mapping = {
            # medspaCy labels
            "PROBLEM": "PROBLEM",
            "TREATMENT": "TREATMENT", 
            "TEST": "TEST",
            # scispaCy/biomedical labels
            "DISEASE": "PROBLEM",
            "CHEMICAL": "TREATMENT",
            "GENE_OR_GENE_PRODUCT": "TEST",
            # Default
            "ENTITY": "PROBLEM"
        }
        return label_mapping.get(label, "OTHER")


class NegationDetector:
    """
    Detect negation and context modifiers for clinical entities.
    Uses medspaCy ConText algorithm (based on NegEx/ConText).
    """
    
    def __init__(self):
        """Initialize negation detector."""
        if MEDSPACY_AVAILABLE:
            self._init_medspacy_context()
        else:
            self._init_simple_negation()
    
    def _init_medspacy_context(self):
        """Initialize medspaCy ConText for negation detection."""
        logger.info("Initializing medspaCy ConText for negation detection...")
        
        self.nlp = medspacy.load()
        
        # ConText is automatically added by medspacy.load()
        # It detects: negation, historical, hypothetical, family history
        
        self.use_medspacy = True
        logger.info("medspaCy ConText initialized")
    
    def _init_simple_negation(self):
        """Initialize simple rule-based negation detection."""
        logger.info("Using simple rule-based negation detection...")
        
        self.use_medspacy = False
        
        # Negation triggers (NegEx-style)
        self.negation_triggers = [
            # Pre-negation
            r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bdenies\b', r'\bdenied\b',
            r'\bnegative\b', r'\bnever\b', r'\bfree of\b', r'\babsent\b',
            r'\bno evidence of\b', r'\brules out\b', r'\bruled out\b',
            r'\bno sign of\b', r'\bno signs of\b', r'\bno history of\b',
            r'\bno further\b', r'\bnot have\b', r'\bdoes not have\b',
            # Post-negation
            r'\bunlikely\b', r'\bwas ruled out\b', r'\bhas been ruled out\b'
        ]
        
        # Hypothetical triggers
        self.hypothetical_triggers = [
            r'\bif\b', r'\bshould\b', r'\bcould\b', r'\bmay\b', r'\bmight\b',
            r'\bpossible\b', r'\bsuspected\b', r'\brule out\b', r'\bevaluate for\b'
        ]
        
        # Historical triggers
        self.historical_triggers = [
            r'\bhistory of\b', r'\bhistory\b', r'\bpast\b', r'\bprevious\b',
            r'\bprior\b', r'\bformer\b', r'\bin the past\b'
        ]
        
        # Family history triggers
        self.family_triggers = [
            r'\bfamily history\b', r'\bfamily\b', r'\bmother\b', r'\bfather\b',
            r'\bsibling\b', r'\bbrother\b', r'\bsister\b', r'\bgrandmother\b',
            r'\bgrandfather\b'
        ]
    
    def detect_negation(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """
        Detect negation and context for entities.
        
        Args:
            text: Source text
            entities: List of extracted entities
            
        Returns:
            Updated entities with negation flags
        """
        if self.use_medspacy:
            return self._detect_with_medspacy(text, entities)
        else:
            return self._detect_with_rules(text, entities)
    
    def _detect_with_medspacy(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """Use medspaCy ConText for negation detection."""
        doc = self.nlp(text)
        
        # Create a mapping from character offsets to entities
        offset_to_entity = {(e.start, e.end): e for e in entities}
        
        for ent in doc.ents:
            key = (ent.start_char, ent.end_char)
            if key in offset_to_entity:
                entity = offset_to_entity[key]
                
                # Check context modifiers
                if hasattr(ent._, 'is_negated'):
                    entity.is_negated = ent._.is_negated
                if hasattr(ent._, 'is_hypothetical'):
                    entity.is_hypothetical = ent._.is_hypothetical
                if hasattr(ent._, 'is_historical'):
                    entity.is_historical = ent._.is_historical
                if hasattr(ent._, 'is_family'):
                    entity.is_family = ent._.is_family
        
        return entities
    
    def _detect_with_rules(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """Use simple rule-based negation detection."""
        text_lower = text.lower()
        
        for entity in entities:
            # Get context window around entity (50 chars before and after)
            start = max(0, entity.start - 50)
            end = min(len(text), entity.end + 50)
            context = text_lower[start:end]
            
            # Check negation
            for pattern in self.negation_triggers:
                if re.search(pattern, context):
                    entity.is_negated = True
                    break
            
            # Check hypothetical
            for pattern in self.hypothetical_triggers:
                if re.search(pattern, context):
                    entity.is_hypothetical = True
                    break
            
            # Check historical
            for pattern in self.historical_triggers:
                if re.search(pattern, context):
                    entity.is_historical = True
                    break
            
            # Check family history
            for pattern in self.family_triggers:
                if re.search(pattern, context):
                    entity.is_family = True
                    break
        
        return entities


class UMLSMapper:
    """
    Map clinical entities to UMLS Concept Unique Identifiers (CUIs).
    Uses scispaCy entity linking to UMLS.
    """
    
    def __init__(self, linker_name: str = "umls"):
        """
        Initialize UMLS mapper.
        
        Args:
            linker_name: Name of the knowledge base to link to ("umls", "mesh", "rxnorm")
        """
        self.linker_name = linker_name
        self.linker = None
        self.nlp = None
        
        if SCISPACY_AVAILABLE:
            self._init_entity_linker()
        else:
            logger.warning("scispaCy not available. UMLS linking disabled.")
    
    def _init_entity_linker(self):
        """Initialize scispaCy entity linker."""
        logger.info(f"Initializing scispaCy entity linker ({self.linker_name})...")
        
        try:
            # Load scientific/biomedical model
            self.nlp = spacy.load("en_core_sci_sm")
            
            # Add entity linker
            # Note: First run will download the knowledge base (~1GB for UMLS)
            self.nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": True,
                    "linker_name": self.linker_name
                }
            )
            
            self.linker = self.nlp.get_pipe("scispacy_linker")
            logger.info("Entity linker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize entity linker: {e}")
            logger.info("UMLS linking will be disabled")
            self.nlp = None
    
    def map_to_umls(self, text: str, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """
        Map entities to UMLS CUIs.
        
        Args:
            text: Source text
            entities: List of extracted entities
            
        Returns:
            Updated entities with UMLS CUIs
        """
        if self.nlp is None:
            return entities
        
        try:
            doc = self.nlp(text)
            
            # Create entity linker results mapping
            for ent in doc.ents:
                # Find matching entity in our list
                for entity in entities:
                    if (ent.start_char == entity.start and 
                        ent.end_char == entity.end):
                        
                        # Get linked entities
                        if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                            # Get top linked entity (highest confidence)
                            cui, score = ent._.kb_ents[0]
                            entity.umls_cui = cui
                            entity.confidence = score
                            
                            # Get concept name from linker
                            if self.linker and cui in self.linker.kb.cui_to_entity:
                                concept = self.linker.kb.cui_to_entity[cui]
                                entity.umls_name = concept.canonical_name
                        break
        
        except Exception as e:
            logger.error(f"Error during UMLS mapping: {e}")
        
        return entities
    
    def batch_map(self, texts: List[str], entities_list: List[List[ClinicalEntity]]) -> List[List[ClinicalEntity]]:
        """
        Batch map entities to UMLS CUIs.
        
        Args:
            texts: List of source texts
            entities_list: List of entity lists (one per text)
            
        Returns:
            Updated entities with UMLS CUIs
        """
        if self.nlp is None:
            return entities_list
        
        results = []
        for text, entities in zip(texts, entities_list):
            updated = self.map_to_umls(text, entities)
            results.append(updated)
        
        return results


class ClinicalNLPPipeline:
    """
    Complete NLP pipeline for processing clinical notes.
    Combines all components: cleaning, segmentation, NER, negation, UMLS mapping.
    """
    
    def __init__(
        self,
        db_path: str,
        use_medspacy: bool = True,
        use_umls: bool = True,
        expand_abbreviations: bool = False
    ):
        """
        Initialize the complete NLP pipeline.
        
        Args:
            db_path: Path to MIMIC-III database
            use_medspacy: Whether to use medspaCy for clinical NER
            use_umls: Whether to enable UMLS mapping
            expand_abbreviations: Whether to expand medical abbreviations
        """
        self.db_path = db_path
        self.expand_abbreviations = expand_abbreviations
        
        # Initialize components
        logger.info("Initializing Clinical NLP Pipeline...")
        
        self.text_cleaner = TextCleaner()
        self.sentence_segmenter = SentenceSegmenter()
        self.ner = ClinicalNER(use_medspacy=use_medspacy)
        self.negation_detector = NegationDetector()
        
        if use_umls:
            self.umls_mapper = UMLSMapper()
        else:
            self.umls_mapper = None
        
        logger.info("Pipeline initialization complete")
    
    def process_note(self, row_id: str, subject_id: str, hadm_id: str, text: str) -> ProcessedNote:
        """
        Process a single clinical note through the complete pipeline.
        
        Args:
            row_id: Note row ID
            subject_id: Patient subject ID
            hadm_id: Hospital admission ID
            text: Raw note text
            
        Returns:
            ProcessedNote object with all extracted information
        """
        # 1. Clean text
        cleaned_text = self.text_cleaner.clean(text, self.expand_abbreviations)
        
        # 2. Segment into sentences
        sentences = self.sentence_segmenter.segment(cleaned_text)
        
        # 3. Extract entities (NER)
        entities = self.ner.extract_entities(cleaned_text)
        
        # 4. Detect negation
        entities = self.negation_detector.detect_negation(cleaned_text, entities)
        
        # 5. Map to UMLS (if enabled)
        if self.umls_mapper:
            entities = self.umls_mapper.map_to_umls(cleaned_text, entities)
        
        # Add sentence context to entities
        sentences_with_spans = self.sentence_segmenter.segment_with_spans(cleaned_text)
        for entity in entities:
            for sent_text, sent_start, sent_end in sentences_with_spans:
                if sent_start <= entity.start < sent_end:
                    entity.sentence = sent_text
                    break
        
        return ProcessedNote(
            row_id=row_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            original_text=text,
            cleaned_text=cleaned_text,
            sentences=sentences,
            entities=entities
        )
    
    def process_discharge_summaries(
        self,
        limit: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process all discharge summaries from MIMIC-III.
        
        Args:
            limit: Maximum number of notes to process (None for all)
            output_path: Path to save results CSV
            
        Returns:
            DataFrame with processed entities
        """
        logger.info("Fetching discharge summaries from database...")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # Query discharge summaries
        query = """
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, TEXT 
        FROM NOTEEVENTS 
        WHERE CATEGORY = 'Discharge summary'
        AND TEXT IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Processing {len(df)} discharge summaries...")
        
        # Process notes
        all_results = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing note {idx + 1}/{len(df)}...")
            
            try:
                processed = self.process_note(
                    row_id=str(row['ROW_ID']),
                    subject_id=str(row['SUBJECT_ID']),
                    hadm_id=str(row['HADM_ID']),
                    text=row['TEXT']
                )
                
                # Convert entities to records
                for entity in processed.entities:
                    result = {
                        'row_id': processed.row_id,
                        'subject_id': processed.subject_id,
                        'hadm_id': processed.hadm_id,
                        'entity_text': entity.text,
                        'entity_label': entity.label,
                        'is_negated': entity.is_negated,
                        'is_hypothetical': entity.is_hypothetical,
                        'is_historical': entity.is_historical,
                        'is_family': entity.is_family,
                        'umls_cui': entity.umls_cui,
                        'umls_name': entity.umls_name,
                        'confidence': entity.confidence,
                        'sentence': entity.sentence,
                        'start_char': entity.start,
                        'end_char': entity.end
                    }
                    
                    # Create modified entity name for negated entities
                    if entity.is_negated:
                        result['modified_entity'] = f"Absent_{entity.text.replace(' ', '_')}"
                    else:
                        result['modified_entity'] = entity.text
                    
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing note {row['ROW_ID']}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of extracted entities."""
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        
        if len(df) == 0:
            logger.info("No entities extracted")
            return
        
        logger.info(f"Total entities extracted: {len(df)}")
        logger.info(f"Unique notes processed: {df['row_id'].nunique()}")
        logger.info(f"Unique patients: {df['subject_id'].nunique()}")
        
        logger.info("\nEntities by category:")
        for label, count in df['entity_label'].value_counts().items():
            logger.info(f"  {label}: {count}")
        
        logger.info("\nNegation statistics:")
        logger.info(f"  Negated: {df['is_negated'].sum()}")
        logger.info(f"  Hypothetical: {df['is_hypothetical'].sum()}")
        logger.info(f"  Historical: {df['is_historical'].sum()}")
        logger.info(f"  Family history: {df['is_family'].sum()}")
        
        if 'umls_cui' in df.columns:
            mapped = df['umls_cui'].notna().sum()
            logger.info(f"\nUMLS mapping: {mapped}/{len(df)} ({100*mapped/len(df):.1f}%)")
        
        logger.info("\nTop 10 entities:")
        for entity, count in df['entity_text'].value_counts().head(10).items():
            logger.info(f"  {entity}: {count}")


def create_feature_vectors(
    entities_df: pd.DataFrame,
    include_negated: bool = True,
    use_cui: bool = True
) -> pd.DataFrame:
    """
    Create feature vectors from extracted entities for machine learning.
    
    Args:
        entities_df: DataFrame with extracted entities
        include_negated: Whether to include negated entities as separate features
        use_cui: Whether to use UMLS CUIs instead of entity text
        
    Returns:
        DataFrame with one row per admission, columns for entity counts
    """
    # Determine feature column
    if use_cui and 'umls_cui' in entities_df.columns:
        feature_col = 'umls_cui'
        entities_df = entities_df[entities_df['umls_cui'].notna()]
    else:
        feature_col = 'entity_text'
    
    # Create feature name
    if include_negated:
        entities_df['feature_name'] = entities_df.apply(
            lambda x: f"Absent_{x[feature_col]}" if x['is_negated'] else x[feature_col],
            axis=1
        )
    else:
        # Filter out negated entities
        entities_df = entities_df[~entities_df['is_negated']]
        entities_df['feature_name'] = entities_df[feature_col]
    
    # Pivot to create feature matrix
    feature_matrix = entities_df.pivot_table(
        index='hadm_id',
        columns='feature_name',
        values='row_id',
        aggfunc='count',
        fill_value=0
    )
    
    return feature_matrix


def main():
    """Main function to run the clinical NLP pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical NLP Pipeline for MIMIC-III')
    parser.add_argument(
        '--db-path',
        type=str,
        default='/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db',
        help='Path to MIMIC-III database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='clinical_entities.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of notes to process'
    )
    parser.add_argument(
        '--no-umls',
        action='store_true',
        help='Disable UMLS mapping'
    )
    parser.add_argument(
        '--expand-abbrev',
        action='store_true',
        help='Expand medical abbreviations'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ClinicalNLPPipeline(
        db_path=args.db_path,
        use_medspacy=MEDSPACY_AVAILABLE,
        use_umls=not args.no_umls,
        expand_abbreviations=args.expand_abbrev
    )
    
    # Process notes
    results = pipeline.process_discharge_summaries(
        limit=args.limit,
        output_path=args.output
    )
    
    # Create feature vectors
    if len(results) > 0:
        features = create_feature_vectors(results)
        features.to_csv('clinical_features.csv')
        logger.info(f"Feature matrix shape: {features.shape}")


if __name__ == '__main__':
    main()

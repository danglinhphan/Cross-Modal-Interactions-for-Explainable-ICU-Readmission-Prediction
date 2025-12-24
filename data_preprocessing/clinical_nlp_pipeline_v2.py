"""
Improved Clinical NLP Pipeline for MIMIC-III Discharge Summaries

This script processes discharge summaries from MIMIC-III NOTEEVENTS table with:
1. Text Cleaning - Remove noise, normalize text
2. Sentence Segmentation - Use medspaCy PyRuSH for clinical text
3. Named Entity Recognition (NER) - Extract Problem, Treatment, Test using medspaCy TargetMatcher
4. Negation Detection - Use medspaCy ConText algorithm
5. UMLS Concept Mapping - QuickUMLS integration

Author: VRES Project
Date: December 2025
"""

import os
import re
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import pandas as pd

# Setup logging - reduce noise from medspaCy
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress PyRuSH debug messages
logging.getLogger('PyRuSH').setLevel(logging.WARNING)

# Import spaCy
import spacy
from spacy.tokens import Doc, Span

# Clinical NLP libraries
try:
    import medspacy
    from medspacy.ner import TargetRule
    MEDSPACY_AVAILABLE = True
except ImportError:
    MEDSPACY_AVAILABLE = False
    logger.warning("medspaCy not installed. Install with: pip install medspacy")

# UMLS mapping library
try:
    from quickumls import QuickUMLS
    QUICKUMLS_AVAILABLE = True
except ImportError:
    QUICKUMLS_AVAILABLE = False
    logger.warning("QuickUMLS not installed. Install with: pip install quickumls")

# scispacy UMLS linker (alternative)
try:
    import scispacy
    from scispacy.linking import EntityLinker
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    logger.warning("scispacy not installed. Install with: pip install scispacy")


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
    is_uncertain: bool = False
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
    """Clean and normalize clinical text."""
    
    def __init__(self):
        self.patterns = {
            'deid': re.compile(r'\[\*\*.*?\*\*\]', re.DOTALL),
            'multi_space': re.compile(r'[ \t]+'),
            'multi_newline': re.compile(r'\n{3,}'),
        }
        
        self.abbreviations = {
            'pt': 'patient', 'pts': 'patients', 'hx': 'history',
            'dx': 'diagnosis', 'tx': 'treatment', 'rx': 'prescription',
            'sx': 'symptoms', 'c/o': 'complains of', 'w/': 'with',
            'w/o': 'without', 's/p': 'status post', 'b/l': 'bilateral',
            'h/o': 'history of', 'f/u': 'follow up',
        }
    
    def clean(self, text: str, expand_abbreviations: bool = False) -> str:
        """Clean clinical text."""
        if not text:
            return ""
        
        text = str(text)
        text = self.patterns['deid'].sub(' ', text)
        text = self.patterns['multi_newline'].sub('\n\n', text)
        text = self.patterns['multi_space'].sub(' ', text)
        text = text.strip()
        
        if expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text


class ClinicalNLPPipeline:
    """
    Complete NLP pipeline for clinical notes using medspaCy.
    """
    
    def __init__(
        self,
        db_path: str,
        expand_abbreviations: bool = False,
        use_umls: bool = False,
        quickumls_path: Optional[str] = None
    ):
        """
        Initialize the complete NLP pipeline.
        
        Args:
            db_path: Path to MIMIC-III database
            expand_abbreviations: Whether to expand medical abbreviations
            use_umls: Whether to use UMLS concept mapping
            quickumls_path: Path to QuickUMLS installation
        """
        self.db_path = db_path
        self.expand_abbreviations = expand_abbreviations
        self.use_umls = use_umls
        self.quickumls_path = quickumls_path
        self.quickumls = None
        self.text_cleaner = TextCleaner()
        
        logger.info("Initializing Clinical NLP Pipeline with medspaCy...")
        
        if not MEDSPACY_AVAILABLE:
            raise ImportError("medspaCy is required. Install with: pip install medspacy")
        
        # Initialize medspaCy pipeline - use full load for ConText to work properly
        self.nlp = medspacy.load()
        
        # Get existing target matcher and add comprehensive clinical terms
        self._setup_target_matcher()
        
        logger.info(f"Pipeline components: {self.nlp.pipe_names}")
        
        # Initialize UMLS mapper if requested
        if self.use_umls and QUICKUMLS_AVAILABLE and self.quickumls_path:
            self._setup_umls_mapper()
        elif self.use_umls and not QUICKUMLS_AVAILABLE:
            logger.warning("UMLS requested but QuickUMLS not available")
        elif self.use_umls and not self.quickumls_path:
            logger.warning("UMLS requested but quickumls_path not provided")
        
        logger.info("Pipeline initialization complete")
    
    def _setup_target_matcher(self):
        """Setup target matcher with comprehensive clinical terminology."""
        from medspacy.ner import TargetMatcher
        
        # Get existing target_matcher from medspacy.load()
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        
        rules = []
        
        # ==================== PROBLEMS ====================
        # Cardiovascular
        cardiovascular_problems = [
            "heart failure", "congestive heart failure", "chf",
            "myocardial infarction", "mi", "heart attack", "stemi", "nstemi",
            "atrial fibrillation", "afib", "a-fib", "atrial flutter",
            "ventricular tachycardia", "v-tach", "ventricular fibrillation", "v-fib",
            "hypertension", "htn", "high blood pressure", "hypotension",
            "coronary artery disease", "cad", "angina", "chest pain",
            "cardiomyopathy", "pericarditis", "endocarditis", "myocarditis",
            "aortic stenosis", "mitral regurgitation", "valve disease",
            "deep vein thrombosis", "dvt", "pulmonary embolism", "pe",
            "peripheral vascular disease", "pvd", "aortic aneurysm"
        ]
        
        # Respiratory
        respiratory_problems = [
            "pneumonia", "community acquired pneumonia", "cap",
            "hospital acquired pneumonia", "hap", "ventilator associated pneumonia", "vap",
            "copd", "chronic obstructive pulmonary disease", "emphysema",
            "asthma", "bronchitis", "bronchiolitis", "bronchiectasis",
            "respiratory failure", "acute respiratory distress syndrome", "ards",
            "pulmonary edema", "pleural effusion", "pneumothorax", "hemothorax",
            "pulmonary fibrosis", "interstitial lung disease", "ild",
            "dyspnea", "shortness of breath", "sob", "hypoxia", "hypoxemia",
            "respiratory distress", "tachypnea", "wheezing", "stridor",
            "cough", "hemoptysis", "aspiration"
        ]
        
        # Neurological
        neurological_problems = [
            "stroke", "cva", "cerebrovascular accident", "ischemic stroke", "hemorrhagic stroke",
            "tia", "transient ischemic attack", "intracranial hemorrhage", "ich",
            "subarachnoid hemorrhage", "sah", "subdural hematoma", "sdh",
            "epidural hematoma", "edh", "intraparenchymal hemorrhage",
            "seizure", "epilepsy", "status epilepticus", "convulsion",
            "encephalopathy", "hepatic encephalopathy", "metabolic encephalopathy",
            "delirium", "altered mental status", "ams", "confusion",
            "dementia", "alzheimer", "parkinson", "parkinsons",
            "meningitis", "encephalitis", "guillain barre", "gbs",
            "neuropathy", "peripheral neuropathy", "multiple sclerosis", "ms",
            "headache", "migraine", "vertigo", "dizziness", "syncope",
            "coma", "obtunded", "lethargic", "unresponsive"
        ]
        
        # Infectious
        infectious_problems = [
            "sepsis", "septic shock", "severe sepsis", "bacteremia",
            "infection", "infectious", "urinary tract infection", "uti",
            "pyelonephritis", "cystitis", "cellulitis", "abscess",
            "osteomyelitis", "endocarditis", "meningitis", "encephalitis",
            "clostridium difficile", "c diff", "cdiff", "c. difficile",
            "mrsa", "vre", "esbl", "pseudomonas", "staph aureus",
            "tuberculosis", "tb", "hiv", "aids", "hepatitis",
            "influenza", "flu", "covid", "covid-19", "coronavirus"
        ]
        
        # Gastrointestinal
        gi_problems = [
            "gi bleed", "gastrointestinal bleeding", "upper gi bleed", "lower gi bleed",
            "melena", "hematochezia", "hematemesis", "bloody stool",
            "pancreatitis", "acute pancreatitis", "chronic pancreatitis",
            "cholecystitis", "cholelithiasis", "choledocholithiasis", "cholangitis",
            "hepatitis", "cirrhosis", "liver failure", "hepatic failure",
            "bowel obstruction", "small bowel obstruction", "sbo", "ileus",
            "appendicitis", "diverticulitis", "colitis", "inflammatory bowel disease", "ibd",
            "crohns", "crohn's disease", "ulcerative colitis",
            "peptic ulcer", "gastric ulcer", "duodenal ulcer", "gerd",
            "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain"
        ]
        
        # Renal
        renal_problems = [
            "acute kidney injury", "aki", "acute renal failure", "arf",
            "chronic kidney disease", "ckd", "esrd", "end stage renal disease",
            "renal failure", "kidney failure", "nephropathy", "glomerulonephritis",
            "nephrotic syndrome", "nephritic syndrome", "pyelonephritis",
            "hydronephrosis", "renal cyst", "polycystic kidney",
            "hyperkalemia", "hypokalemia", "hypernatremia", "hyponatremia",
            "acidosis", "metabolic acidosis", "respiratory acidosis",
            "alkalosis", "metabolic alkalosis", "respiratory alkalosis",
            "uremia", "azotemia", "oliguria", "anuria", "hematuria", "proteinuria"
        ]
        
        # Endocrine/Metabolic
        endocrine_problems = [
            "diabetes", "diabetes mellitus", "dm", "type 1 diabetes", "type 2 diabetes",
            "diabetic ketoacidosis", "dka", "hyperosmolar", "hhs",
            "hypoglycemia", "hyperglycemia", "glucose intolerance",
            "hypothyroidism", "hyperthyroidism", "thyroid disease", "goiter",
            "adrenal insufficiency", "addisons", "cushings",
            "hyponatremia", "hypernatremia", "siadh",
            "hypocalcemia", "hypercalcemia", "osteoporosis",
            "obesity", "malnutrition", "failure to thrive"
        ]
        
        # Hematologic/Oncologic
        hematologic_problems = [
            "anemia", "iron deficiency anemia", "b12 deficiency",
            "pancytopenia", "leukopenia", "neutropenia", "thrombocytopenia",
            "leukocytosis", "thrombocytosis", "polycythemia",
            "coagulopathy", "dic", "disseminated intravascular coagulation",
            "bleeding", "hemorrhage", "hematoma",
            "cancer", "malignancy", "tumor", "mass", "neoplasm",
            "leukemia", "lymphoma", "myeloma", "carcinoma",
            "metastatic", "metastasis", "recurrence"
        ]
        
        # Other common problems
        other_problems = [
            "pain", "acute pain", "chronic pain", "back pain", "abdominal pain",
            "fever", "febrile", "hypothermia", "chills", "rigors",
            "edema", "peripheral edema", "pulmonary edema", "anasarca",
            "rash", "pruritus", "urticaria", "cellulitis",
            "anxiety", "depression", "psychosis", "bipolar", "schizophrenia",
            "alcohol withdrawal", "drug overdose", "intoxication",
            "fall", "trauma", "fracture", "laceration", "contusion",
            "weakness", "fatigue", "malaise", "lethargy",
            "weight loss", "weight gain", "dehydration"
        ]
        
        # Combine all problems
        all_problems = (cardiovascular_problems + respiratory_problems + 
                       neurological_problems + infectious_problems + gi_problems +
                       renal_problems + endocrine_problems + hematologic_problems + other_problems)
        
        for term in all_problems:
            rules.append(TargetRule(literal=term, category="PROBLEM"))
        
        # ==================== TREATMENTS ====================
        # Cardiovascular medications
        cv_medications = [
            "aspirin", "asa", "clopidogrel", "plavix", "ticagrelor", "prasugrel",
            "heparin", "lovenox", "enoxaparin", "warfarin", "coumadin",
            "apixaban", "eliquis", "rivaroxaban", "xarelto", "dabigatran",
            "metoprolol", "lopressor", "carvedilol", "atenolol", "propranolol",
            "lisinopril", "enalapril", "ramipril", "losartan", "valsartan",
            "amlodipine", "diltiazem", "verapamil", "nifedipine",
            "furosemide", "lasix", "bumetanide", "spironolactone", "aldactone",
            "hydralazine", "nitroprusside", "nitroglycerin", "ntg",
            "digoxin", "amiodarone", "sotalol", "flecainide",
            "atorvastatin", "lipitor", "simvastatin", "rosuvastatin",
            "dobutamine", "dopamine", "norepinephrine", "levophed",
            "epinephrine", "vasopressin", "phenylephrine"
        ]
        
        # Antibiotics
        antibiotics = [
            "vancomycin", "linezolid", "daptomycin",
            "ceftriaxone", "rocephin", "cefepime", "ceftazidime", "cefazolin",
            "piperacillin", "zosyn", "piperacillin-tazobactam",
            "meropenem", "imipenem", "ertapenem",
            "ciprofloxacin", "levofloxacin", "levaquin", "moxifloxacin",
            "metronidazole", "flagyl", "clindamycin",
            "azithromycin", "zithromax", "clarithromycin", "erythromycin",
            "amoxicillin", "augmentin", "ampicillin", "penicillin",
            "trimethoprim", "bactrim", "sulfamethoxazole",
            "doxycycline", "tetracycline", "gentamicin", "tobramycin",
            "fluconazole", "diflucan", "micafungin", "amphotericin",
            "acyclovir", "valacyclovir", "oseltamivir", "tamiflu"
        ]
        
        # Pain medications
        pain_medications = [
            "morphine", "hydromorphone", "dilaudid", "fentanyl",
            "oxycodone", "oxycontin", "percocet", "hydrocodone", "vicodin",
            "tramadol", "codeine", "methadone", "buprenorphine", "suboxone",
            "acetaminophen", "tylenol", "ibuprofen", "advil", "motrin",
            "naproxen", "aleve", "ketorolac", "toradol",
            "gabapentin", "neurontin", "pregabalin", "lyrica",
            "lidocaine", "ketamine"
        ]
        
        # Other medications
        other_medications = [
            "insulin", "metformin", "glucophage", "glipizide", "glyburide",
            "levothyroxine", "synthroid", "prednisone", "methylprednisolone",
            "dexamethasone", "hydrocortisone",
            "omeprazole", "prilosec", "pantoprazole", "protonix", "famotidine",
            "ondansetron", "zofran", "promethazine", "phenergan", "metoclopramide",
            "lorazepam", "ativan", "diazepam", "valium", "midazolam", "versed",
            "haloperidol", "haldol", "quetiapine", "seroquel", "olanzapine",
            "propofol", "precedex", "dexmedetomidine",
            "albuterol", "ipratropium", "budesonide", "fluticasone",
            "oxygen", "supplemental oxygen", "bipap", "cpap"
        ]
        
        # Procedures
        procedures = [
            "surgery", "operation", "procedure", "intervention",
            "intubation", "extubation", "mechanical ventilation", "ventilator",
            "tracheostomy", "trach", "bronchoscopy",
            "central line", "central venous catheter", "cvc", "picc", "picc line",
            "arterial line", "a-line", "foley", "foley catheter",
            "dialysis", "hemodialysis", "hd", "peritoneal dialysis", "pd", "crrt",
            "blood transfusion", "transfusion", "prbc", "ffp", "platelet transfusion",
            "chest tube", "thoracentesis", "paracentesis",
            "cardiac catheterization", "cath", "pci", "stent",
            "cabg", "bypass surgery", "valve replacement",
            "endoscopy", "egd", "colonoscopy", "ercp",
            "lumbar puncture", "lp", "spinal tap",
            "ct guided biopsy", "biopsy", "aspiration"
        ]
        
        # Combine all treatments
        all_treatments = cv_medications + antibiotics + pain_medications + other_medications + procedures
        
        for term in all_treatments:
            rules.append(TargetRule(literal=term, category="TREATMENT"))
        
        # ==================== TESTS ====================
        # Lab tests
        lab_tests = [
            "cbc", "complete blood count", "hemoglobin", "hgb", "hematocrit", "hct",
            "wbc", "white blood cell", "platelet", "plt",
            "bmp", "basic metabolic panel", "cmp", "comprehensive metabolic panel",
            "sodium", "potassium", "chloride", "bicarbonate", "co2",
            "bun", "creatinine", "cr", "gfr", "egfr",
            "glucose", "blood sugar", "a1c", "hemoglobin a1c",
            "lfts", "liver function tests", "ast", "alt", "alkaline phosphatase",
            "bilirubin", "albumin", "total protein", "inr", "pt", "ptt",
            "troponin", "trop", "bnp", "pro-bnp", "ck", "ck-mb",
            "lactate", "lactic acid", "blood gas", "abg", "vbg",
            "lipase", "amylase", "tsh", "t4", "free t4",
            "blood culture", "urine culture", "sputum culture",
            "urinalysis", "ua", "urine analysis",
            "procalcitonin", "crp", "c-reactive protein", "esr", "sed rate",
            "d-dimer", "fibrinogen", "ldh",
            "magnesium", "phosphorus", "calcium", "ionized calcium"
        ]
        
        # Imaging
        imaging_tests = [
            "x-ray", "xray", "cxr", "chest x-ray", "chest xray",
            "ct", "ct scan", "cat scan", "ct head", "ct chest", "ct abdomen",
            "cta", "ct angiography", "ct angiogram",
            "mri", "mra", "mr angiography",
            "ultrasound", "us", "echocardiogram", "echo", "tte", "tee",
            "ekg", "ecg", "electrocardiogram", "telemetry",
            "eeg", "electroencephalogram",
            "pet scan", "pet-ct", "nuclear medicine scan",
            "doppler", "venous doppler", "arterial doppler",
            "angiogram", "angiography", "cardiac cath"
        ]
        
        # Combine all tests
        all_tests = lab_tests + imaging_tests
        
        for term in all_tests:
            rules.append(TargetRule(literal=term, category="TEST"))
        
        target_matcher.add(rules)
        logger.info(f"Added {len(rules)} target rules")
    
    def _setup_umls_mapper(self):
        """Initialize QuickUMLS for concept mapping."""
        try:
            logger.info(f"Loading QuickUMLS from {self.quickumls_path}...")
            self.quickumls = QuickUMLS(
                self.quickumls_path,
                overlapping_criteria='score',
                similarity_name='jaccard',
                threshold=0.7
            )
            logger.info("QuickUMLS loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load QuickUMLS: {e}")
            self.quickumls = None
    
    def _map_to_umls(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Map text to UMLS concept.
        
        Args:
            text: Entity text to map
            
        Returns:
            Tuple of (CUI, preferred name, similarity score)
        """
        if not self.quickumls:
            return None, None, 0.0
        
        try:
            matches = self.quickumls.match(text, best_match=True, ignore_syntax=False)
            if matches and len(matches) > 0:
                best_match = matches[0][0]  # First match, first result
                return best_match['cui'], best_match['term'], best_match['similarity']
        except Exception as e:
            logger.debug(f"UMLS mapping failed for '{text}': {e}")
        
        return None, None, 0.0
    
    def process_note(self, row_id: str, subject_id: str, hadm_id: str, text: str) -> ProcessedNote:
        """Process a single clinical note through the complete pipeline."""
        # 1. Clean text
        cleaned_text = self.text_cleaner.clean(text, self.expand_abbreviations)
        
        # 2. Process with medspaCy (includes sentence segmentation, NER, negation detection)
        doc = self.nlp(cleaned_text)
        
        # 3. Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # 4. Extract entities with context modifiers
        entities = []
        for ent in doc.ents:
            # Map to UMLS if available
            umls_cui, umls_name, umls_confidence = None, None, 0.0
            if self.quickumls:
                umls_cui, umls_name, umls_confidence = self._map_to_umls(ent.text)
            
            entity = ClinicalEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                is_negated=getattr(ent._, 'is_negated', False),
                is_hypothetical=getattr(ent._, 'is_hypothetical', False),
                is_historical=getattr(ent._, 'is_historical', False),
                is_family=getattr(ent._, 'is_family', False),
                is_uncertain=getattr(ent._, 'is_uncertain', False),
                umls_cui=umls_cui,
                umls_name=umls_name,
                confidence=umls_confidence
            )
            
            # Find containing sentence
            for sent in doc.sents:
                if sent.start_char <= entity.start < sent.end_char:
                    entity.sentence = sent.text.strip()
                    break
            
            entities.append(entity)
        
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
        """Process all discharge summaries from MIMIC-III."""
        logger.info("Fetching discharge summaries from database...")
        
        conn = sqlite3.connect(self.db_path)
        
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
                        'is_uncertain': entity.is_uncertain,
                        'umls_cui': entity.umls_cui,
                        'umls_name': entity.umls_name,
                        'sentence': entity.sentence,
                        'start_char': entity.start,
                        'end_char': entity.end
                    }
                    
                    # Create modified entity name
                    if entity.is_negated:
                        result['modified_entity'] = f"Absent_{entity.text.replace(' ', '_')}"
                    elif entity.is_hypothetical:
                        result['modified_entity'] = f"Possible_{entity.text.replace(' ', '_')}"
                    elif entity.is_historical:
                        result['modified_entity'] = f"History_{entity.text.replace(' ', '_')}"
                    elif entity.is_family:
                        result['modified_entity'] = f"Family_{entity.text.replace(' ', '_')}"
                    else:
                        result['modified_entity'] = entity.text.replace(' ', '_')
                    
                    all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing note {row['ROW_ID']}: {e}")
                continue
        
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        
        if len(df) == 0:
            print("No entities extracted")
            return
        
        print(f"Total entities extracted: {len(df)}")
        print(f"Unique notes processed: {df['row_id'].nunique()}")
        print(f"Unique patients: {df['subject_id'].nunique()}")
        
        print("\nEntities by category:")
        for label, count in df['entity_label'].value_counts().items():
            print(f"  {label}: {count}")
        
        print("\nContext modifiers:")
        print(f"  Negated: {df['is_negated'].sum()}")
        print(f"  Hypothetical: {df['is_hypothetical'].sum()}")
        print(f"  Historical: {df['is_historical'].sum()}")
        print(f"  Family history: {df['is_family'].sum()}")
        
        print("\nTop 20 entities:")
        for entity, count in df['entity_text'].str.lower().value_counts().head(20).items():
            print(f"  {entity}: {count}")
        
        # UMLS mapping statistics
        if 'umls_cui' in df.columns:
            mapped = df['umls_cui'].notna().sum()
            print(f"\nUMLS Mapping:")
            print(f"  Entities with UMLS CUI: {mapped} ({100*mapped/len(df):.1f}%)")
            if mapped > 0:
                print(f"  Unique CUIs: {df['umls_cui'].nunique()}")


def create_feature_vectors(
    entities_df: pd.DataFrame,
    include_negated: bool = True,
    min_count: int = 10
) -> pd.DataFrame:
    """
    Create feature vectors from extracted entities.
    
    Args:
        entities_df: DataFrame with extracted entities
        include_negated: Whether to include negated entities as separate features
        min_count: Minimum occurrences for a feature to be included
    
    Returns:
        DataFrame with one row per admission
    """
    if include_negated:
        feature_col = 'modified_entity'
    else:
        entities_df = entities_df[~entities_df['is_negated']]
        feature_col = 'entity_text'
    
    # Normalize feature names
    entities_df['feature_name'] = entities_df[feature_col].str.lower().str.replace(' ', '_')
    
    # Filter by minimum count
    feature_counts = entities_df['feature_name'].value_counts()
    valid_features = feature_counts[feature_counts >= min_count].index
    entities_df = entities_df[entities_df['feature_name'].isin(valid_features)]
    
    # Pivot to feature matrix
    feature_matrix = entities_df.pivot_table(
        index='hadm_id',
        columns='feature_name',
        values='row_id',
        aggfunc='count',
        fill_value=0
    )
    
    return feature_matrix


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical NLP Pipeline for MIMIC-III')
    parser.add_argument('--db-path', type=str, 
                        default='/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db')
    parser.add_argument('--output', type=str, default='outputs/clinical_entities.csv')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--expand-abbrev', action='store_true')
    parser.add_argument('--use-umls', action='store_true', help='Enable UMLS concept mapping')
    parser.add_argument('--quickumls-path', type=str, default=None,
                        help='Path to QuickUMLS installation')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pipeline = ClinicalNLPPipeline(
        db_path=args.db_path,
        expand_abbreviations=args.expand_abbrev,
        use_umls=args.use_umls,
        quickumls_path=args.quickumls_path
    )
    
    results = pipeline.process_discharge_summaries(
        limit=args.limit,
        output_path=args.output
    )
    
    if len(results) > 0:
        features = create_feature_vectors(results)
        features_path = args.output.replace('.csv', '_features.csv')
        features.to_csv(features_path)
        print(f"\nFeature matrix shape: {features.shape}")
        print(f"Features saved to: {features_path}")


if __name__ == '__main__':
    main()

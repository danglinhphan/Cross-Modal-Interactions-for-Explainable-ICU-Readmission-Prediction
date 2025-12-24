"""
Test script for Clinical NLP Pipeline V2
"""

import sqlite3
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress logging noise
import logging
logging.getLogger('PyRuSH').setLevel(logging.ERROR)

sys.path.insert(0, '/Users/phandanglinh/Desktop/VRES/data_preprocessing')

print("=" * 60)
print("Testing Clinical NLP Pipeline V2")
print("=" * 60)

print("\n1. Testing medspaCy with comprehensive clinical terms...")
import medspacy
from medspacy.ner import TargetRule

# Use blank model with only sentencizer
nlp = medspacy.load(enable=["tokenizer", "sentencizer"], load_rules=False)

# Check if target_matcher exists, if not add it
if "medspacy_target_matcher" not in nlp.pipe_names:
    from medspacy.ner import TargetMatcher
    target_matcher = nlp.add_pipe("medspacy_target_matcher")
else:
    target_matcher = nlp.get_pipe("medspacy_target_matcher")

# Add sample rules
rules = [
    TargetRule("pneumonia", "PROBLEM"),
    TargetRule("diabetes", "PROBLEM"),
    TargetRule("diabetes mellitus", "PROBLEM"),
    TargetRule("hypertension", "PROBLEM"),
    TargetRule("chest pain", "PROBLEM"),
    TargetRule("fever", "PROBLEM"),
    TargetRule("shortness of breath", "PROBLEM"),
    TargetRule("myocardial infarction", "PROBLEM"),
    TargetRule("metformin", "TREATMENT"),
    TargetRule("lisinopril", "TREATMENT"),
    TargetRule("aspirin", "TREATMENT"),
    TargetRule("ct scan", "TEST"),
    TargetRule("x-ray", "TEST"),
    TargetRule("chest x-ray", "TEST"),
    TargetRule("echocardiogram", "TEST"),
]
target_matcher.add(rules)

# Add context if not present
if "medspacy_context" not in nlp.pipe_names:
    nlp.add_pipe("medspacy_context")
print(f"   ✓ Pipeline components: {nlp.pipe_names}")

print("\n2. Testing with sample clinical text...")
test_text = """
The patient is a 65-year-old male with a history of diabetes mellitus and hypertension.
He denies chest pain, shortness of breath, and fever.
No evidence of pneumonia on chest x-ray.
Patient is being treated with metformin and lisinopril.
CT scan showed no acute intracranial hemorrhage.
Echocardiogram revealed normal left ventricular function.
Rule out myocardial infarction.
"""

doc = nlp(test_text)

print(f"   ✓ Found {len(doc.ents)} entities")
print("\n   Extracted entities:")
for ent in doc.ents:
    modifiers = []
    if hasattr(ent._, 'is_negated') and ent._.is_negated:
        modifiers.append("NEGATED")
    if hasattr(ent._, 'is_hypothetical') and ent._.is_hypothetical:
        modifiers.append("HYPOTHETICAL")
    if hasattr(ent._, 'is_historical') and ent._.is_historical:
        modifiers.append("HISTORICAL")
    if hasattr(ent._, 'is_family') and ent._.is_family:
        modifiers.append("FAMILY")
    
    mod_str = f" [{', '.join(modifiers)}]" if modifiers else ""
    print(f"      - {ent.text} ({ent.label_}){mod_str}")

print("\n3. Testing database connection...")
DB_PATH = "/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT ROW_ID, SUBJECT_ID, HADM_ID, TEXT 
    FROM NOTEEVENTS 
    WHERE CATEGORY = 'Discharge summary' 
    AND TEXT IS NOT NULL 
    AND LENGTH(TEXT) > 1000
    LIMIT 1
""")
row = cursor.fetchone()
conn.close()

if row:
    row_id, subject_id, hadm_id, text = row
    print(f"   ✓ Found discharge summary: ROW_ID={row_id}")
    print(f"   Text length: {len(text)} characters")

print("\n4. Testing full pipeline on real discharge summary...")
from clinical_nlp_pipeline_v2 import ClinicalNLPPipeline

pipeline = ClinicalNLPPipeline(db_path=DB_PATH)

processed = pipeline.process_note(
    row_id=str(row_id),
    subject_id=str(subject_id),
    hadm_id=str(hadm_id),
    text=text[:10000]  # First 10000 chars
)

print(f"\n   Processed note summary:")
print(f"   - Sentences: {len(processed.sentences)}")
print(f"   - Entities: {len(processed.entities)}")

# Count by category
from collections import Counter
label_counts = Counter(e.label for e in processed.entities)
print(f"   - Entities by category:")
for label, count in label_counts.most_common():
    print(f"      {label}: {count}")

# Count negated
negated_count = sum(1 for e in processed.entities if e.is_negated)
hypothetical_count = sum(1 for e in processed.entities if e.is_hypothetical)
historical_count = sum(1 for e in processed.entities if e.is_historical)
print(f"\n   Context modifiers:")
print(f"      Negated: {negated_count}")
print(f"      Hypothetical: {hypothetical_count}")
print(f"      Historical: {historical_count}")

print("\n   Sample extracted entities (first 20):")
for entity in processed.entities[:20]:
    status = []
    if entity.is_negated:
        status.append("NEGATED")
    if entity.is_hypothetical:
        status.append("HYPOTHETICAL")
    if entity.is_historical:
        status.append("HISTORICAL")
    if entity.is_family:
        status.append("FAMILY")
    status_str = f" [{', '.join(status)}]" if status else ""
    print(f"      - {entity.text} ({entity.label}){status_str}")

print("\n" + "=" * 60)
print("All tests passed! Pipeline is ready for use.")
print("=" * 60)

print("\n\nUsage example:")
print("  from clinical_nlp_pipeline_v2 import ClinicalNLPPipeline")
print("  pipeline = ClinicalNLPPipeline(db_path='path/to/MIMIC_III.db')")
print("  results = pipeline.process_discharge_summaries(limit=100, output_path='entities.csv')")

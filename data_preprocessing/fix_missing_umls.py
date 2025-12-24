#!/usr/bin/env python3
"""
Fix missing UMLS CUI mappings in clinical entities.

Two strategies:
1. Propagate CUIs from same entity_text that has been mapped elsewhere
2. Use abbreviation dictionary for common clinical abbreviations

Usage:
    python fix_missing_umls.py --input outputs/cohort_clinical_entities_umls.csv --output outputs/cohort_clinical_entities_umls_fixed.csv
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clinical abbreviation dictionary mapping to UMLS CUI
# Format: abbreviation -> (CUI, UMLS Name, Category)
CLINICAL_ABBREVIATIONS = {
    # Lab Tests
    'plt': ('C0032181', 'Platelet Count', 'TEST'),
    'plts': ('C0032181', 'Platelet Count', 'TEST'),
    'platelets': ('C0032181', 'Platelet Count', 'TEST'),
    'pt': ('C0033707', 'Prothrombin Time', 'TEST'),
    'ptt': ('C0030605', 'Partial Thromboplastin Time', 'TEST'),
    'inr': ('C0525032', 'International Normalized Ratio', 'TEST'),
    'hct': ('C0018935', 'Hematocrit', 'TEST'),
    'hgb': ('C0019029', 'Hemoglobin', 'TEST'),
    'wbc': ('C0023508', 'White Blood Cell Count', 'TEST'),
    'rbc': ('C0014772', 'Red Blood Cell Count', 'TEST'),
    'bun': ('C0005845', 'Blood Urea Nitrogen', 'TEST'),
    'cr': ('C0010294', 'Creatinine', 'TEST'),
    'creat': ('C0010294', 'Creatinine', 'TEST'),
    'lfts': ('C0023901', 'Liver Function Tests', 'TEST'),
    'lft': ('C0023901', 'Liver Function Tests', 'TEST'),
    'ast': ('C0004002', 'Aspartate Aminotransferase', 'TEST'),
    'alt': ('C0001899', 'Alanine Aminotransferase', 'TEST'),
    'alp': ('C0002059', 'Alkaline Phosphatase', 'TEST'),
    'bili': ('C0005437', 'Bilirubin', 'TEST'),
    'tbili': ('C0005437', 'Total Bilirubin', 'TEST'),
    'abg': ('C0150411', 'Arterial Blood Gas', 'TEST'),
    'vbg': ('C0428205', 'Venous Blood Gas', 'TEST'),
    'cbc': ('C0009555', 'Complete Blood Count', 'TEST'),
    'bmp': ('C0428236', 'Basic Metabolic Panel', 'TEST'),
    'cmp': ('C0428235', 'Comprehensive Metabolic Panel', 'TEST'),
    'trop': ('C0041199', 'Troponin', 'TEST'),
    'troponin': ('C0041199', 'Troponin', 'TEST'),
    'bnp': ('C0054015', 'Brain Natriuretic Peptide', 'TEST'),
    'lactate': ('C0522568', 'Lactate Level', 'TEST'),
    'hba1c': ('C0474680', 'Hemoglobin A1c', 'TEST'),
    'a1c': ('C0474680', 'Hemoglobin A1c', 'TEST'),
    'tsh': ('C0040160', 'Thyroid Stimulating Hormone', 'TEST'),
    'ft4': ('C0428420', 'Free T4', 'TEST'),
    't4': ('C0040165', 'Thyroxine', 'TEST'),
    'esr': ('C0014772', 'Erythrocyte Sedimentation Rate', 'TEST'),
    'crp': ('C0006560', 'C-Reactive Protein', 'TEST'),
    'ua': ('C0042014', 'Urinalysis', 'TEST'),
    'ucx': ('C0430404', 'Urine Culture', 'TEST'),
    'bcx': ('C0200949', 'Blood Culture', 'TEST'),
    
    # Imaging/Procedures
    'tte': ('C0013516', 'Transthoracic Echocardiogram', 'TEST'),
    'tee': ('C0206054', 'Transesophageal Echocardiogram', 'TEST'),
    'echo': ('C0013516', 'Echocardiogram', 'TEST'),
    'ekg': ('C0013798', 'Electrocardiogram', 'TEST'),
    'ecg': ('C0013798', 'Electrocardiogram', 'TEST'),
    'cxr': ('C0039985', 'Chest X-Ray', 'TEST'),
    'ct': ('C0040405', 'CT Scan', 'TEST'),
    'mri': ('C0024485', 'MRI', 'TEST'),
    'us': ('C0041618', 'Ultrasound', 'TEST'),
    'kub': ('C0412617', 'Kidney Ureter Bladder X-Ray', 'TEST'),
    'lp': ('C0037943', 'Lumbar Puncture', 'TEST'),
    'eeg': ('C0013819', 'Electroencephalogram', 'TEST'),
    'emg': ('C0013839', 'Electromyography', 'TEST'),
    'pft': ('C0024119', 'Pulmonary Function Test', 'TEST'),
    'pfts': ('C0024119', 'Pulmonary Function Tests', 'TEST'),
    
    # Devices/Equipment
    'cath': ('C0007430', 'Catheter', 'TREATMENT'),
    'foley': ('C0179181', 'Foley Catheter', 'TREATMENT'),
    'picc': ('C0152023', 'PICC Line', 'TREATMENT'),
    'trach': ('C0040590', 'Tracheostomy', 'TREATMENT'),
    'vent': ('C0042497', 'Ventilator', 'TREATMENT'),
    'ventilator': ('C0042497', 'Mechanical Ventilator', 'TREATMENT'),
    'bipap': ('C0376472', 'BiPAP', 'TREATMENT'),
    'cpap': ('C0087027', 'CPAP', 'TREATMENT'),
    'ng': ('C0085678', 'Nasogastric Tube', 'TREATMENT'),
    'ngt': ('C0085678', 'Nasogastric Tube', 'TREATMENT'),
    'peg': ('C0176620', 'PEG Tube', 'TREATMENT'),
    'ivf': ('C0015222', 'IV Fluids', 'TREATMENT'),
    'tpn': ('C0030548', 'Total Parenteral Nutrition', 'TREATMENT'),
    'prbc': ('C0032852', 'Packed Red Blood Cells', 'TREATMENT'),
    'prbcs': ('C0032852', 'Packed Red Blood Cells', 'TREATMENT'),
    'ffp': ('C0016792', 'Fresh Frozen Plasma', 'TREATMENT'),
    'cryo': ('C0010408', 'Cryoprecipitate', 'TREATMENT'),
    
    # Medications (common abbreviations)
    'asa': ('C0004057', 'Aspirin', 'TREATMENT'),
    'acei': ('C0003015', 'ACE Inhibitor', 'TREATMENT'),
    'arb': ('C0815017', 'Angiotensin Receptor Blocker', 'TREATMENT'),
    'bb': ('C0001645', 'Beta Blocker', 'TREATMENT'),
    'ccb': ('C0006684', 'Calcium Channel Blocker', 'TREATMENT'),
    'ppi': ('C0358514', 'Proton Pump Inhibitor', 'TREATMENT'),
    'ssri': ('C0360105', 'SSRI', 'TREATMENT'),
    'nsaid': ('C0003211', 'NSAID', 'TREATMENT'),
    'nsaids': ('C0003211', 'NSAIDs', 'TREATMENT'),
    'abx': ('C0003232', 'Antibiotics', 'TREATMENT'),
    'vanc': ('C0042313', 'Vancomycin', 'TREATMENT'),
    'vanco': ('C0042313', 'Vancomycin', 'TREATMENT'),
    'zosyn': ('C0724639', 'Piperacillin-Tazobactam', 'TREATMENT'),
    'pip-tazo': ('C0724639', 'Piperacillin-Tazobactam', 'TREATMENT'),
    'meropenem': ('C0066005', 'Meropenem', 'TREATMENT'),
    'flagyl': ('C0025872', 'Metronidazole', 'TREATMENT'),
    'levo': ('C0282386', 'Levofloxacin', 'TREATMENT'),
    'cipro': ('C0008809', 'Ciprofloxacin', 'TREATMENT'),
    'augmentin': ('C0054066', 'Amoxicillin-Clavulanate', 'TREATMENT'),
    'amox': ('C0002645', 'Amoxicillin', 'TREATMENT'),
    'azithro': ('C0052796', 'Azithromycin', 'TREATMENT'),
    'heparin': ('C0019134', 'Heparin', 'TREATMENT'),
    'lovenox': ('C0206460', 'Enoxaparin', 'TREATMENT'),
    'coumadin': ('C0699129', 'Warfarin', 'TREATMENT'),
    'warfarin': ('C0043031', 'Warfarin', 'TREATMENT'),
    'lasix': ('C0016860', 'Furosemide', 'TREATMENT'),
    'bumex': ('C0700459', 'Bumetanide', 'TREATMENT'),
    'digoxin': ('C0012265', 'Digoxin', 'TREATMENT'),
    'dig': ('C0012265', 'Digoxin', 'TREATMENT'),
    'metoprolol': ('C0025859', 'Metoprolol', 'TREATMENT'),
    'lopressor': ('C0591228', 'Metoprolol', 'TREATMENT'),
    'lisinopril': ('C0065374', 'Lisinopril', 'TREATMENT'),
    'amlodipine': ('C0051696', 'Amlodipine', 'TREATMENT'),
    'norvasc': ('C0591228', 'Amlodipine', 'TREATMENT'),
    'diltiazem': ('C0012373', 'Diltiazem', 'TREATMENT'),
    'cardizem': ('C0699858', 'Diltiazem', 'TREATMENT'),
    'amiodarone': ('C0002598', 'Amiodarone', 'TREATMENT'),
    'prednisone': ('C0032952', 'Prednisone', 'TREATMENT'),
    'pred': ('C0032952', 'Prednisone', 'TREATMENT'),
    'solumedrol': ('C0086443', 'Methylprednisolone', 'TREATMENT'),
    'dexamethasone': ('C0011777', 'Dexamethasone', 'TREATMENT'),
    'decadron': ('C0699921', 'Dexamethasone', 'TREATMENT'),
    'insulin': ('C0021641', 'Insulin', 'TREATMENT'),
    'metformin': ('C0025598', 'Metformin', 'TREATMENT'),
    'glucophage': ('C0591573', 'Metformin', 'TREATMENT'),
    'morphine': ('C0026549', 'Morphine', 'TREATMENT'),
    'dilaudid': ('C0699862', 'Hydromorphone', 'TREATMENT'),
    'fentanyl': ('C0015846', 'Fentanyl', 'TREATMENT'),
    'oxycodone': ('C0030049', 'Oxycodone', 'TREATMENT'),
    'percocet': ('C0086787', 'Oxycodone-Acetaminophen', 'TREATMENT'),
    'tylenol': ('C0699142', 'Acetaminophen', 'TREATMENT'),
    'apap': ('C0000970', 'Acetaminophen', 'TREATMENT'),
    'ibuprofen': ('C0020740', 'Ibuprofen', 'TREATMENT'),
    'advil': ('C0593507', 'Ibuprofen', 'TREATMENT'),
    'motrin': ('C0699203', 'Ibuprofen', 'TREATMENT'),
    'albuterol': ('C0001927', 'Albuterol', 'TREATMENT'),
    'nebs': ('C0027552', 'Nebulizer Treatment', 'TREATMENT'),
    'duonebs': ('C1170161', 'Albuterol-Ipratropium', 'TREATMENT'),
    'fluticasone': ('C0082607', 'Fluticasone', 'TREATMENT'),
    'flovent': ('C0939237', 'Fluticasone', 'TREATMENT'),
    'singulair': ('C0698988', 'Montelukast', 'TREATMENT'),
    'ativan': ('C0699153', 'Lorazepam', 'TREATMENT'),
    'lorazepam': ('C0024002', 'Lorazepam', 'TREATMENT'),
    'valium': ('C0699136', 'Diazepam', 'TREATMENT'),
    'diazepam': ('C0012010', 'Diazepam', 'TREATMENT'),
    'haldol': ('C0699129', 'Haloperidol', 'TREATMENT'),
    'haloperidol': ('C0018546', 'Haloperidol', 'TREATMENT'),
    'seroquel': ('C0698996', 'Quetiapine', 'TREATMENT'),
    'risperdal': ('C0698851', 'Risperidone', 'TREATMENT'),
    'zyprexa': ('C0699219', 'Olanzapine', 'TREATMENT'),
    'zoloft': ('C0699190', 'Sertraline', 'TREATMENT'),
    'lexapro': ('C1099456', 'Escitalopram', 'TREATMENT'),
    'prozac': ('C0699828', 'Fluoxetine', 'TREATMENT'),
    'effexor': ('C0700430', 'Venlafaxine', 'TREATMENT'),
    'wellbutrin': ('C0699877', 'Bupropion', 'TREATMENT'),
    'trazodone': ('C0040617', 'Trazodone', 'TREATMENT'),
    'ambien': ('C0700017', 'Zolpidem', 'TREATMENT'),
    'benadryl': ('C0699142', 'Diphenhydramine', 'TREATMENT'),
    'zofran': ('C0700017', 'Ondansetron', 'TREATMENT'),
    'reglan': ('C0699185', 'Metoclopramide', 'TREATMENT'),
    'phenergan': ('C0699139', 'Promethazine', 'TREATMENT'),
    'colace': ('C0699101', 'Docusate', 'TREATMENT'),
    'miralax': ('C0720429', 'Polyethylene Glycol', 'TREATMENT'),
    'senna': ('C0036626', 'Senna', 'TREATMENT'),
    'protonix': ('C0698948', 'Pantoprazole', 'TREATMENT'),
    'nexium': ('C1170353', 'Esomeprazole', 'TREATMENT'),
    'prilosec': ('C0699206', 'Omeprazole', 'TREATMENT'),
    'pepcid': ('C0699114', 'Famotidine', 'TREATMENT'),
    'zantac': ('C0699184', 'Ranitidine', 'TREATMENT'),
    'lipitor': ('C0593906', 'Atorvastatin', 'TREATMENT'),
    'atorvastatin': ('C0286651', 'Atorvastatin', 'TREATMENT'),
    'simvastatin': ('C0074554', 'Simvastatin', 'TREATMENT'),
    'zocor': ('C0699231', 'Simvastatin', 'TREATMENT'),
    'plavix': ('C0633084', 'Clopidogrel', 'TREATMENT'),
    'clopidogrel': ('C0070166', 'Clopidogrel', 'TREATMENT'),
    
    # Symptoms/Problems (common)
    'sob': ('C0013404', 'Shortness of Breath', 'PROBLEM'),
    'dyspnea': ('C0013404', 'Dyspnea', 'PROBLEM'),
    'cp': ('C0008031', 'Chest Pain', 'PROBLEM'),
    'n/v': ('C0027498', 'Nausea and Vomiting', 'PROBLEM'),
    'nv': ('C0027498', 'Nausea and Vomiting', 'PROBLEM'),
    'ha': ('C0018681', 'Headache', 'PROBLEM'),
    'loc': ('C0041657', 'Loss of Consciousness', 'PROBLEM'),
    'ams': ('C0233794', 'Altered Mental Status', 'PROBLEM'),
    'doi': ('C0242429', 'Delirium', 'PROBLEM'),
    'uti': ('C0042029', 'Urinary Tract Infection', 'PROBLEM'),
    'pna': ('C0032285', 'Pneumonia', 'PROBLEM'),
    'chf': ('C0018802', 'Congestive Heart Failure', 'PROBLEM'),
    'hf': ('C0018801', 'Heart Failure', 'PROBLEM'),
    'cad': ('C0010054', 'Coronary Artery Disease', 'PROBLEM'),
    'mi': ('C0027051', 'Myocardial Infarction', 'PROBLEM'),
    'nstemi': ('C0262681', 'NSTEMI', 'PROBLEM'),
    'stemi': ('C0262681', 'STEMI', 'PROBLEM'),
    'afib': ('C0004238', 'Atrial Fibrillation', 'PROBLEM'),
    'a-fib': ('C0004238', 'Atrial Fibrillation', 'PROBLEM'),
    'af': ('C0004238', 'Atrial Fibrillation', 'PROBLEM'),
    'rvr': ('C0521650', 'Rapid Ventricular Response', 'PROBLEM'),
    'vt': ('C0042514', 'Ventricular Tachycardia', 'PROBLEM'),
    'vf': ('C0042510', 'Ventricular Fibrillation', 'PROBLEM'),
    'svt': ('C0039240', 'Supraventricular Tachycardia', 'PROBLEM'),
    'htn': ('C0020538', 'Hypertension', 'PROBLEM'),
    'dm': ('C0011849', 'Diabetes Mellitus', 'PROBLEM'),
    'dm2': ('C0011860', 'Type 2 Diabetes', 'PROBLEM'),
    'dm1': ('C0011854', 'Type 1 Diabetes', 'PROBLEM'),
    'iddm': ('C0011854', 'Insulin-Dependent Diabetes', 'PROBLEM'),
    'niddm': ('C0011860', 'Non-Insulin-Dependent Diabetes', 'PROBLEM'),
    'dka': ('C0011880', 'Diabetic Ketoacidosis', 'PROBLEM'),
    'hhs': ('C0020626', 'Hyperosmolar Hyperglycemic State', 'PROBLEM'),
    'copd': ('C0024117', 'COPD', 'PROBLEM'),
    'ckd': ('C1561643', 'Chronic Kidney Disease', 'PROBLEM'),
    'esrd': ('C0022661', 'End Stage Renal Disease', 'PROBLEM'),
    'aki': ('C2609414', 'Acute Kidney Injury', 'PROBLEM'),
    'arf': ('C0022660', 'Acute Renal Failure', 'PROBLEM'),
    'cva': ('C0038454', 'Stroke', 'PROBLEM'),
    'tia': ('C0007787', 'Transient Ischemic Attack', 'PROBLEM'),
    'dvt': ('C0149871', 'Deep Vein Thrombosis', 'PROBLEM'),
    'pe': ('C0034065', 'Pulmonary Embolism', 'PROBLEM'),
    'ards': ('C0035222', 'ARDS', 'PROBLEM'),
    'sepsis': ('C0243026', 'Sepsis', 'PROBLEM'),
    'sirs': ('C0242966', 'SIRS', 'PROBLEM'),
    'gi': ('C0017181', 'Gastrointestinal', 'PROBLEM'),
    'gib': ('C0017181', 'GI Bleeding', 'PROBLEM'),
    'ugib': ('C0041909', 'Upper GI Bleed', 'PROBLEM'),
    'lgib': ('C0023533', 'Lower GI Bleed', 'PROBLEM'),
    'brbpr': ('C0018932', 'Hematochezia', 'PROBLEM'),
    'melena': ('C0025222', 'Melena', 'PROBLEM'),
    'ascites': ('C0003962', 'Ascites', 'PROBLEM'),
    'sbo': ('C0037384', 'Small Bowel Obstruction', 'PROBLEM'),
    'lbo': ('C0023532', 'Large Bowel Obstruction', 'PROBLEM'),
    'etoh': ('C0001962', 'Alcohol', 'PROBLEM'),
    'etoh abuse': ('C0001973', 'Alcohol Abuse', 'PROBLEM'),
    'ivda': ('C0242566', 'IV Drug Abuse', 'PROBLEM'),
    'c diff': ('C0079134', 'Clostridioides difficile', 'PROBLEM'),
    'cdiff': ('C0079134', 'Clostridioides difficile', 'PROBLEM'),
    'mrsa': ('C0343401', 'MRSA', 'PROBLEM'),
    'vre': ('C0085562', 'VRE', 'PROBLEM'),
    'hiv': ('C0019693', 'HIV', 'PROBLEM'),
    'aids': ('C0001175', 'AIDS', 'PROBLEM'),
    'hep c': ('C0019196', 'Hepatitis C', 'PROBLEM'),
    'hcv': ('C0019196', 'Hepatitis C', 'PROBLEM'),
    'hep b': ('C0019163', 'Hepatitis B', 'PROBLEM'),
    'hbv': ('C0019163', 'Hepatitis B', 'PROBLEM'),
    'cirrhosis': ('C0023890', 'Liver Cirrhosis', 'PROBLEM'),
    'encephalopathy': ('C0014038', 'Encephalopathy', 'PROBLEM'),
    'sz': ('C0036572', 'Seizure', 'PROBLEM'),
    'seizure': ('C0036572', 'Seizure', 'PROBLEM'),
    'sz disorder': ('C0014544', 'Epilepsy', 'PROBLEM'),
    
    # Common symptoms
    'cough': ('C0010200', 'Cough', 'PROBLEM'),
    'fever': ('C0015967', 'Fever', 'PROBLEM'),
    'chills': ('C0085593', 'Chills', 'PROBLEM'),
    'fatigue': ('C0015672', 'Fatigue', 'PROBLEM'),
    'weakness': ('C0004093', 'Asthenia', 'PROBLEM'),
    'dizziness': ('C0012833', 'Dizziness', 'PROBLEM'),
    'syncope': ('C0039070', 'Syncope', 'PROBLEM'),
    'nausea': ('C0027497', 'Nausea', 'PROBLEM'),
    'vomiting': ('C0042963', 'Vomiting', 'PROBLEM'),
    'diarrhea': ('C0011991', 'Diarrhea', 'PROBLEM'),
    'constipation': ('C0009806', 'Constipation', 'PROBLEM'),
    'edema': ('C0013604', 'Edema', 'PROBLEM'),
    'swelling': ('C0038999', 'Swelling', 'PROBLEM'),
    'pain': ('C0030193', 'Pain', 'PROBLEM'),
    'anxiety': ('C0003467', 'Anxiety', 'PROBLEM'),
    'depression': ('C0011570', 'Depression', 'PROBLEM'),
    'insomnia': ('C0917801', 'Insomnia', 'PROBLEM'),
    'malaise': ('C0231218', 'Malaise', 'PROBLEM'),
    'unresponsive': ('C0241526', 'Unresponsive', 'PROBLEM'),
    
    # Common substances
    'oxygen': ('C0030054', 'Oxygen', 'TREATMENT'),
    'o2': ('C0030054', 'Oxygen', 'TREATMENT'),
    'glucose': ('C0017725', 'Glucose', 'TEST'),
    'potassium': ('C0032821', 'Potassium', 'TEST'),
    'k': ('C0032821', 'Potassium', 'TEST'),
    'sodium': ('C0037473', 'Sodium', 'TEST'),
    'na': ('C0037473', 'Sodium', 'TEST'),
    'magnesium': ('C0024467', 'Magnesium', 'TEST'),
    'mg': ('C0024467', 'Magnesium', 'TEST'),
    'calcium': ('C0006675', 'Calcium', 'TEST'),
    'ca': ('C0006675', 'Calcium', 'TEST'),
    'phosphorus': ('C0031705', 'Phosphorus', 'TEST'),
    'phos': ('C0031705', 'Phosphorus', 'TEST'),
    
    # Cancer types
    'cancer': ('C0006826', 'Malignant Neoplasm', 'PROBLEM'),
    'ca': ('C0006826', 'Cancer', 'PROBLEM'),
    'lymphoma': ('C0024299', 'Lymphoma', 'PROBLEM'),
    'leukemia': ('C0023418', 'Leukemia', 'PROBLEM'),
    'mets': ('C0027627', 'Metastasis', 'PROBLEM'),
    'metastatic': ('C0027627', 'Metastatic', 'PROBLEM'),
}


def build_cui_lookup_from_data(df: pd.DataFrame) -> dict:
    """
    Build a lookup dictionary from entities that have been successfully mapped.
    Uses the most common CUI for each entity_text.
    """
    logger.info("Building CUI lookup from existing mappings...")
    
    # Filter to rows with CUI
    with_cui = df[df['umls_cui_mapped'].notna()].copy()
    
    # Normalize entity text
    with_cui['entity_lower'] = with_cui['entity_text'].str.lower().str.strip()
    
    # For each entity, find the most common CUI
    cui_lookup = {}
    grouped = with_cui.groupby('entity_lower')
    
    for entity, group in grouped:
        # Get most common CUI for this entity
        cui_counts = group.groupby(['umls_cui_mapped', 'umls_name_mapped']).size()
        if len(cui_counts) > 0:
            best_cui, best_name = cui_counts.idxmax()
            best_score = group[group['umls_cui_mapped'] == best_cui]['umls_score'].mean()
            cui_lookup[entity] = (best_cui, best_name, best_score)
    
    logger.info(f"Built lookup with {len(cui_lookup):,} unique entity mappings")
    return cui_lookup


def fix_missing_umls(df: pd.DataFrame, cui_lookup: dict) -> pd.DataFrame:
    """
    Fix missing UMLS mappings using abbreviation dictionary and data-derived lookup.
    """
    logger.info("Fixing missing UMLS mappings...")
    
    # Create copy
    df = df.copy()
    
    # Normalize entity text for lookup
    df['entity_lower'] = df['entity_text'].str.lower().str.strip()
    
    # Track fixes
    fixed_from_abbrev = 0
    fixed_from_data = 0
    
    # Find rows without CUI
    missing_mask = df['umls_cui_mapped'].isna()
    missing_indices = df[missing_mask].index
    
    logger.info(f"Processing {len(missing_indices):,} entities without CUI...")
    
    for idx in missing_indices:
        entity = df.loc[idx, 'entity_lower']
        
        # Try abbreviation dictionary first
        if entity in CLINICAL_ABBREVIATIONS:
            cui, name, _ = CLINICAL_ABBREVIATIONS[entity]
            df.loc[idx, 'umls_cui_mapped'] = cui
            df.loc[idx, 'umls_name_mapped'] = name
            df.loc[idx, 'umls_score'] = 1.0  # Manual mapping = high confidence
            fixed_from_abbrev += 1
        # Then try data-derived lookup
        elif entity in cui_lookup:
            cui, name, score = cui_lookup[entity]
            df.loc[idx, 'umls_cui_mapped'] = cui
            df.loc[idx, 'umls_name_mapped'] = name
            df.loc[idx, 'umls_score'] = score
            fixed_from_data += 1
    
    # Drop helper column
    df = df.drop(columns=['entity_lower'])
    
    logger.info(f"Fixed {fixed_from_abbrev:,} from abbreviation dictionary")
    logger.info(f"Fixed {fixed_from_data:,} from data-derived lookup")
    logger.info(f"Total fixed: {fixed_from_abbrev + fixed_from_data:,}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Fix missing UMLS CUI mappings')
    parser.add_argument('--input', required=True, help='Input CSV with UMLS mappings')
    parser.add_argument('--output', required=True, help='Output CSV with fixed mappings')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    
    initial_missing = df['umls_cui_mapped'].isna().sum()
    initial_total = len(df)
    logger.info(f"Initial: {initial_missing:,}/{initial_total:,} ({initial_missing/initial_total*100:.1f}%) missing CUI")
    
    # Build lookup from existing data
    cui_lookup = build_cui_lookup_from_data(df)
    
    # Fix missing
    df = fix_missing_umls(df, cui_lookup)
    
    # Report results
    final_missing = df['umls_cui_mapped'].isna().sum()
    logger.info(f"Final: {final_missing:,}/{initial_total:,} ({final_missing/initial_total*100:.1f}%) missing CUI")
    logger.info(f"Improvement: {initial_missing - final_missing:,} entities fixed")
    
    # Save
    logger.info(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("UMLS FIX SUMMARY")
    print("="*60)
    print(f"Total entities: {initial_total:,}")
    print(f"Before fix - Missing CUI: {initial_missing:,} ({initial_missing/initial_total*100:.1f}%)")
    print(f"After fix  - Missing CUI: {final_missing:,} ({final_missing/initial_total*100:.1f}%)")
    print(f"Entities fixed: {initial_missing - final_missing:,}")
    print(f"New success rate: {(initial_total - final_missing)/initial_total*100:.1f}%")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()

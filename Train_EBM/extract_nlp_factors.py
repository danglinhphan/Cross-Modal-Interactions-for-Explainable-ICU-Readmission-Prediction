
import pandas as pd
import numpy as np
import argparse
import re
import os
import logging
from textblob import TextBlob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Clinical Risk Keywords (regex patterns)
# These map to common ICU admission reasons or complications
RISK_KEYWORDS = {
    'nlp_agitation': r'\bagitated\b|\bagitation\b|\bcombative\b|\brestraints\b',
    'nlp_intubation': r'\bintubated\b|\bintubation\b|\bventilat',
    'nlp_suicide': r'\bsuicide\b|\boverdose\b|\bself[- ]harm\b',
    'nlp_withdrawal': r'\bwithdrawal\b|\bciwa\b|\balcohol\b|\bdt\b',
    'nlp_sepsis': r'\bsepsis\b|\bseptic\b|\binfection\b|\bbacteremia\b',
    'nlp_hypotension': r'\bhypotension\b|\bhypotensive\b|\bshock\b|\bpressors\b',
    'nlp_respiratory_failure': r'\brespiratory failure\b|\bapnea\b|\bhypoxia\b|\bcpap\b|\bbipap\b',
    'nlp_altering_mental_status': r'\baltered mental status\b|\bconfusion\b|\bdisoriented\b|\bencephalopathy\b',
    'nlp_fall': r'\bfall\b|\bfell\b|\btrauma\b',
    'nlp_pain': r'\bpain\b|\bdiscomfort\b|\banalgesics\b',
    'nlp_kidney_injury': r'\baki\b|\brenal failure\b|\bdialysis\b|\bcrrt\b',
    'nlp_bleeding': r'\bbleeding\b|\bhemorrhage\b|\bgi bleed\b|\bhematemesis\b',
    'nlp_cardiac_event': r'\bcardiac arrest\b|\bmi\b|\bstemi\b|\bheart failure\b|\bchf\b',
    'nlp_pneumonia': r'\bpneumonia\b|\baspiration\b|\bconsolidation\b',
    'nlp_anemia': r'\banemia\b|\bhgb\b|\btransfusion\b'
}

def get_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity

def count_keywords(text, pattern):
    if not isinstance(text, str):
        return 0
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw notes CSV')
    parser.add_argument('--output', required=True, help='Path to output features CSV')
    args = parser.parse_args()
    
    logger.info(f"Reading notes from {args.input}...")
    # Load notes, assume HADM_ID and DISCHARGE_SUMMARY_TEXT (or similar)
    # The file we inspected had: SUBJECT_ID,HADM_ID,...,DISCHARGE_SUMMARY_TEXT
    try:
        df = pd.read_csv(args.input, usecols=['HADM_ID', 'CLEAN_TEXT', 'Y'])
    except ValueError:
        # Fallback if 'Y' is not in this file (it might be in labels file)
        logger.warning("'Y' column not found, loading HADM_ID and text only.")
        df = pd.read_csv(args.input, usecols=['HADM_ID', 'CLEAN_TEXT'])
        
    logger.info(f"Loaded {len(df)} notes.")
    
    # Preprocessing
    df['text_clean'] = df['CLEAN_TEXT'].fillna('').astype(str).str.lower()
    
    # 1. Sentiment Analysis
    logger.info("Extracting Sentiment...")
    df['nlp_sentiment_score'] = df['text_clean'].apply(get_sentiment)
    
    # 2. Keyword Extraction
    logger.info("Extracting Risk Factors...")
    for feature_name, pattern in RISK_KEYWORDS.items():
        df[feature_name] = df['text_clean'].apply(lambda x: 1 if re.search(pattern, x) else 0)
        
    # 3. Note Length (proxy for complexity)
    df['nlp_note_length'] = df['text_clean'].apply(len)
    
    # Select columns to save
    out_cols = ['HADM_ID', 'nlp_sentiment_score', 'nlp_note_length'] + list(RISK_KEYWORDS.keys())
    out_df = df[out_cols]
    
    # Group by HADM_ID (take max for risks, mean for sentiment)
    # In case there are multiple notes per admission
    agg_funcs = {col: 'max' for col in RISK_KEYWORDS.keys()}
    agg_funcs['nlp_sentiment_score'] = 'mean'
    agg_funcs['nlp_note_length'] = 'sum'
    
    out_df = out_df.groupby('HADM_ID').agg(agg_funcs).reset_index()
    
    logger.info(f"Final shape: {out_df.shape}")
    out_df.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

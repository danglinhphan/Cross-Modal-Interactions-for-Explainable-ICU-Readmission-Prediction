"""
Process discharge summaries for the cohort defined in features_locf_v2.csv

This script extracts clinical entities from discharge summaries for patients
in the current cohort only (matching HADM_IDs from the feature file).
"""

import os
import sys
import sqlite3
import logging
import pandas as pd
from typing import List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress PyRuSH debug messages
logging.getLogger('PyRuSH').setLevel(logging.WARNING)

# Import the pipeline
from clinical_nlp_pipeline_v2 import ClinicalNLPPipeline, ClinicalEntity


def get_cohort_hadm_ids(feature_file: str) -> List[int]:
    """Load HADM_IDs from the feature file."""
    df = pd.read_csv(feature_file)
    hadm_ids = df['HADM_ID'].unique().tolist()
    logger.info(f"Loaded {len(hadm_ids)} HADM_IDs from {feature_file}")
    return hadm_ids


def process_cohort_notes(
    db_path: str,
    feature_file: str,
    output_path: str,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Process discharge summaries for cohort patients only.
    
    Args:
        db_path: Path to MIMIC-III database
        feature_file: Path to feature file with HADM_IDs
        output_path: Output path for extracted entities
        batch_size: Number of notes to process at a time
    """
    # Get HADM_IDs from cohort
    hadm_ids = get_cohort_hadm_ids(feature_file)
    
    # Initialize pipeline
    logger.info("Initializing NLP pipeline...")
    pipeline = ClinicalNLPPipeline(db_path=db_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get discharge summaries for cohort
    logger.info("Fetching discharge summaries for cohort...")
    
    all_results = []
    total_processed = 0
    
    # Process in batches to manage memory
    for batch_start in range(0, len(hadm_ids), batch_size):
        batch_hadm_ids = hadm_ids[batch_start:batch_start + batch_size]
        placeholders = ','.join([str(x) for x in batch_hadm_ids])
        
        query = f'''
        SELECT ROW_ID, SUBJECT_ID, HADM_ID, TEXT 
        FROM NOTEEVENTS 
        WHERE CATEGORY = 'Discharge summary'
        AND TEXT IS NOT NULL
        AND HADM_ID IN ({placeholders})
        '''
        
        df_notes = pd.read_sql_query(query, conn)
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: {len(df_notes)} notes (total progress: {total_processed}/{len(hadm_ids)})")
        
        for idx, row in df_notes.iterrows():
            try:
                processed = pipeline.process_note(
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
        
        total_processed += len(df_notes)
        
        # Save intermediate results every 5000 notes
        if total_processed % 5000 < batch_size:
            intermediate_df = pd.DataFrame(all_results)
            intermediate_path = output_path.replace('.csv', f'_intermediate_{total_processed}.csv')
            intermediate_df.to_csv(intermediate_path, index=False)
            logger.info(f"Saved intermediate results: {len(intermediate_df)} entities")
    
    conn.close()
    
    # Create final DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COHORT NLP EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total entities extracted: {len(results_df)}")
    print(f"Unique notes processed: {results_df['row_id'].nunique()}")
    print(f"Unique HADM_IDs: {results_df['hadm_id'].nunique()}")
    print(f"Unique patients: {results_df['subject_id'].nunique()}")
    
    print("\nEntities by category:")
    for label, count in results_df['entity_label'].value_counts().items():
        print(f"  {label}: {count}")
    
    print("\nContext modifiers:")
    print(f"  Negated: {results_df['is_negated'].sum()}")
    print(f"  Hypothetical: {results_df['is_hypothetical'].sum()}")
    print(f"  Historical: {results_df['is_historical'].sum()}")
    print(f"  Family history: {results_df['is_family'].sum()}")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process discharge summaries for cohort')
    parser.add_argument('--db-path', type=str, 
                        default='/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db')
    parser.add_argument('--feature-file', type=str,
                        default='/Users/phandanglinh/Desktop/VRES/cohort/features_locf_v2.csv')
    parser.add_argument('--output', type=str, 
                        default='/Users/phandanglinh/Desktop/VRES/outputs/cohort_clinical_entities.csv')
    parser.add_argument('--batch-size', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = process_cohort_notes(
        db_path=args.db_path,
        feature_file=args.feature_file,
        output_path=args.output,
        batch_size=args.batch_size
    )
    
    print(f"\nProcessing complete!")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()

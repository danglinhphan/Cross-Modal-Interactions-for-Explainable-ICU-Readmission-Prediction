"""
Fill missing embeddings for HADM_IDs that don't have Discharge summaries.
Uses other note types (Nursing, Radiology, etc.) to create embeddings.
"""
import pandas as pd
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import argparse


def concatenate_notes(notes_list, max_length=10000):
    """Concatenate multiple notes with separator, truncate to max_length."""
    combined = "\n\n---\n\n".join(notes_list)
    if len(combined) > max_length:
        combined = combined[:max_length]
    return combined


def main(db_path, features_csv, output_csv, model_name='all-MiniLM-L6-v2'):
    print("=== FILL MISSING EMBEDDINGS ===")
    print()
    
    # Load features
    df = pd.read_csv(features_csv)
    emb_cols = [c for c in df.columns if c.startswith('note_emb_')]
    X_emb = df[emb_cols].values
    norms = np.linalg.norm(X_emb, axis=1)
    
    zero_mask = norms == 0
    zero_count = zero_mask.sum()
    print(f"HADM_IDs with zero embeddings: {zero_count}")
    
    if zero_count == 0:
        print("No missing embeddings. Done.")
        return
    
    zero_hadms = df[zero_mask]['HADM_ID'].tolist()
    zero_indices = np.where(zero_mask)[0]
    
    # Load sentence transformer
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get notes for each HADM_ID
    print("\nExtracting notes for missing HADMs...")
    texts = []
    found_hadms = []
    not_found_hadms = []
    
    # Priority order for note types
    priority_categories = [
        'Nursing/other', 'Nursing', 'Physician ', 'Radiology', 
        'Echo', 'ECG', 'General', 'Social Work'
    ]
    
    for hadm_id in zero_hadms:
        cursor = conn.execute("""
            SELECT CATEGORY, TEXT 
            FROM NOTEEVENTS 
            WHERE HADM_ID = ?
            ORDER BY CHARTTIME
        """, (hadm_id,))
        rows = cursor.fetchall()
        
        if not rows:
            not_found_hadms.append(hadm_id)
            continue
        
        # Prioritize certain note types
        notes_by_cat = {}
        for cat, text in rows:
            if cat not in notes_by_cat:
                notes_by_cat[cat] = []
            notes_by_cat[cat].append(text)
        
        # Select notes in priority order
        selected_notes = []
        for cat in priority_categories:
            if cat in notes_by_cat:
                selected_notes.extend(notes_by_cat[cat])
        
        # Add any remaining notes
        for cat in notes_by_cat:
            if cat not in priority_categories:
                selected_notes.extend(notes_by_cat[cat])
        
        if selected_notes:
            combined = concatenate_notes(selected_notes, max_length=10000)
            texts.append(combined)
            found_hadms.append(hadm_id)
        else:
            not_found_hadms.append(hadm_id)
    
    conn.close()
    
    print(f"  Found notes for: {len(found_hadms)} HADMs")
    print(f"  No notes for: {len(not_found_hadms)} HADMs")
    
    if texts:
        # Compute embeddings
        print(f"\nComputing embeddings for {len(texts)} notes...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Update DataFrame
        hadm_to_emb = dict(zip(found_hadms, embeddings))
        
        for idx in zero_indices:
            hadm_id = df.iloc[idx]['HADM_ID']
            if hadm_id in hadm_to_emb:
                df.loc[idx, emb_cols] = hadm_to_emb[hadm_id]
    
    # Check result
    X_emb_new = df[emb_cols].values
    norms_new = np.linalg.norm(X_emb_new, axis=1)
    still_zero = (norms_new == 0).sum()
    print(f"\nAfter filling:")
    print(f"  Still zero embeddings: {still_zero}")
    print(f"  HADMs without any notes: {not_found_hadms}")
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
    print(f"  Shape: {df.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='dataset/MIMIC_III.db')
    parser.add_argument('--features', default='cohort/features_clean_v3.csv')
    parser.add_argument('--output', default='cohort/features_clean_v4.csv')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    
    main(args.db, args.features, args.output, args.model)

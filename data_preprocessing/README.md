create_cohort_icu_readmission.py
===============================

This script builds a fresh cohort for ICU readmission prediction using the MIMIC_III SQLite
database (dataset/MIMIC_III.db). It produces two output files in `cohort/`:

- new_cohort_icu_readmission.csv — cohort rows with aggregated numeric features (per-variable Avg/Std/Min/Max) and `GENDER` as a string (M/F)
- new_cohort_icu_readmission_labels.csv — labels (SUBJECT_ID, HADM_ID, ICUSTAY_ID, Y)

Filters applied:
- Keep first ICU stay for each patient
- Age >= 18 at admission
- Exclude patients who died during the first ICU stay
- Exclude patients who had a later hospital admission after the index discharge (post-discharge readmission)
- Only keep clinical variables measured in >= 80% of the cohort
- Remove patients missing >1/3 of the selected important variables

Run (from repo root, after creating/activating your python environment and installing requirements):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_preprocessing/create_cohort_icu_readmission.py
```

Notes:
- The script re-uses the project's `feature_engineering.py` helpers and reads directly from the
  SQLite DB at `dataset/MIMIC_III.db`.
- The outputs are intentionally saved as new files and will not overwrite existing cohort files.

Additional utility
------------------
- `create_hadm_text_map.py` — Build a HADM_ID -> DISCHARGE_SUMMARY_TEXT map from CSV'd notes and filter cohort by text availability. Useful for NLP pipelines that need consistent mapping from admission IDs to text. Run `python data_preprocessing/create_hadm_text_map.py --help` for usage.
 - `create_hadm_text_map.py` — Build a HADM_ID -> DISCHARGE_SUMMARY_TEXT map from CSV'd notes and filter cohort by text availability. Useful for NLP pipelines that need consistent mapping from admission IDs to text. Run `python data_preprocessing/create_hadm_text_map.py --help` for usage.

Generate NLP features
---------------------
The script also supports converting the HADM_ID -> text map into numeric NLP features suitable for ML training:

 - If `sentence_transformers` is installed, it computes dense sentence embeddings (one vector per HADM_ID), exporting a CSV with columns `HADM_ID,nlp_0,nlp_1,...`. Use `--to-nlp-csv` and `--embed-model` (default `all-MiniLM-L6-v2`) to override model.
 - If `sentence_transformers` is not installed or fails, the script falls back to simple numeric text features (length, word count, unique token count) and writes `HADM_ID,nlp_len,nlp_word_count,nlp_unique_tokens`.

Example to create embedding features (if model installed):
```bash
python data_preprocessing/create_hadm_text_map.py --note-files "cohort/filtered_cohort_with_discharge_summary_test.csv" \
  --to-nlp-csv --nlp-out cohort/nlp_features.csv --embed-model all-MiniLM-L6-v2 --embed-batch 64

Text preprocessing utilities
----------------------------
This project provides a small, dependency-free text preprocessing helper in `data_preprocessing/text_preprocessing.py` which you can use to normalize and tokenize discharge summaries. Example usage in Python:

```py
from data_preprocessing import preprocess

text = "Patient is 23 years old. Presents with abdominal pain."
tokens = preprocess(text)  # lowercases, removes punctuation, removes English stopwords and returns tokens
clean_text = preprocess(text, return_tokens=False)  # returns cleaned string
```
```

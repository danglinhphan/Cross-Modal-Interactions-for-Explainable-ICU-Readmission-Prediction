# DETAILED DATA PIPELINE REPORT
**Project**: ICU Readmission Prediction (Glassbox Ensemble)
**Date**: Dec 29, 2025
**Model Version**: Phase 17 (Champion)

This report documents the rigorous data engineering pipeline used to train the final high-performance models.

---

## 1. Cohort Selection (Filtering Strategy)
The cohort was derived from the **MIMIC-III Critical Care Database**. We applied strict inclusion/exclusion criteria to ensuring the model predicts *true* ICU readmissions and not artifacts.

### Criteria:
1.  **Adult Patients Only**: `Age >= 18` at time of admission.
2.  **First ICU Stay**: We selected the *first* ICU stay of the *first* hospital admission for each patient to ensure independence (no leakage from future history).
3.  **Survival Check**: Patients who died *during* the first ICU stay were excluded (Readmission is impossible).
4.  **Data Quality Filter**: Patients missing more than **33%** of critical clinical variables were removed to ensure high data fidelity.
5.  **Exclusion of Transfers**: (Handled in Labeling) Patients transferred immediately to another facility without opportunity for readmission were filtered out in later stages via "Transfer Count" logic.

**Target Label (`Y`)**:
- **Y=1 (Readmit)**: Patient was discharged from ICU but returned to the ICU (Readmitted) within the **same** Hospital Admission (`HADM_ID`).
- **Y=0 (No Readmit)**: Patient was discharged from ICU and did not return to ICU during the remainder of their hospital stay.

---

## 2. Feature Engineering Pipeline
We engineered **582 Features** across 4 independent modalities.

### A. Clinical Vitals & Labs (Structured Data)
*Source*: `CHARTEVENTS`, `LABEVENTS`.
*Preprocessing*:
- **Aggregation**: For each variable (e.g., Heart Rate, Glucose), we computed: `Min`, `Max`, `Average`, `Standard Deviation` over the entire first ICU stay.
- **Outlier Handling**: Biological bounds applied (e.g., Temperature limited to 10-45°C).
- **Time-Series Derivatives**: `Slope`, `Percentage Change (First vs Last)`, and `Count` (Monitoring Intensity).
- *Key Features*: Glucose Variability, Oxygen Saturation Trends, Nursing Charting Frequency.

### B. High-Fidelity Clinical Features (Phase 12)
*Source*: `INPUTEVENTS`, `OUTPUTEVENTS`, `MICROBIOLOGYEVENTS`.
*Preprocessing*:
- **Fluid Balance**: `Total Input (mL) - Total Output (mL)` per 24h.
- **Infection Markers**: Number of positive microbiology cultures before discharge.
- **Active Antibiotics**: Count of unique antibiotics administered (Proxy for infection severity).

### C. Natural Language Processing (Unstructured Data)
*Source*: `NOTEEVENTS` (Nursing Notes, Discharge Summaries - *Only extracted parts written BEFORE ICU discharge*).
*Preprocessing*:
- **TF-IDF Vectorization**: Top 100 medical keywords (e.g., "Insulin", "Family", "Hemorrhage").
- **Concept Extraction**: SciSpacy used to map text to UMLS concepts (e.g., `C0011849` -> "Diabetes Mellitus").
- **Sentiment Analysis**: "Concern" vs "Reassurance" scores derived from nursing tone.

### D. Social Determinants of Health (Phase 14)
*Source*: `ADMISSIONS`, `PATIENTS`.
*Preprocessing*:
- **Demographics**: Insurance Type (Proxy for SES), Marital Status (Proxy for social support), Ethnicity.
- **Encoding**: One-Hot Encoding used, with missing values mapped to "UNKNOWN".

---

## 3. The "Honesty" Protocol (Leakage Removal)
In **Phase 15**, a rigorous audit identified features that leaked "future" information (information not available at the moment of ICU discharge decision).
**REMOVED Features (Forbidden)**:
1.  `WARD_LOS_HRS`: Length of stay *after* ICU discharge (Future event).
2.  `DISCHARGE_LOCATION`: Where the patient went *after* hospital (Future decision).
3.  `MICRO_POS_48H`: Microbiology results that returned days later.
4.  `TRANSFER_COUNT`: Total transfers in the *whole* admission (includes future transfers).

**Result**: The final "Honest" dataset contains only information strictly available **at or before** the timestamp `ICUSTAY.OUTTIME`.

---

## 4. Modeling & Training Pipeline

### A. Imbalance Handling
- **Problem**: Readmission is rare (~10-15%).
- **Solution**: **Random Under Sampling (RUS)**.
- **Ratio**: `0.4` (Majority class reduced so that Minority is ~28% of data).
- *Note*: **Strictly NO Synthetic Data (SMOTE)** was used in the final model to ensure medical validity.

### B. Algorithm: Glassbox Ensemble (Phase 17)
- **Model**: Explainable Boosting Machine (EBM).
- **Architecture**:
    - 5 Separate EBMs trained with different random seeds.
    - Each EBM discovers "Interactions" (Pairs of features) automatically.
    - **Interactions**: 40 pairs per model x 5 models = Top robust interactions.
- **Predictions**: Soft Voting (Average Probability).

### C. Validation
- **Method**: Train/Test Split (80/20).
- **Stratification**: Preserved class ratio in split.
- **Threshold**: Optimized via Precision-Recall Curve (Threshold `0.7915` chosen to maximize F1).

---

## 5. Final Reproducibility Checklist
To reproduce the best model (`ebm_final_glassbox_ensemble.pkl`), the data flows as follows:

1.  `MIMIC-III.db` (Raw SQL)
2.  `create_cohort_icu_readmission.py` -> `new_cohort_icu_readmission.csv` (Base Cohort)
3.  `extract_phase12_features.py` -> `features_phase12_extra.csv` (Fluids)
4.  `extract_phase14_social.py` -> `features_phase14_social.csv` (Social)
5.  `train_phase16_honest.py` -> Loads all CSVs -> **DROPS LEAKAGE** -> `X_honest`.
6.  `train_phase17_glassbox.py` -> Loads `X_honest` -> Applies RUS(0.4) -> Trains Ensemble -> Output.

This pipeline ensures 100% transparency and reproducibility.

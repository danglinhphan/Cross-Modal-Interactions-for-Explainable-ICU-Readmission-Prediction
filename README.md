# Cross-Modal Interactions for Explainable ICU Readmission Prediction

## Project Overview
This project develops an interpretable AI model to predict **ICU Readmission Risk**. 
Unlike traditional "Black Box" models, our **Glassbox Ensemble** provides fully transparent explanations for every prediction, helping clinicians understand *why* a patient is at risk.

## Key Achievements
- **State-of-the-art Performance**: Achieved **F1 Score 0.85** on the MIMIC-III dataset.
- **Cross-Modal Synergy**: The model discovers interactions between **Clinical Vitals** (e.g., Glucose) and **Nursing Notes** (e.g., "Insulin"), significantly boosting precision.
- **Data Integrity**: Rigorous removal of all future-leakage features to ensure the model is "Safe & Honest" for deployment.

## Deliverables (Where is the code?)
The final, optimized version of the project is located in the **[Final_Deliverables_Glassbox](Final_Deliverables_Glassbox)** folder.

> **[Go to Final Deliverables Folder](Final_Deliverables_Glassbox)**
> Contains:
> - **Champion Model** (`ebm_final_glassbox_ensemble.pkl`)
> - **Training Code** (Reproducible Pipeline)
> - **Deployment Instructions**
> - **Detailed Metrics Report**

## How it works (Simplified)
Instead of using complex Neural Networks that are hard to trust, we use **Explainable Boosting Machines (EBMs)**.
1.  **Transparent**: You can see exactly how much "Heart Rate" or "Age" contributes to risk.
2.  **Smart Interactions**: The model answers questions like: *"Is High Heart Rate bad? It depends on whether the nurse noted 'Anxiety' or 'Sepsis'."*

## Repository Structure
- `Final_Deliverables_Glassbox/`: **Start Here**. The Golden Set of code/models.
- `Research_Archive/`: Archive of all experimental scripts and previous phases (Phase 1-16).
- `cohort/` & `dataset/`: Raw data directories.

## Contact
Developed by VRES Optimization Team.
For questions, please open an Issue.

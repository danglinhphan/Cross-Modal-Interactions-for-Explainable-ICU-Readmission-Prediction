# VRES (Ventilator & Readmission Prediction System)

This repository contains the codebase for the VRES project, focusing on predicting ICU readmissions and ventilator events using various machine learning and deep learning approaches.

## Project Structure

*   **Train_EBM/**: Explainable Boosting Machine (EBM) models and optimization scripts.
*   **Train_model_XGB/**: XGBoost models and inference scripts.
*   **cohort/**: Cohort generation code and intermediate data processing (large data files are excluded).
*   **dataset/**: Database interaction and raw data handling (MIMIC-III/IV DBs are excluded).
*   **data_preprocessing/**: NLP pipelines, embedding generation, and cleaning scripts.
*   **ebm-llm-frontend/**: Next.js frontend application for visualizing model explanations.
*   **scripts/**: Utility and maintenance scripts.
*   **outputs/**: Directory for model outputs and logs (excluded from repo).

## Getting Started

1.  **Environment Setup**: Install dependencies from `requirements.txt`.
2.  **Data**: Ensure you have access to the MIMIC-III/IV database. The `dataset/` folder expects the database files to be present locally (but they are ignored by git).
3.  **Training**: Explore `Train_EBM` or `Train_model_XGB` for training scripts.

## License
[Insert License Here]

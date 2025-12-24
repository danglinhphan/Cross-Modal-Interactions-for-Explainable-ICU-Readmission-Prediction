# Cross-Modal Interactions for Explainable ICU Readmission Prediction

## 1. Project Overview
This project presents an Interpretable Deep Learning framework for identifying ICU patients at high risk of **unplanned readmission**.

Unlike standard "black-box" models (e.g., Deep Neural Networks, XGBoost), our approach uses an **Explainable Boosting Machine (EBM)** to achieve state-of-the-art performance while remaining fully transparent. The core innovation is the discovery of **"Cross-Modal Interactions"**—specific ways in which clinical notes (NLP) contextualize vital signs (Structured Data).

## 2. The Core Problem: Lack of Context
Standard models treat vital signs and clinical notes as separate streams of information.
*   **Vital Signs:** "Heart Rate is 110." (High, but is it bad?)
*   **Clinical Notes:** "Patient is anxious about discharge." vs "Patient has signs of sepsis."

**The Missing Link:** A heart rate of 110 is benign in an anxious patient but life-threatening in a septic patient. Standard models miss this *contextual* nuance. Our model explicitly learns these $f(\text{Vital}, \text{NLP})$ relationships.

## 3. Solution: Cross-Modal EBM
We use an **Explainable Boosting Machine (EBM)**, which is a Generalized Additive Model (GAM) with interaction terms.

### The Formula
The model predicts the log-odds of readmission:

$$
\log\left(\frac{P(Y=1)}{1-P(Y=1)}\right) = \beta_0 + \underbrace{\sum f_i(x_{\text{vital}})}_{\text{Vitals}} + \underbrace{\sum g_j(x_{\text{nlp}})}_{\text{NLP}} + \underbrace{\sum h_{k}(x_{\text{vital}}, x_{\text{nlp}})}_{\text{Cross-Interactions}}
$$

### Why EBM?
1.  **Accuracy of Boosting:** It uses gradient boosting principles (like XGBoost/LightGBM) to learn complex non-linear patterns.
2.  **Interpretability of Linear Models:** Instead of a giant forest of trees, it learns **Shape Functions** $f(x)$ for each feature. We can plot $f(\text{HeartRate})$ to see exactly how risk changes as Heart Rate increases.

## 4. What is "Cross-Interaction"?
A cross-interaction is a mathematical term $h(v, t)$ where the impact of a vital sign $v$ depends on the presence of a textual concept $t$.

**Example: Lactate Level $\times$ "Liver Failure"**
*   **Scenario A (No Liver Failure):** High Lactate is a strong alarm signal for Sepsis/Hypoperfusion. Risk of readmission spikes rapidly.
*   **Scenario B (Liver Failure present):** High Lactate is expected due to poor clearance by the liver. It is *less* alarmist. The model learns a flatter risk curve.

This allows the model to "think" like a doctor: *"The lactate is high, but that's just his liver condition, don't panic."*

## 5. Methodology: The FAST Algorithm
Finding meaningful interactions is computationally expensive ($300 \text{ Vitals} \times 4000 \text{ Concepts} = 1.2 \text{ million pairs}$). We developed the **FAST (Feature Analysis for Selection of Transformations)** algorithm to efficiently discover them.

### Step 1: Train Main Effects
We first train a standard EBM with **NO** interactions:
$$ \hat{y}_{\text{main}} \approx \sum f(V) + \sum g(T) $$
This model captures the baseline risk of all features independently.

### Step 2: Compute Residuals
We calculate the errors (residuals) of this model:
$$ r = y - \hat{y}_{\text{main}} $$
The residual $r$ represents the "unexplained" risk.

### Step 3: Rank Interactions
For every pair $(v, t)$, we check if the residual pattern of $v$ changes when $t$ is present vs absent.
*   We bin the vital sign $v$ (e.g., low, medium, high).
*   We compare the residuals in each bin for patients with $t=0$ vs $t=1$.
*   **Logic:** If the "error" pattern is significantly different, it means the Main Effects model failed to capture a dependency between $v$ and $t$.

### Step 4: Retrain Final Model
We select the Top-N (e.g., 50) ranked pairs and retrain the EBM from scracth, forcing it to learn these specific interaction terms $h(v, t)$.

## 6. Results
Our **EBM-Cross** model achieves performance comparable to or better than "Black-Box" deep learning models while offering full transparency.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **F1-Score** | **75.22%** | High balance of Precision/Recall |
| **AUC-ROC** | **97.78%** | Excellent ranking capability |
| **Accuracy** | 96.42% | |

## 7. Repository Structure
*   `Train_EBM/`: Code for training the EBM and running the FAST algorithm.
*   `Train_model_XGB/`: XGBoost baselines.
*   `ebm-llm-frontend/`: A Next.js web application to visualize the model's explanations for doctors.
*   `cohort/`: Feature extraction and data processing logic.

## 8. Usage
To train the model (assuming you have MIMIC-III access):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Cross-Interaction Discovery
python Train_EBM/cross_interaction_discovery.py \
    --vital-features cohort/features_phase4_clinical.csv \
    --nlp-features cohort/nlp_features.csv \
    --output outputs/interactions

# 3. Train Final EBM
python Train_EBM/train_ebm_enhanced.py --interactions outputs/interactions/top_50.json
```

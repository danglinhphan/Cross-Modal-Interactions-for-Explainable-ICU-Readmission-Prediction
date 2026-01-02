
"""
Phase 19 Serving Script (Strict Honest Model)
- Loads EBM Ensemble
- Loads TF-IDF Vectorizer (Strict Honest)
- Processes Input Text dynamically
- Returns Prediction + Explanation + Highlighted Text
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any, Optional
import uvicorn
import os, pickle, json, re
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ICU Bounceback Prediction (Phase 19)")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume run from root or Final_Deliverables
ROOT_DIR = os.path.join(BASE_DIR, '..') 
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'ebm_phase19_strict')

MODEL_PATH = os.path.join(OUTPUT_DIR, 'ebm_ensemble_strict.pkl')
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.json')

class PatientData(BaseModel):
    clinical_features: Dict[str, Any]
    text: Optional[str] = ""

class PredictResponse(BaseModel):
    prediction: int    # 0 or 1
    probability: float # 0.0 - 1.0
    risk_score: int    # 0 - 100
    threshold: float
    highlighted_text: str # HTML/Markdown highlighted
    explanation: List[Dict[str, Any]] # SHAP/EBM contributions
    llm_narrative: str # Generated text explanation

# Global State
models = []
vectorizer = None
threshold = 0.5
feature_names = []

@app.on_event("startup")
def load_artifacts():
    global models, vectorizer, threshold, feature_names
    
    print(f"Loading Model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
        
    print(f"Loading Vectorizer from {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            threshold = metrics.get('threshold', 0.5)
            print(f"Loaded Threshold: {threshold}")
            
    # Get feature names from first model
    try:
        feature_names = models[0].feature_names_in_
    except:
        feature_names = models[0].feature_names
        
    print("Serving Ready!")

def highlight_text(text: str, vectorizer) -> str:
    """
    Highlights words in text that are present in the TF-IDF vocabulary.
    Returns HTML string with <mark> tags.
    """
    if not text:
        return ""
        
    vocab = set(vectorizer.get_feature_names_out())
    # Simple tokenization for display (split by non-alphanumeric)
    # We want to preserve punctuation in display but match tokens
    
    def replacer(match):
        word = match.group(0)
        # Vectorizer processing (lowercase)
        clean_word = word.lower()
        if clean_word in vocab:
            # Highlight
            return f'<mark style="background-color: #ffeb3b; font-weight: bold;">{word}</mark>'
        return word
        
    # Replace words
    highlighted = re.sub(r'\b\w+\b', replacer, text)
    return highlighted

@app.post("/predict", response_model=PredictResponse)
def predict(data: PatientData):
    global models, vectorizer, threshold, feature_names
    
    if not models or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    # 1. Process Text
    raw_text = data.text or ""
    tfidf_matrix = vectorizer.transform([raw_text])
    tfidf_cols = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)
    
    # 2. Process Clinical
    clinical_data = data.clinical_features.copy()
    # Normalize keys (uppercase)
    clinical_data = {k.upper(): v for k, v in clinical_data.items()}
    df_clinical = pd.DataFrame([clinical_data])
    
    # 3. Merge
    df_input = pd.concat([df_clinical, df_tfidf], axis=1)
    
    # 4. Alignment
    # Add missing cols with 0
    missing_cols = [c for c in feature_names if c not in df_input.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df_input.index, columns=missing_cols)
        df_input = pd.concat([df_input, df_missing], axis=1)
        
    # Order columns
    df_input = df_input[feature_names]
    
    # 5. Predict (Ensemble)
    probas = np.zeros(len(df_input))
    explanations = {} # Sum contributions
    
    for model in models:
        probas += model.predict_proba(df_input)[:, 1]
        
        # Local Explanation using EBM
        local_exp = model.explain_local(df_input, df_input['Y'] if 'Y' in df_input else None)
        # Extract data from explanation (this is tricky with EBM API)
        # EBM explain_local returns an Explanation object.
        # We can simulate contributions or just use interpret's data
        # Valid EBM has .explain_local() which usually works on fitted data. 
        # For new data, it works too.
        
        data_dicts = local_exp.data(0)
        names = data_dicts['names']
        scores = data_dicts['scores']
        values = data_dicts['values']
        
        for n, s, v in zip(names, scores, values):
            if n not in explanations:
                explanations[n] = {'score': 0, 'value': v}
            explanations[n]['score'] += s

    # Average
    avg_proba = probas[0] / len(models)
    avg_explanations = []
    
    for k, v in explanations.items():
        avg_score = v['score'] / len(models)
        if abs(avg_score) > 0.001: # Filter noise
            avg_explanations.append({
                'feature': k,
                'value': v['value'],
                'contribution': avg_score,
                'type': 'risk' if avg_score > 0 else 'protective'
            })
            
    # Sort by absolute impact
    avg_explanations.sort(key=lambda x: abs(x['contribution']), reverse=True)
    top_factors = avg_explanations[:10]

    # --- NARRATIVE GENERATION (Pseudo-LLM) ---
    narrative = generate_clinical_narrative(avg_proba, threshold, top_factors)
    
    return PredictResponse(
        prediction=int(avg_proba >= threshold),
        probability=float(avg_proba),
        risk_score=int(avg_proba * 100),
        threshold=float(threshold),
        highlighted_text=highlight_text(raw_text, vectorizer),
        explanation=top_factors, # Structured data
        llm_narrative=narrative  # Natural language
    )

def generate_clinical_narrative(proba: float, thresh: float, factors: List[Dict]) -> str:
    """Generates a clinical narrative explaining the risk score."""
    risk_level = "HIGH" if proba >= thresh else "LOW"
    score_pct = int(proba * 100)
    
    lines = [f"**Patient Risk Assessment: {risk_level} RISK (Score: {score_pct}/100)**\n"]
    
    if risk_level == "HIGH":
        lines.append(f"The patient shows significant signs of instability, with a calculated probability of ICU bounceback of {proba:.2f} (Threshold: {thresh:.2f}).\n")
        lines.append("**Primary Drivers of Risk:**")
    else:
        lines.append(f"The patient appears relatively stable calculating a low probability of readmission ({proba:.2f}). However, specific factors warrant monitoring.\n")
        lines.append("**Key Observations:**")
        
    for f in factors:
        name = f['feature'].replace('tfidf_', 'Note keyword: ').replace('_', ' ')
        val = f['value']
        impact = f['contribution']
        direction = "increases risk" if impact > 0 else "reduces risk"
        
        # Clinical translation of common features
        if "URINE" in name.upper() and impact > 0:
            lines.append(f"- **Urine Output**: Low output ({val}) suggests potential kidney strain or dehydration, which significantly {direction}.")
        elif "LACTATE" in name.upper() and impact > 0:
            lines.append(f"- **Lactate**: Levels or monitoring intensity ({val}) indicate ongoing tissue hypoperfusion.")
        elif "keyword" in name:
            term = name.split(": ")[1]
            lines.append(f"- **Clinical Note**: The term '{term}' appearing in nursing notes is associated with {direction}.")
        else:
            lines.append(f"- **{name}**: Value of {val} {direction} (Impact: {impact:.3f}).")
            
    lines.append("\n**Recommendation:**")
    if risk_level == "HIGH":
        lines.append("Consider postponing discharge or transferring to a step-down unit with enhanced monitoring. Review fluid balance and infection markers.")
    else:
        lines.append("Proceed with discharge planning according to standard protocol, but maintain vigilance on the flagged factors.")
        
    return "\n".join(lines)

if __name__ == "__main__":
    uvicorn.run("serve_phase19:app", host="0.0.0.0", port=8000, reload=True)

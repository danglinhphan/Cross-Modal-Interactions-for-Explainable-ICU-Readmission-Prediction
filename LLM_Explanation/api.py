"""
FastAPI Backend for EBM + LLM Explanation Service.
Connects the frontend with EBM model and LLM explanation generation.
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn

# Import our LLM_Explanation modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LLM_Explanation.config import LLMConfig
from LLM_Explanation.ebm_explainer import EBMExplainer
from LLM_Explanation.llm_service import LLMService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="EBM + LLM Explanation API",
    description="API for ICU Readmission Risk Prediction with Explainable AI",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy loaded)
_config: Optional[LLMConfig] = None
_ebm_explainer: Optional[EBMExplainer] = None
_llm_service: Optional[LLMService] = None


def get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config


def get_ebm_explainer() -> EBMExplainer:
    global _ebm_explainer
    if _ebm_explainer is None:
        _ebm_explainer = EBMExplainer(get_config())
    return _ebm_explainer


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(get_config())
    return _llm_service


# ============== Pydantic Models ==============

class Feature(BaseModel):
    """Feature contribution from EBM."""
    name: str
    impact: float  # contribution score
    value: str  # human-readable description
    direction: str = "increases"  # increases or decreases risk


class CrossInteraction(BaseModel):
    """Cross-interaction between vital and NLP feature."""
    vital: str
    nlp: str
    description: str
    importance: float


class PredictionRequest(BaseModel):
    """Request body for prediction."""
    patient_data: Dict[str, Any] = Field(..., description="Patient features (320 features)")
    clinical_notes: Optional[str] = Field(None, description="Raw clinical notes for NLP extraction")
    generate_llm_explanation: bool = Field(True, description="Whether to generate LLM explanation")


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    riskScore: float = Field(..., description="Risk score 0-100")
    confidence: float = Field(..., description="Model confidence")
    riskLevel: str = Field(..., description="Risk level: Low/Moderate/Elevated/High")
    features: List[Feature] = Field(..., description="Top contributing features")
    crossInteractions: List[CrossInteraction] = Field(default=[], description="Active cross-interactions")
    features: List[Feature] = Field(..., description="Top contributing features")
    crossInteractions: List[CrossInteraction] = Field(default=[], description="Active cross-interactions")
    llmExplanation: str = Field(..., description="Patient-friendly explanation from LLM")
    nlpHighlights: List[NLPHighlight] = Field(default=[], description="NLP highlights in clinical notes")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ebm_loaded: bool
    ollama_available: bool
    model_name: str


# ============== API Endpoints ==============

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "EBM + LLM Explanation API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model availability."""
    config = get_config()
    ebm = get_ebm_explainer()
    llm = get_llm_service()
    
    return HealthResponse(
        status="healthy",
        ebm_loaded=ebm.model is not None,
        ollama_available=llm.check_ollama_available(),
        model_name=config.model_name
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict ICU readmission risk and generate explanation.
    
    Accepts patient data (320 features) and optional clinical notes.
    Returns risk score, contributing features, and LLM explanation.
    """
    try:
        ebm = get_ebm_explainer()
        
        # Get EBM prediction and explanation
        ebm_result = ebm.explain(
            patient_data=request.patient_data,
            clinical_notes=request.clinical_notes
        )
        
        # Format features for frontend
        features = [
            Feature(
                name=f["name"],
                impact=f["contribution"] * 100,  # Scale to percentage points
                value=f["description"],
                direction=f["direction"]
            )
            for f in ebm_result.top_factors
        ]
        
        # Format cross-interactions
        interactions = [
            CrossInteraction(
                vital=i["vital"],
                nlp=i["nlp"],
                description=i["description"],
                importance=i["importance"]
            )
            for i in ebm_result.cross_interactions
        ]
        
        if request.generate_llm_explanation:
            try:
                llm = get_llm_service()
                if llm.check_ollama_available():
                    patient_explanation = llm.generate_explanation(ebm_result)
                    llm_explanation = patient_explanation.summary
                else:
                    llm_explanation = _generate_fallback_explanation(ebm_result)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                llm_explanation = _generate_fallback_explanation(ebm_result)
        else:
            llm_explanation = _generate_fallback_explanation(ebm_result)
            
        # Extract NLP Highlights for Visualization
        # Use TF-IDF vectorizer from EBMExplainer if available (Phase 19 requirement)
        if ebm.vectorizer:
            h_data = ebm.extract_tfidf_highlights(request.clinical_notes or "")
            highlights = [NLPHighlight(**h) for h in h_data]
            logger.info(f"Generated {len(highlights)} TF-IDF highlights")
        else:
            # Fallback to regex extractor if no vectorizer
            from LLM_Explanation.nlp_extractor import NLPExtractor
            extractor = NLPExtractor()
            nlp_res = extractor.extract_with_positions(request.clinical_notes or "")
            highlights = [NLPHighlight(**h) for h in nlp_res["highlights"]]
        
        return PredictionResponse(
            riskScore=ebm_result.risk_score * 100,  # Convert to percentage
            confidence=85.02,  # F1 Score from full_metrics.txt (Global Model Performance)
            riskLevel=ebm_result.risk_level,
            features=features,
            crossInteractions=interactions,
            llmExplanation=llm_explanation,
            nlpHighlights=highlights
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_fallback_explanation(ebm_result) -> str:
    """Generate a simple explanation when LLM is not available."""
    risk_pct = ebm_result.risk_score * 100
    level = ebm_result.risk_level
    
    factors_text = ""
    for f in ebm_result.top_factors[:3]:
        direction = "increases" if f["contribution"] > 0 else "decreases"
        factors_text += f"\n• {f['description']} ({direction} risk)"
    
    return f"""Based on the analysis, the ICU readmission risk is {risk_pct:.1f}% ({level} risk).

Key contributing factors:{factors_text}

Please consult with your healthcare provider for personalized medical advice. This assessment is for informational purposes only."""


@app.post("/predict/ebm-only", response_model=dict)
async def predict_ebm_only(request: PredictionRequest):
    """
    Get EBM prediction only (no LLM explanation).
    Faster response for real-time use cases.
    """
    try:
        ebm = get_ebm_explainer()
        ebm_result = ebm.explain(
            patient_data=request.patient_data,
            clinical_notes=request.clinical_notes
        )
        
        return {
            "riskScore": ebm_result.risk_score * 100,
            "riskLevel": ebm_result.risk_level,
            "topFactors": ebm_result.top_factors,
            "crossInteractions": ebm_result.cross_interactions
        }
    except Exception as e:
        logger.error(f"EBM prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", response_model=dict)
async def get_features():
    """Get list of all 320 feature names expected by the model."""
    ebm = get_ebm_explainer()
    return {
        "total": len(ebm.feature_names) if ebm.feature_names else 0,
        "features": ebm.feature_names or []
    }


@app.get("/interactions", response_model=dict)
async def get_interactions():
    """Get list of all 75 cross-interactions in the model."""
    ebm = get_ebm_explainer()
    return {
        "total": len(ebm.cross_interactions) if ebm.cross_interactions else 0,
        "interactions": ebm.cross_interactions or []
    }


# ============== NLP Extraction Endpoint ==============

class NLPExtractionRequest(BaseModel):
    """Request for NLP extraction."""
    clinical_notes: str = Field(..., description="Clinical notes text")


class NLPHighlight(BaseModel):
    """Single highlight in clinical notes."""
    feature: str
    text: str
    start: int
    end: int


class NLPExtractionResponse(BaseModel):
    """Response from NLP extraction."""
    features: Dict[str, int] = Field(..., description="41 NLP features (0/1)")
    highlights: List[NLPHighlight] = Field(..., description="Matched positions in text")
    active_count: int = Field(..., description="Number of active features")


@app.post("/extract-nlp", response_model=NLPExtractionResponse)
async def extract_nlp(request: NLPExtractionRequest):
    """
    Extract NLP features from clinical notes with highlighting.
    
    Returns 41 binary NLP features and their positions in the text.
    """
    from LLM_Explanation.nlp_extractor import NLPExtractor
    
    extractor = NLPExtractor()
    result = extractor.extract_with_positions(request.clinical_notes)
    
    active_count = sum(1 for v in result["features"].values() if v == 1)
    
    return NLPExtractionResponse(
        features=result["features"],
        highlights=[NLPHighlight(**h) for h in result["highlights"]],
        active_count=active_count
    )


# ============== Raw Data Processing Endpoint ==============

class HADMPredictionRequest(BaseModel):
    """Request for prediction from HADM_ID."""
    hadm_id: int = Field(..., description="Hospital admission ID from MIMIC-III")
    generate_llm_explanation: bool = Field(True, description="Whether to generate LLM explanation")


class HADMPredictionResponse(BaseModel):
    """Response from HADM prediction."""
    hadm_id: int
    riskScore: float
    confidence: float
    riskLevel: str
    features: List[Feature]
    nlpHighlights: List[NLPHighlight]
    clinicalNotes: str
    llmExplanation: str
    patientInfo: Dict[str, Any]


@app.post("/predict/from-hadm", response_model=HADMPredictionResponse)
async def predict_from_hadm(request: HADMPredictionRequest):
    """
    Predict ICU readmission risk from raw MIMIC data.
    
    Takes HADM_ID, queries MIMIC-III database, computes features, and returns prediction.
    """
    try:
        from LLM_Explanation.feature_engineering import FeatureEngineer
        from LLM_Explanation.nlp_extractor import NLPExtractor
        
        # Compute features from raw data
        engineer = FeatureEngineer()
        raw_result = engineer.compute_all_features(request.hadm_id)
        
        patient_features = raw_result["features"]
        clinical_notes = raw_result["clinical_notes"]
        patient_info = raw_result["patient_info"]
        
        # Extract NLP features with highlights
        nlp_extractor = NLPExtractor()
        nlp_result = nlp_extractor.extract_with_positions(clinical_notes)
        
        # Merge NLP features into patient_features
        patient_features.update(nlp_result["features"])
        
        # Get EBM prediction
        ebm = get_ebm_explainer()
        ebm_result = ebm.explain(patient_features)
        
        # Format features
        features = [
            Feature(
                name=f["name"],
                impact=f["contribution"] * 100,
                value=f["description"],
                direction=f["direction"]
            )
            for f in ebm_result.top_factors
        ]
        
        # Generate LLM explanation if requested
        llm_explanation = ""
        if request.generate_llm_explanation:
            try:
                llm = get_llm_service()
                if llm.check_ollama_available():
                    patient_explanation = llm.generate_explanation(ebm_result)
                    llm_explanation = patient_explanation.summary
                else:
                    llm_explanation = _generate_fallback_explanation(ebm_result)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                llm_explanation = _generate_fallback_explanation(ebm_result)
        
        return HADMPredictionResponse(
            hadm_id=request.hadm_id,
            riskScore=ebm_result.risk_score * 100,
            confidence=75.22,
            riskLevel=ebm_result.risk_level,
            features=features,
            nlpHighlights=[NLPHighlight(**h) for h in nlp_result["highlights"]],
            clinicalNotes=clinical_notes[:5000] if clinical_notes else "",  # Limit for response
            llmExplanation=llm_explanation,
            patientInfo=patient_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"HADM prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Main ==============

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

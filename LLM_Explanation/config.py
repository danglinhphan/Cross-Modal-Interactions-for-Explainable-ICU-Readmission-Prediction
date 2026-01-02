"""
Configuration for LLM Explanation Service.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base paths
MODULE_DIR = Path(__file__).parent
PROMPTS_DIR = MODULE_DIR / "prompts"
EBM_MODEL_DIR = Path("/Users/phandanglinh/Desktop/VRES/outputs/ebm_optimized_final")

@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    # Model settings - Med42: Clinically-aligned medical model from M42 Health
    model_name: str = "thewindmom/llama3-med42-8b"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    
    # Prompt settings
    prompt_template: str = "patient_explanation.txt"
    
    # EBM Model paths
    # Phase 19: Strict Honest (TF-IDF Auto) - ACTIVE
    ebm_model_path: str = "/Users/phandanglinh/Desktop/VRES/outputs/ebm_phase19_strict/ebm_ensemble_strict.pkl"
    vectorizer_path: str = "/Users/phandanglinh/Desktop/VRES/outputs/ebm_phase19_strict/tfidf_vectorizer.pkl"
    feature_names_path: str = "/Users/phandanglinh/Desktop/VRES/outputs/ebm_phase19_strict/feature_info.json"
    cross_interactions_path: str = "/Users/phandanglinh/Desktop/VRES/outputs/ebm_phase19_strict/ebm_interactions.json"
    threshold: float = 0.7735  # Phase 19 Optimal Threshold
    
    # Output settings
    top_k_factors: int = 5
    top_k_interactions: int = 3


@dataclass 
class ExplanationResult:
    """Result from EBM explanation."""
    risk_score: float
    risk_level: str
    top_factors: list = field(default_factory=list)
    cross_interactions: list = field(default_factory=list)
    
    def to_prompt_context(self) -> dict:
        """Convert to context dict for prompt template."""
        factors_str = "\n".join([
            f"- {f['name']}: {f['description']} (contribution: {f['contribution']:.3f})"
            for f in self.top_factors
        ])
        
        interactions_str = "\n".join([
            f"- {i['vital']} + {i['nlp']}: {i['description']}"
            for i in self.cross_interactions
        ])
        
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "top_factors": factors_str if factors_str else "No significant factors identified.",
            "cross_interactions": interactions_str if interactions_str else "No significant interactions."
        }


@dataclass
class PatientExplanation:
    """Patient-friendly explanation from LLM."""
    summary: str
    key_points: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    disclaimer: str = "This is for informational purposes only and does not replace professional medical advice."
    raw_response: Optional[str] = None

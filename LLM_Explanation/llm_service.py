"""
LLM Service for generating patient-friendly explanations.
Uses Ollama with Medical-Llama3-8B.
"""
import logging
from pathlib import Path
from typing import Optional

from .config import LLMConfig, ExplanationResult, PatientExplanation, PROMPTS_DIR

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating patient explanations using Ollama LLM."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = None
        self.prompt_template = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of LLM and prompt template."""
        if self._initialized:
            return
        
        # Load prompt template
        prompt_path = PROMPTS_DIR / self.config.prompt_template
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
            logger.info(f"Loaded prompt template from {prompt_path}")
        else:
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        
        # Initialize Ollama LLM
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=self.config.model_name,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
            )
            logger.info(f"Initialized Ollama with model: {self.config.model_name}")
        except ImportError:
            logger.error("langchain-ollama not installed. Run: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
        
        self._initialized = True
    
    def _format_prompt(self, explanation: ExplanationResult) -> str:
        """Format the prompt template with explanation data."""
        context = explanation.to_prompt_context()
        
        return self.prompt_template.format(
            risk_score=context["risk_score"],
            risk_level=context["risk_level"],
            top_factors=context["top_factors"],
            cross_interactions=context["cross_interactions"]
        )
    
    def generate_explanation(
        self, 
        ebm_result: ExplanationResult
    ) -> PatientExplanation:
        """
        Generate patient-friendly explanation from EBM result.
        
        Args:
            ebm_result: ExplanationResult from EBMExplainer
            
        Returns:
            PatientExplanation with summary, key points, and disclaimer
        """
        self._ensure_initialized()
        
        # Format prompt
        prompt = self._format_prompt(ebm_result)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Call LLM
        try:
            response = self.llm.invoke(prompt)
            logger.info("Successfully generated LLM response")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return PatientExplanation(
                summary=f"Unable to generate explanation. Error: {str(e)}",
                raw_response=None
            )
        
        # Parse response
        return self._parse_response(response, ebm_result)
    
    def _parse_response(
        self, 
        response: str, 
        ebm_result: ExplanationResult
    ) -> PatientExplanation:
        """Parse LLM response into structured PatientExplanation."""
        # Extract key points (sentences that contain important info)
        sentences = response.split('. ')
        key_points = []
        
        for s in sentences[:5]:  # Take first 5 sentences as potential key points
            s = s.strip()
            if len(s) > 20 and not s.lower().startswith("this is for informational"):
                key_points.append(s + ('.' if not s.endswith('.') else ''))
        
        # Extract recommendations (look for advice-like phrases)
        recommendations = []
        advice_keywords = ["should", "recommend", "advised", "important to", "consider"]
        for s in sentences:
            if any(kw in s.lower() for kw in advice_keywords):
                recommendations.append(s.strip() + ('.' if not s.endswith('.') else ''))
        
        return PatientExplanation(
            summary=response.strip(),
            key_points=key_points[:3],
            recommendations=recommendations[:2],
            disclaimer="This is for informational purposes only and does not replace professional medical advice.",
            raw_response=response
        )
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            import requests
            response = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                logger.info(f"Available Ollama models: {model_names}")
                return any(self.config.model_name.split(":")[0] in n for n in model_names)
            return False
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

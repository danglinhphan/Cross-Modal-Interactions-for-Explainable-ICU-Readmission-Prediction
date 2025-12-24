#!/usr/bin/env python3
"""
CLI for testing LLM Explanation module locally.
Usage: python -m LLM_Explanation.main [options]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from LLM_Explanation.config import LLMConfig
from LLM_Explanation.ebm_explainer import EBMExplainer
from LLM_Explanation.llm_service import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_patient(sample_path: str = None) -> dict:
    """Load a sample patient from file or generate synthetic data."""
    if sample_path and Path(sample_path).exists():
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    # Synthetic sample patient (typical ICU discharge patient)
    # NOTE: NLP features will be auto-extracted from clinical_notes if provided
    return {
        # Demographics
        "AGE": 68,
        "LOS_Hospital": 7,
        "LOS_ICU": 3,
        
        # Lab/Vital Features (279 features, showing key ones)
        "Creatinine_Std": 0.8,
        "Creatinine_Max": 2.1,
        "WBC_Last": 11.2,
        "WBC_abnormal_last": 1,
        "Platelet_Min": 95,
        "PTT_Std": 5.2,
        "Lactate_Max": 3.1,
        "SBP_Min": 88,
        "SpO2_Min": 91,
        "GCS_Min": 13,
        "SIRS_Score": 2,
        "MEWS_Score": 4,
        "Ventilation_Usage": 1,
        "CCI_Score": 5,
        
        # Clinical Notes (NLP features will be extracted from this)
        "clinical_notes": """
        DISCHARGE SUMMARY:
        68 year old male admitted for cellulitis of left lower extremity.
        Hospital course complicated by pneumonia requiring supplemental oxygen.
        Past medical history significant for diabetes mellitus type 2, 
        hypertension, and chronic kidney disease stage 3.
        
        Patient was treated with IV antibiotics for cellulitis with improvement.
        Developed hospital-acquired pneumonia on day 3, treated with broad-spectrum
        antibiotics. Required intermittent ventilatory support via BiPAP.
        
        Labs notable for elevated WBC (peak 15.2), creatinine stable at 2.1.
        No evidence of sepsis or septic shock.
        
        Patient stable for discharge to skilled nursing facility.
        Follow-up with primary care in 1 week.
        """
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM Explanation for EBM Model"
    )
    parser.add_argument(
        "--patient", "-p",
        type=str,
        default=None,
        help="Path to JSON file with patient data"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="thewindmom/llama3-med42-8b",
        help="Ollama model name (default: thewindmom/llama3-med42-8b)"
    )
    parser.add_argument(
        "--ebm-only",
        action="store_true",
        help="Only run EBM explanation (skip LLM)"
    )
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Check if Ollama is available and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration
    config = LLMConfig(model_name=args.model)
    
    # Check Ollama availability
    if args.check_ollama:
        llm_service = LLMService(config)
        available = llm_service.check_ollama_available()
        if available:
            print(f"✅ Ollama is available with model: {args.model}")
            return 0
        else:
            print(f"❌ Ollama not available or model {args.model} not found")
            print("   Run: ollama pull lazarevtill/Medical-Llama3-8B")
            return 1
    
    # Load patient data
    patient_data = load_sample_patient(args.patient)
    print("\n" + "="*60)
    print("🏥 PATIENT DATA")
    print("="*60)
    print(json.dumps(patient_data, indent=2)[:500] + "...")
    
    # Step 1: EBM Explanation
    print("\n" + "="*60)
    print("📊 EBM EXPLANATION")
    print("="*60)
    
    try:
        explainer = EBMExplainer(config)
        
        # Extract clinical_notes if present (for NLP auto-extraction)
        clinical_notes = patient_data.pop("clinical_notes", None)
        if clinical_notes:
            print("\n📝 Clinical notes provided - auto-extracting NLP features...")
        
        ebm_result = explainer.explain(patient_data, clinical_notes=clinical_notes)
        
        print(f"\n🎯 Risk Score: {ebm_result.risk_score:.1%}")
        print(f"📈 Risk Level: {ebm_result.risk_level}")
        
        print("\n📋 Top Contributing Factors:")
        for i, factor in enumerate(ebm_result.top_factors, 1):
            print(f"  {i}. {factor['name']}: {factor['description']}")
            print(f"     Contribution: {factor['contribution']:.4f} ({factor['direction']} risk)")
        
        print("\n🔗 Active Cross-Interactions:")
        for i, interaction in enumerate(ebm_result.cross_interactions, 1):
            print(f"  {i}. {interaction['description']}")
            
    except Exception as e:
        logger.error(f"EBM explanation failed: {e}")
        print(f"❌ EBM explanation error: {e}")
        return 1
    
    if args.ebm_only:
        print("\n✅ EBM explanation complete (skipping LLM)")
        return 0
    
    # Step 2: LLM Explanation
    print("\n" + "="*60)
    print("🤖 LLM PATIENT EXPLANATION")
    print("="*60)
    
    try:
        llm_service = LLMService(config)
        
        if not llm_service.check_ollama_available():
            print(f"❌ Ollama not available. Please run:")
            print(f"   ollama serve")
            print(f"   ollama pull {args.model}")
            return 1
        
        print("\n⏳ Generating patient-friendly explanation...")
        patient_explanation = llm_service.generate_explanation(ebm_result)
        
        print("\n📝 EXPLANATION FOR PATIENT:")
        print("-" * 40)
        print(patient_explanation.summary)
        print("-" * 40)
        
        if patient_explanation.key_points:
            print("\n🔑 Key Points:")
            for point in patient_explanation.key_points:
                print(f"  • {point}")
        
        if patient_explanation.recommendations:
            print("\n💡 Recommendations:")
            for rec in patient_explanation.recommendations:
                print(f"  • {rec}")
        
        print(f"\n⚠️  {patient_explanation.disclaimer}")
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        print(f"❌ LLM explanation error: {e}")
        return 1
    
    print("\n✅ Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
UMLS Concept Mapper using scispacy

This module provides UMLS concept mapping functionality using scispacy's 
EntityLinker component. It can be used standalone or integrated with the
clinical NLP pipeline.

Usage:
    from umls_mapper import UMLSMapper
    
    mapper = UMLSMapper()
    results = mapper.map_entity("diabetes mellitus")
    # Returns: {'cui': 'C0011849', 'name': 'Diabetes Mellitus', 'score': 0.95, 'aliases': [...]}
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import scispacy
try:
    import spacy
    import scispacy
    from scispacy.linking import EntityLinker
    SCISPACY_AVAILABLE = True
except ImportError:
    SCISPACY_AVAILABLE = False
    logger.warning("scispacy not installed. Install with: pip install scispacy")


@dataclass
class UMLSConcept:
    """Represents a UMLS concept."""
    cui: str
    name: str
    score: float
    aliases: List[str]
    definition: Optional[str] = None
    semantic_types: List[str] = None
    
    def __post_init__(self):
        if self.semantic_types is None:
            self.semantic_types = []


class UMLSMapper:
    """
    UMLS Concept Mapper using scispacy EntityLinker.
    
    This class provides methods to map clinical text to UMLS concepts.
    It uses scispacy's en_core_sci_sm model with the UMLS entity linker.
    """
    
    def __init__(self, linker_name: str = "umls", threshold: float = 0.7):
        """
        Initialize the UMLS mapper.
        
        Args:
            linker_name: Name of the knowledge base to use ('umls', 'mesh', 'rxnorm', 'go', 'hpo')
            threshold: Minimum similarity threshold for concept matching (0.0 - 1.0)
        """
        if not SCISPACY_AVAILABLE:
            raise ImportError(
                "scispacy is required for UMLS mapping. "
                "Install with: pip install scispacy\n"
                "And download model: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            )
        
        self.linker_name = linker_name
        self.threshold = threshold
        self.nlp = None
        self.linker = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the spaCy pipeline with entity linker."""
        logger.info(f"Loading scispacy model with {self.linker_name} linker...")
        
        try:
            # Load the scientific/medical spaCy model
            self.nlp = spacy.load("en_core_sci_sm")
            
            # Add the entity linker
            # This will download the UMLS knowledge base on first use (~1GB)
            self.nlp.add_pipe(
                "scispacy_linker", 
                config={
                    "resolve_abbreviations": True,
                    "linker_name": self.linker_name,
                    "threshold": self.threshold
                }
            )
            
            self.linker = self.nlp.get_pipe("scispacy_linker")
            
            logger.info("UMLS mapper initialized successfully")
            
        except OSError as e:
            if "en_core_sci_sm" in str(e):
                raise ImportError(
                    "en_core_sci_sm model not found. Install with:\n"
                    "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
                )
            raise
    
    def map_entity(self, text: str) -> Optional[UMLSConcept]:
        """
        Map a single entity text to UMLS concept.
        
        Args:
            text: The entity text to map (e.g., "diabetes mellitus")
            
        Returns:
            UMLSConcept if found, None otherwise
        """
        if not self.nlp:
            return None
        
        doc = self.nlp(text)
        
        # Get entities from the document
        for ent in doc.ents:
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                # Get the top matching concept
                cui, score = ent._.kb_ents[0]
                
                # Get concept details from the knowledge base
                concept_info = self.linker.kb.cui_to_entity.get(cui)
                
                if concept_info:
                    return UMLSConcept(
                        cui=cui,
                        name=concept_info.canonical_name,
                        score=score,
                        aliases=list(concept_info.aliases)[:5],  # Limit aliases
                        definition=concept_info.definition,
                        semantic_types=list(concept_info.types)
                    )
        
        # If no entity was detected, try to match the whole text
        # This is useful for single terms
        if len(doc.ents) == 0 and len(text.split()) <= 5:
            # Create a fake span and try linking
            try:
                from spacy.tokens import Span
                span = doc[:]
                # Manual linking attempt not supported, return None
                pass
            except:
                pass
        
        return None
    
    def map_entities_batch(self, texts: List[str]) -> List[Optional[UMLSConcept]]:
        """
        Map multiple entity texts to UMLS concepts.
        
        Args:
            texts: List of entity texts to map
            
        Returns:
            List of UMLSConcept objects (or None for unmapped entities)
        """
        results = []
        
        # Process in batches for efficiency
        for doc in self.nlp.pipe(texts, batch_size=50):
            found = False
            for ent in doc.ents:
                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    cui, score = ent._.kb_ents[0]
                    concept_info = self.linker.kb.cui_to_entity.get(cui)
                    
                    if concept_info:
                        results.append(UMLSConcept(
                            cui=cui,
                            name=concept_info.canonical_name,
                            score=score,
                            aliases=list(concept_info.aliases)[:5],
                            definition=concept_info.definition,
                            semantic_types=list(concept_info.types)
                        ))
                        found = True
                        break
            
            if not found:
                results.append(None)
        
        return results
    
    def process_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Process a full document and extract all entities with UMLS mappings.
        
        Args:
            text: Full document text
            
        Returns:
            List of dictionaries containing entity info and UMLS mappings
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        results = []
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'umls_cui': None,
                'umls_name': None,
                'umls_score': 0.0,
                'umls_types': [],
                'umls_definition': None
            }
            
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                cui, score = ent._.kb_ents[0]
                concept_info = self.linker.kb.cui_to_entity.get(cui)
                
                if concept_info:
                    entity_info['umls_cui'] = cui
                    entity_info['umls_name'] = concept_info.canonical_name
                    entity_info['umls_score'] = score
                    entity_info['umls_types'] = list(concept_info.types)
                    entity_info['umls_definition'] = concept_info.definition
            
            results.append(entity_info)
        
        return results


def add_umls_to_extracted_entities(
    entities_csv_path: str,
    output_path: Optional[str] = None,
    text_column: str = 'entity_text',
    batch_size: int = 100
) -> None:
    """
    Add UMLS mappings to an existing CSV of extracted entities.
    
    Args:
        entities_csv_path: Path to CSV with extracted entities
        output_path: Output path for enhanced CSV (default: overwrites input)
        text_column: Name of column containing entity text
        batch_size: Number of entities to process at once
    """
    import pandas as pd
    
    logger.info(f"Loading entities from {entities_csv_path}")
    df = pd.read_csv(entities_csv_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV")
    
    # Initialize mapper
    mapper = UMLSMapper()
    
    # Get unique entities to map (more efficient)
    unique_entities = df[text_column].unique().tolist()
    logger.info(f"Mapping {len(unique_entities)} unique entities to UMLS...")
    
    # Map in batches
    entity_to_umls = {}
    for i in range(0, len(unique_entities), batch_size):
        batch = unique_entities[i:i + batch_size]
        results = mapper.map_entities_batch(batch)
        
        for entity, umls in zip(batch, results):
            entity_to_umls[entity] = umls
        
        if (i + batch_size) % 500 == 0:
            logger.info(f"Processed {i + batch_size}/{len(unique_entities)} entities")
    
    # Add UMLS columns
    df['umls_cui_mapped'] = df[text_column].apply(
        lambda x: entity_to_umls.get(x).cui if entity_to_umls.get(x) else None
    )
    df['umls_name_mapped'] = df[text_column].apply(
        lambda x: entity_to_umls.get(x).name if entity_to_umls.get(x) else None
    )
    df['umls_score'] = df[text_column].apply(
        lambda x: entity_to_umls.get(x).score if entity_to_umls.get(x) else 0.0
    )
    
    # Save
    output_path = output_path or entities_csv_path
    df.to_csv(output_path, index=False)
    
    # Summary
    mapped_count = df['umls_cui_mapped'].notna().sum()
    logger.info(f"UMLS mapping complete: {mapped_count}/{len(df)} entities mapped ({100*mapped_count/len(df):.1f}%)")
    logger.info(f"Results saved to {output_path}")


def main():
    """Demo and test the UMLS mapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UMLS Concept Mapper')
    parser.add_argument('--test', action='store_true', help='Run demo test')
    parser.add_argument('--map-csv', type=str, help='Path to CSV file with entities to map')
    parser.add_argument('--output', type=str, help='Output path for mapped CSV')
    parser.add_argument('--text-column', type=str, default='entity_text', 
                        help='Column name containing entity text')
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing UMLS Mapper...")
        print("=" * 60)
        
        mapper = UMLSMapper()
        
        test_entities = [
            "diabetes mellitus",
            "hypertension",
            "myocardial infarction",
            "pneumonia",
            "aspirin",
            "metformin",
            "chest x-ray",
            "MRI",
            "shortness of breath",
            "edema"
        ]
        
        print("\nMapping clinical entities to UMLS concepts:")
        print("-" * 60)
        
        for entity in test_entities:
            result = mapper.map_entity(entity)
            if result:
                print(f"  '{entity}':")
                print(f"    CUI: {result.cui}")
                print(f"    Name: {result.name}")
                print(f"    Score: {result.score:.3f}")
                print(f"    Types: {', '.join(result.semantic_types[:3])}")
            else:
                print(f"  '{entity}': No UMLS match found")
            print()
    
    elif args.map_csv:
        add_umls_to_extracted_entities(
            args.map_csv,
            output_path=args.output,
            text_column=args.text_column
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

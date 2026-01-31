from dataclasses import dataclass
from typing import Optional

@dataclass
class KBEntity:
    """
    Representation of a knowledge base entity.

    Fields:
        uri: Entity URI (e.g., DBpedia resource URI)
        label: Primary label/title of the entity
        score: Confidence score from entity linker (0-1)
        start: Character start position in source text (optional)
        end: Character end position in source text (optional)
        description: Entity description text (optional)
    """
    uri: str
    label: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None
    description: Optional[str] = None


@dataclass
class KBCandidate:
    """
    A candidate expansion term derived from a KB entity.

    Fields:
        term: The actual term string (could be label, alias, or description term)
        source_entity: The entity this term came from
        term_type: Type of term ("label", "alias", "description", "related")
        confidence: Combined confidence score
    """
    term: str
    source_entity: KBEntity
    term_type: str  # "label", "alias", "description", "related"
    confidence: float

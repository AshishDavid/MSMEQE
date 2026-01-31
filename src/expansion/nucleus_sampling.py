
import numpy as np
import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class ExpansionCandidate:
    """Simple container for candidate terms."""
    def __init__(self, term: str, value: float):
        self.term = term
        self.value = value
        self.count = 1  # Default count

def apply_nucleus_sampling(candidates: List[Any], p: float = 0.9, max_terms: int = 1000) -> List[Any]:
    """
    Select terms using Top-P (Nucleus) Sampling based on their predicted Value.
    
    Args:
        candidates: List of objects with a .value attribute (predicted importance).
        p: Cumulative probability threshold (0.0 to 1.0).
        max_terms: Hard safety limit on number of terms.
        
    Returns:
        List of selected candidates.
    """
    # 1. Sort by Value (Descending)
    sorted_candidates = sorted(candidates, key=lambda x: x.value, reverse=True)
    
    # 2. Softmax (or just normalize positive values) to get probabilities
    # We assume 'value' is a raw logit or score. We need to convert to probability mass.
    # Simple approach: Positivity + Normalization
    values = np.array([c.value for c in sorted_candidates])
    
    # Handle negative values if model predicts them (shift to positive range for prob calc)
    if len(values) == 0:
        return []

    # Strategy: Temperature scaled softmax to determine "mass"
    # Or simple linear normalization of top-k if values are already probabilities
    # Assuming values are raw scores from a regressor (e.g., 0.1 to 0.9 range)
    
    # Robust normalization: Clip at 0
    values = np.maximum(values, 0)
    total_mass = np.sum(values)
    
    if total_mass == 0:
        return sorted_candidates[:5] # Fallback
        
    probs = values / total_mass
    
    # 3. Cumulative Sum
    cumulative_probs = np.cumsum(probs)
    
    # 4. Cutoff
    # Find index where cumulative sum >= p
    cutoff_idx = np.searchsorted(cumulative_probs, p)
    
    # Include the boundary item
    cutoff_idx = min(cutoff_idx, len(candidates) - 1)
    
    # Apply safety limit
    final_count = min(cutoff_idx + 1, max_terms)
    
    return sorted_candidates[:final_count]

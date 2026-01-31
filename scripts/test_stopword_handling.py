
import sys
import os
sys.path.append(os.getcwd())
import logging
from unittest.mock import MagicMock
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor

def main():
    # Mock encoder
    mock_encoder = MagicMock()
    mock_encoder.get_dim.return_value = 768
    # Mock encode to return a numpy array of shape (n_docs, 768)
    def side_effect(text_list):
        return np.zeros((len(text_list), 768))
    mock_encoder.encode.side_effect = side_effect

    print("Initializing Lucene...")
    from src.utils.lucene_utils import initialize_lucene
    if not initialize_lucene("lucene_jars"):
        print("Failed to initialize Lucene")
        return

    print("Initializing Extractor...")
    # Initialize (adjust index path if needed)
    extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index",
        encoder=mock_encoder
    )

    # Test cases
    test_queries = [
        "what is why", 
        "why is the sky blue",
        "interactive query expansion"
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        
        # Test 1: Tokenize and Stem (Analyzer behavior)
        # This tells us if Lucene's EnglishAnalyzer considers them stopwords
        tokens = extractor._tokenize_and_stem(q)
        print(f"Lucene Analyzer Tokens: {tokens}")

        # Test 2: Check standard stopwords set
        print("In self.stopwords?")
        for word in q.split():
            is_stop = word.lower() in extractor.stopwords
            print(f"  '{word}': {is_stop}")

        # Test 3: Query Stats behavior
        # We can't see inside easily without prints, but we can infer from logs if debug is on,
        # or just rely on our code reading. 
        # But running it ensures no crashes and gives us baseline values.
        stats = extractor.compute_query_stats(q)
        print("Computed Stats:", stats)

if __name__ == "__main__":
    main()

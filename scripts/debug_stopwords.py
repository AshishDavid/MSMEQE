
import sys
import os
import joblib
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.utils.lucene_utils import initialize_lucene
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder

# Mock Encoder to avoid loading model
class MockEncoder:
    def encode(self, texts):
        return []

def main():
    print("Initializing Lucene...")
    initialize_lucene(lucene_path="lucene_jars/")
    
    print("\n--- Initializing Pipeline with Strict Stopwords ---")
    # Initialize extractor which should now use the CUSTOM analyzer
    extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index", 
        encoder=MockEncoder() # Avoid loading heavy model
    )
    
    print("\n--- Testing Pipeline Analyzer ---")
    analyzer = extractor.analyzer
    
    test_text = "what is the reason for this why"
    print(f"Input: '{test_text}'")
    
    token_stream = analyzer.tokenStream("contents", test_text)
    
    # Need to manually get attributes from the stream, traversing via JNI
    # Ideally reuse helper but we will just check if "what" slips through
    from jnius import autoclass
    CharTermAttribute = autoclass("org.apache.lucene.analysis.tokenattributes.CharTermAttribute")
    term_attr = token_stream.getAttribute(CharTermAttribute)
    
    token_stream.reset()
    tokens = []
    while token_stream.incrementToken():
        tokens.append(term_attr.toString())
    token_stream.end()
    token_stream.close()
    
    print(f"Tokens: {tokens}")
    
    if "what" in tokens:
        print("FAIL: 'what' is still present")
    else:
        print("SUCCESS: 'what' filtered out")
        
    if "why" in tokens:
        print("FAIL: 'why' is still present")
    else:
        print("SUCCESS: 'why' filtered out")

    print("\n--- Checking Python Stopword Set ---")
    if "why" in extractor.stopwords:
         print("SUCCESS: 'why' in Python set")
    else:
         print("FAIL: 'why' NOT in Python set")

if __name__ == "__main__":
    main()

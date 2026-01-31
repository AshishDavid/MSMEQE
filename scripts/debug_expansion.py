#!/usr/bin/env python3
import sys
import json
import joblib
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.utils.lucene_utils import initialize_lucene
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor

def show_expansion(qid, qtext):
    initialize_lucene("lucene_jars")
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load models
    value_model = joblib.load("models/value_model.pkl")
    weight_model = joblib.load("models/weight_model.pkl")
    budget_model = joblib.load("models/budget_model.pkl")
    
    # Initialize Extractor
    emb_extractor = EmbeddingCandidateExtractor(encoder=encoder, vocab_path="data/vocab_embeddings.pkl")
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index",
        encoder=encoder,
        kb_extractor=None,
        emb_extractor=emb_extractor
    )
    
    # Load KB candidates (from the same file used in reranking)
    kb_candidates = []
    with open("data/kb_candidates_all_v2.jsonl", 'r') as f:
        for line in f:
            data = json.loads(line)
            if str(data.get('qid')) == str(qid):
                kb_candidates = data.get('candidates', [])
                break

    msmeqe_model = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=8841823,
        lambda_interp=0.4
    )

    # Extract
    candidates = candidate_extractor.extract_all_candidates(qtext, query_id=qid, kb_override=kb_candidates)
    stats = candidate_extractor.compute_query_stats(qtext)
    pseudo_centroid = candidate_extractor.compute_pseudo_centroid(qtext)
    
    # Expand
    selected_terms, _ = msmeqe_model.expand(
        query_text=qtext,
        candidates=candidates,
        pseudo_doc_centroid=pseudo_centroid,
        query_stats=stats
    )
    
    print(f"\nQUERY ID: {qid}")
    print(f"ORIGINAL QUERY: {qtext}")
    print("-" * 30)
    print("SELECTED EXPANSION TERMS:")
    for t in selected_terms:
        print(f" - [{t.source.upper()}] {t.term} (Value: {t.value:.4f}, Count: {t.count})")

if __name__ == "__main__":
    # Example: DL 19 query with many potential terms
    show_expansion("915593", "what types of food can you cook sous vide")

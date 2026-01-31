# src/expansion/msmeqe_expansion.py

"""
msmeqe/expansion/msmeqe_expansion.py

Core MS-MEQE expansion model.

Implements the pipeline described in the methodology section:

  - Takes multi-source candidate terms (docs / KB / embeddings)
  - Computes value and weight feature vectors using FeatureExtractor
  - Predicts per-term value and weight using trained models
  - Predicts a query-specific expansion budget
  - Solves an unbounded knapsack to select term frequencies
  - Builds a magnitude-encoded enhanced query embedding

Dependencies:
  - msmeqe.reranking.semantic_encoder.SemanticEncoder
  - msmeqe.features.feature_extraction.FeatureExtractor
  - Regressor objects for value, weight, and budget (with .predict)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor, create_candidate_stats_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateTerm:
    """
    Representation of a single term-source candidate.

    Fields:
      - term: lexical term (e.g., "neural", "insulin resistance")
      - source: one of {"docs", "kb", "emb"} (or compatible)

      - rm3_score: RM3 term score from pseudo-relevant docs
      - tf_pseudo: normalized term frequency in pseudo-relevant docs
      - coverage_pseudo: fraction of pseudo docs that contain this term (0–1)

      - df: document frequency in the full collection
      - cf: collection frequency (optional; can be 0)

      - native_rank: rank within its own source (1 = best)
      - native_score: native score within its own source
                      (e.g., entity-linking confidence, cosine similarity)
    """
    term: str
    source: str

    rm3_score: float = 0.0
    tf_pseudo: float = 0.0
    coverage_pseudo: float = 0.0

    df: int = 0
    cf: int = 0

    native_rank: int = 0
    native_score: float = 0.0


@dataclass
class SelectedTerm:
    """
    Output of MS-MEQE after knapsack:

      - term: lexical term
      - source: which source it came from
      - value: predicted value v_{t,i}
      - weight: predicted weight w_{t,i}
      - count: frequency c_k chosen by unbounded knapsack
    """
    term: str
    source: str
    value: float
    weight: float
    count: int


# ---------------------------------------------------------------------------
# Core MS-MEQE model
# ---------------------------------------------------------------------------

class MSMEQEExpansionModel:
    """
    Core MS-MEQE expansion model.

    Assumes:
      - A Sentence-BERT encoder (SemanticEncoder) for embeddings
      - FeatureExtractor for computing features
      - Pretrained regressors for value, weight, and budget
      - Candidates passed in as CandidateTerm instances
      - Pseudo-relevant document centroid embedding is provided (or None)

    Typical usage:

        from msmeqe.reranking.semantic_encoder import SemanticEncoder
        from msmeqe.features.feature_extraction import FeatureExtractor
        from msmeqe.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm

        encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        feature_extractor = FeatureExtractor(collection_size=N_docs)
        value_model = joblib.load("value_model.pkl")
        weight_model = joblib.load("weight_model.pkl")
        budget_model = joblib.load("budget_model.pkl")

        msmeqe = MSMEQEExpansionModel(
            encoder=encoder,
            feature_extractor=feature_extractor,
            value_model=value_model,
            weight_model=weight_model,
            budget_model=budget_model,
            lambda_interp=0.3,
        )

        selected_terms, q_star = msmeqe.expand(
            query_text=q,
            candidates=candidate_list,
            pseudo_doc_centroid=pseudo_centroid,   # np.ndarray or None
            query_stats=query_stats_dict,          # clarity, entropy, etc.
        )
    """

    def __init__(
            self,
            encoder: SemanticEncoder,
            value_model,
            weight_model,
            budget_model,
            collection_size: int,
            lambda_interp: float = 0.3,
            min_budget: int = 20,
            max_budget: int = 80,
            budget_step: int = 5,
    ) -> None:
        """
        Args:
            encoder: SemanticEncoder instance (Sentence-BERT wrapper)
            feature_extractor: FeatureExtractor instance
            value_model: regressor with .predict(X) → values
            weight_model: regressor with .predict(X) → weights
            budget_model: regressor with .predict(X_query_features) → budget
            lambda_interp: interpolation parameter λ in final query embedding
            min_budget, max_budget, budget_step: budget range & discretization
        """
        self.encoder = encoder
        self.feature_extractor = FeatureExtractor(collection_size=collection_size)
        self.value_model = value_model
        self.weight_model = weight_model
        self.budget_model = budget_model

        self.lambda_interp = float(lambda_interp)

        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.budget_step = int(budget_step)

        if self.min_budget <= 0 or self.max_budget <= 0:
            raise ValueError("Budgets must be positive")
        if self.max_budget < self.min_budget:
            raise ValueError("max_budget must be >= min_budget")
        if self.budget_step <= 0:
            raise ValueError("budget_step must be positive")

        logger.info(
            "Initialized MSMEQEExpansionModel: λ=%.3f, budget=[%d,%d], step=%d",
            self.lambda_interp,
            self.min_budget,
            self.max_budget,
            self.budget_step,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
            self,
            query_text: str,
            candidates: List[CandidateTerm],
            pseudo_doc_centroid: Optional[np.ndarray],
            query_stats: Dict[str, float],
    ) -> Tuple[List[SelectedTerm], np.ndarray]:
        """
        Run full MS-MEQE for one query.

        Args:
            query_text: original query string
            candidates: list of CandidateTerm objects (multi-source)
            pseudo_doc_centroid: centroid embedding of pseudo-relevant docs
                                 (same dim as encoder embeddings) or None
            query_stats: dictionary with query-level stats for budget pred:
                - "clarity": float
                - "entropy": float
                - "avg_idf": float
                - "max_idf": float
                - "avg_bm25": float
                - "var_bm25": float
                - "q_len": int
                - "q_type": "navigational" | "informational" | "transactional"

        Returns:
            selected_terms: list of SelectedTerm with counts (frequencies)
            q_star: enhanced query embedding (np.ndarray)
        """
        if not candidates:
            logger.warning(
                "MSMEQE.expand called with no candidates; returning original query embedding."
            )
            q_emb = self.encoder.encode([query_text])[0]
            return [], q_emb

        # Encode query
        logger.debug("MSMEQE: encoding query")
        q_emb = self.encoder.encode([query_text])[0]  # (d,)

        # Encode candidate terms
        logger.debug("MSMEQE: encoding %d candidate terms", len(candidates))
        term_strings = [c.term for c in candidates]
        term_embs = self.encoder.encode(term_strings)  # (m, d)

        # Build features using FeatureExtractor
        logger.debug("MSMEQE: extracting value features")
        X_val = self._build_value_features(
            query_text=query_text,
            query_embedding=q_emb,
            pseudo_centroid=pseudo_doc_centroid,
            candidates=candidates,
            term_embeddings=term_embs,
        )

        logger.debug("MSMEQE: extracting weight features")
        X_wt = self._build_weight_features(
            query_text=query_text,
            query_embedding=q_emb,
            candidates=candidates,
            term_embeddings=term_embs,
        )

        # Predict values and weights
        logger.debug("MSMEQE: predicting values and weights")
        v_pred = self._predict_values(X_val, candidates)  # Pass candidates for bias
        w_pred = self._predict_weights(X_wt, candidates)  # (m,)
        w_pred = np.maximum(w_pred, 1e-6)  # ensure non-negative / non-zero

        # Predict budget
        logger.debug("MSMEQE: predicting budget")
        W = self._predict_budget(query_stats)

        # Solve Selection Problem (Polymorphic)
        logger.debug("MSMEQE: solving selection problem (m=%d, W=%d)", len(candidates), W)
        
        counts = self._solve_selection(
            values=v_pred, 
            weights=w_pred, 
            budget=float(W)
        )

        # Collect selected terms
        selected_terms: List[SelectedTerm] = []
        for cand, v, w, c in zip(candidates, v_pred, w_pred, counts):
            if c <= 0:
                continue
            selected_terms.append(
                SelectedTerm(
                    term=cand.term,
                    source=cand.source,
                    value=float(v),
                    weight=float(w),
                    count=int(c),
                )
            )

        logger.debug("MSMEQE: %d terms selected by optimization", len(selected_terms))

        # Build enhanced query embedding
        logger.debug("MSMEQE: constructing enhanced query embedding")
        q_star = self._build_enhanced_query_embedding(
            query_embedding=q_emb,
            term_embeddings=term_embs,
            values=v_pred,
            counts=counts,
            clarity=query_stats.get('clarity', 12.0),
        )

        return selected_terms, q_star

    def _solve_selection(self, values: np.ndarray, weights: np.ndarray, budget: float) -> np.ndarray:
        """
        Solve the selection problem using Knapsack Optimization (OR-Tools).
        Subclasses can override this to implement Greedy or other heuristics.
        Returns:
            counts (np.ndarray): Integer array of counts (0/1).
        """
        # Import here to avoid circular dependencies
        from src.utils.knapsack_solver import solve_knapsack_ortools
        
        selected_mask = solve_knapsack_ortools(
            values=values,
            weights=weights,
            budget=budget
        )
        return selected_mask.astype(np.int32)

    # ------------------------------------------------------------------
    # Feature extraction (using FeatureExtractor)
    # ------------------------------------------------------------------

    def _build_value_features(
            self,
            query_text: str,
            query_embedding: np.ndarray,
            pseudo_centroid: Optional[np.ndarray],
            candidates: List[CandidateTerm],
            term_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Build value feature matrix using FeatureExtractor.

        Returns:
            Feature matrix of shape (m, 18)
        """
        m = len(candidates)
        if m == 0:
            return np.zeros((0, 18), dtype=np.float32)

        # Prepare candidate tuples for batch extraction
        candidate_tuples = [
            (
                c.term,
                c.source,
                create_candidate_stats_dict(
                    rm3_score=c.rm3_score,
                    tf_pseudo=c.tf_pseudo,
                    coverage_pseudo=c.coverage_pseudo,
                    df=c.df,
                    cf=c.cf,
                    native_rank=c.native_rank,
                    native_score=c.native_score,
                )
            )
            for c in candidates
        ]

        # Use FeatureExtractor's batch method
        features = self.feature_extractor.extract_value_features_batch(
            candidates=candidate_tuples,
            query_text=query_text,
            query_embedding=query_embedding,
            term_embeddings=term_embeddings,
            pseudo_centroid=pseudo_centroid,
        )

        return features

    def _build_weight_features(
            self,
            query_text: str,
            query_embedding: np.ndarray,
            candidates: List[CandidateTerm],
            term_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Build weight feature matrix using FeatureExtractor.

        Returns:
            Feature matrix of shape (m, 20)
        """
        m = len(candidates)
        if m == 0:
            return np.zeros((0, 20), dtype=np.float32)

        # Prepare candidate tuples
        candidate_tuples = [
            (
                c.term,
                c.source,
                create_candidate_stats_dict(
                    rm3_score=c.rm3_score,
                    tf_pseudo=c.tf_pseudo,
                    coverage_pseudo=c.coverage_pseudo,
                    df=c.df,
                    cf=c.cf,
                    native_rank=c.native_rank,
                    native_score=c.native_score,
                )
            )
            for c in candidates
        ]

        # Use FeatureExtractor's batch method
        features = self.feature_extractor.extract_weight_features_batch(
            candidates=candidate_tuples,
            query_text=query_text,
            query_embedding=query_embedding,
            term_embeddings=term_embeddings,
        )

        return features

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------

    def _predict_values(self, X: np.ndarray, candidates: List[CandidateTerm] = None) -> np.ndarray:
        """Predict value v_{t,i} for each candidate using the value model."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        y = self.value_model.predict(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        # Add source-based bias if candidates are provided
        if candidates is not None:
            for i, cand in enumerate(candidates):
                # Small boost for high-confidence sources like KB or LLM 
                # to prevent they being ignored by a conservative model
                if cand.source == 'kb':
                    y[i] += 0.01
                elif cand.source == 'llm':
                    y[i] += 0.005
        
        return y

    def _predict_weights(self, X: np.ndarray, candidates: List[CandidateTerm] = None) -> np.ndarray:
        """Predict weight (risk) w_{t,i} for each candidate using the weight model."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        y = self.weight_model.predict(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        # Add hard penalties for generic/short terms if candidates are provided
        if candidates is not None:
            for i, cand in enumerate(candidates):
                term = cand.term.lower()
                # Heavy penalty for very short terms
                if len(term) < 3:
                    y[i] += 0.5
                # Penalty for common/generic terms not in stopword list but still risky
                if term in ['number', 'example', 'type', 'using', 'means', 'defined', 'includes', 'common', 'also']:
                    y[i] += 0.2
        
        return y

    def _predict_budget(self, query_stats: Dict[str, float]) -> int:
        """
        Predict query-specific expansion budget W from query-level features.

        Expected keys in query_stats:
            - clarity
            - entropy
            - avg_idf
            - max_idf
            - avg_bm25
            - var_bm25
            - q_len
            - q_type  ("navigational" / "informational" / "transactional")

        Returns:
            Integer budget in [min_budget, max_budget] snapped to nearest budget_step.
        """
        # Extract query features using FeatureExtractor
        x_vec = self.feature_extractor.extract_query_features(
            query_text="",  # Not used in query feature extraction
            query_stats=query_stats,
        )

        if self.budget_model is None:
            # If no model provided (e.g. during training or fixed budget), use min_budget
            # (which should be equal to max_budget in fixed budget mode)
            budget_pred = float(self.min_budget)
        else:
            x_vec = x_vec.reshape(1, -1)
            budget_pred = float(self.budget_model.predict(x_vec)[0])

        # Clip and snap to nearest step
        clipped = max(self.min_budget, min(self.max_budget, budget_pred))
        snapped = self.budget_step * int(round(clipped / self.budget_step))

        logger.debug(
            "Budget prediction: raw=%.2f, clipped=%.2f, snapped=%d",
            budget_pred,
            clipped,
            snapped,
        )
        return snapped

    # ------------------------------------------------------------------
    # Unbounded knapsack
    # ------------------------------------------------------------------

    def _solve_greedy_selection(
            self,
            values: np.ndarray,
            weights: np.ndarray,
            term_embeddings: np.ndarray,
            budget: int,
    ) -> np.ndarray:
        """
        Greedy selection with Marginal Novelty and Soft Saturation.
        
        Algorithm:
        1. While budget > 0:
        2.   For each candidate k:
        3.     Calculate marginal value: v'_k = v_k * (1 - max_sim(t_k, S)) * log(1 + current_count_k + 1)/log(1 + current_count_k)
        4.     Pick k with highest v'_k / w_k
        5.   If best v'_k / w_k > 0 and w_k <= budget:
        6.     Add t_k, budget -= w_k
        7.   Else: break
        """
        m = len(values)
        if m == 0 or budget <= 0:
            return np.zeros((m,), dtype=np.int32)

        counts = np.zeros(m, dtype=np.int32)
        remaining_budget = float(budget)
        
        # Precompute similarities if possible or do it on the fly
        # For simplicity and small m (~100), on the fly is fine.
        
        while remaining_budget > 0:
            best_idx = -1
            best_marginal_yield = -1.0
            
            for i in range(m):
                if weights[i] > remaining_budget or counts[i] >= 3: # Hard limit of 3
                    continue
                
                # Soft Saturation: gain = log(1 + c + 1) - log(1 + c) = log((c+2)/(c+1))
                # normalized by log(2) so that first item has full value
                saturation_gain = np.log2((counts[i] + 2) / (counts[i] + 1))
                
                # Marginal Novelty
                novelty_factor = 1.0
                if np.any(counts > 0):
                    selected_indices = np.where(counts > 0)[0]
                    # term_embeddings is (m, d)
                    # max similarity of candidate i with all currently selected terms
                    similarities = np.dot(term_embeddings[selected_indices], term_embeddings[i])
                    max_sim = np.max(similarities)
                    # Penalize novelty: if highly similar (e.g. 0.9), novelty is low (0.1)
                    novelty_factor = 1.0 - max(0, max_sim)
                
                marginal_value = values[i] * novelty_factor * saturation_gain
                yield_score = marginal_value / (weights[i] + 1e-6)
                
                if yield_score > best_marginal_yield:
                    best_marginal_yield = yield_score
                    best_idx = i
            
            if best_idx != -1 and best_marginal_yield > 0:
                counts[best_idx] += 1
                remaining_budget -= weights[best_idx]
            else:
                break
                
        return counts

    # ------------------------------------------------------------------
    # Query embedding construction
    # ------------------------------------------------------------------

    def _build_enhanced_query_embedding(
            self,
            query_embedding: np.ndarray,
            term_embeddings: np.ndarray,
            values: np.ndarray,
            counts: np.ndarray,
            clarity: float,
    ) -> np.ndarray:
        """
        Build magnitude-encoded enhanced query embedding:

            e_scaled(t_k) = c_k * v_k * e(t_k)

            q* = (1 - λ) e(q) + λ * (sum e_scaled / sum (c_k * v_k))

        If no terms are selected (all counts=0), returns original e(q).
        """
        if term_embeddings.shape[0] == 0 or np.all(counts <= 0):
            return query_embedding

        scaled_vectors = []
        scaled_weights = []

        for emb, v, c in zip(term_embeddings, values, counts):
            if c <= 0 or v <= 0:
                continue
            weight = float(c) * float(v)
            scaled_vectors.append(weight * emb)
            scaled_weights.append(weight)

        if not scaled_vectors:
            return query_embedding

        scaled_vectors = np.stack(scaled_vectors, axis=0)
        scaled_weights = np.array(scaled_weights, dtype=np.float32)

        # Dynamic Lambda: trust clear queries more
        # Base clarity center is ~12. Range 8 to 20.
        # If base_lambda=0.0, we keep it 0.0.
        if self.lambda_interp > 0:
            dyn_lambda = min(0.6, max(0.05, self.lambda_interp * (clarity / 12.0)))
        else:
            dyn_lambda = 0.0

        weighted_avg = np.sum(scaled_vectors, axis=0) / (np.sum(scaled_weights) + 1e-12)
        q_star = (1.0 - dyn_lambda) * query_embedding + dyn_lambda * weighted_avg
        return q_star
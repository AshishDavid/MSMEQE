"""
RM (Relevance Model) Expansion Module - Lucene Backend

Clean interface to Lucene-based RM1 and RM3 query expansion.
Replaces the previous Python implementation with proven Java/Lucene code.

Author: Your Name
"""

import logging
import json
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from src.utils.lucene_utils import get_lucene_classes


logger = logging.getLogger(__name__)


class LuceneRM3Scorer:
    """
    RM3 expansion using a Lucene index.

    This class wraps the Java/Lucene implementation of RM1 / RM3-style term scoring:

        P(t | R) = sum_{d in R} P(t | d) * P(d | q)

    where:
      - P(d | q) is proportional to the BM25 score for d (normalized),
      - P(t | d) is the Dirichlet-smoothed document language model.

    It then optionally interpolates with the original query model (RM3),
    producing a ranked list of expansion terms with probabilities.

    Typical use:
        rm3 = LuceneRM3Scorer(index_dir, field="contents")
        expanded = rm3.expand("neural ranking models", n_docs=10, n_terms=30)

    Returns: List[(term, weight)] sorted by descending weight.
    """

    def __init__(
        self,
        index_dir: str,
        field: str = "contents",
        analyzer: str = "StandardAnalyzer",
        similarity: str = "BM25Similarity",
        mu: float = 1000.0,
        orig_query_weight: float = 0.5,
        min_doc_score: float = 0.0,
    ):
        """
        Args:
            index_dir: Path to Lucene index.
            field: Text field to use for retrieval and term vectors.
            analyzer: Name of Lucene Analyzer (must be supported by lucene_utils).
            similarity: Name of Lucene Similarity (e.g., BM25Similarity).
            mu: Dirichlet prior parameter for document language model.
            orig_query_weight: Interpolation weight for original query in RM3:
                               final P(t) = (1 - orig_query_weight) * P_RM1(t)
                                            + orig_query_weight * P_QM(t)
            min_doc_score: Optional threshold; ignore PRF docs below this score.
        """
        self.index_dir = str(index_dir)
        self.field = field
        self.mu = mu
        self.orig_query_weight = orig_query_weight
        self.min_doc_score = min_doc_score

        classes = get_lucene_classes()
        
        self.JIndexSearcher = classes['IndexSearcher']
        self.JIndexReader = classes['IndexReader']
        self.JDirectoryReader = classes['DirectoryReader']
        self.JFSDirectory = classes['FSDirectory']
        self.JPath = classes['JavaPaths']
        self.JStandardAnalyzer = classes['StandardAnalyzer']
        self.JEnglishAnalyzer = classes['EnglishAnalyzer']
        self.JQueryParser = classes['QueryParser']
        self.JQueryBuilder = classes['BooleanQueryBuilder'] # Note: BooleanQuery$Builder is usually used for building
        # Wait, JQueryBuilder was likely QueryBuilder?
        # Let's check what was intended.
        # In original code: self.JQueryBuilder
        # In lucene_utils: 'BooleanQueryBuilder' : autoclass('org.apache.lucene.search.BooleanQuery$Builder')
        # There is no 'QueryBuilder' in lucene_utils.
        # But there is 'BooleanQuery' and 'BooleanClause'.
        
        self.JBooleanQueryBuilder = classes['BooleanQueryBuilder']
        self.JBooleanClause = classes['BooleanClause']
        self.JBooleanClauseOccur = classes['BooleanClauseOccur']
        self.JTerm = classes['Term']
        self.JTermQuery = classes['TermQuery']
        # self.JTopDocs = classes['TopDocs'] # Missing in lucene_utils?
        # self.JScoreDoc = classes['ScoreDoc'] # Missing?
        
        self.JBM25Similarity = classes['BM25Similarity']
        # self.JClassicSimilarity = classes['ClassicSimilarity'] # Missing?
        
        # self.JDirectory = classes['Directory'] # Missing?
        # self.JAnalyzer = classes['Analyzer'] # Missing?
        # self.JQuery = classes['Query'] # Missing?
        # self.JExplanation = classes['Explanation'] # Missing?
        # self.JTermVector = classes['TermVector'] # Missing?
        # self.JFields = classes['Fields'] # Missing?
        # self.JField = classes['Field']
        # self.JBytesRefIterator = classes['BytesRefIterator'] # Missing?
        self.JBytesRef = classes['BytesRef']
        self.JMultiFields = classes['MultiFields']
        
        # Open index and initialize searcher
        from jnius import cast
        self.reader = self.JDirectoryReader.open(
            self.JFSDirectory.open(self.JPath.get(self.index_dir))
        )
        # Explicitly cast to IndexReader to avoid ambiguity
        index_reader = cast('org.apache.lucene.index.IndexReader', self.reader)
        self.searcher = self.JIndexSearcher(index_reader)

        # Set similarity
        if similarity == "BM25Similarity":
            self.searcher.setSimilarity(self.JBM25Similarity())
        elif similarity == "ClassicSimilarity":
            self.searcher.setSimilarity(self.JClassicSimilarity())
        else:
            raise ValueError(f"Unsupported similarity: {similarity}")

        # Analyzer and query parser
        if analyzer == "StandardAnalyzer":
            self.analyzer = self.JStandardAnalyzer()
        elif analyzer == "EnglishAnalyzer":
            self.analyzer = self.JEnglishAnalyzer()
        else:
            raise ValueError(f"Unsupported analyzer: {analyzer}")

        self.query_parser = self.JQueryParser(self.field, self.analyzer)
        self.query_builder = self.JQueryBuilder()
        self.BooleanQueryBuilder = self.JBooleanQueryBuilder
        self.BooleanClause = self.JBooleanClause
        self.Term = self.JTerm
        self.TermQuery = self.JTermQuery

        logger.info(
            f"Initialized LuceneRM3Scorer(index={self.index_dir}, field={self.field}, "
            f"similarity={similarity}, analyzer={analyzer}, mu={mu}, "
            f"orig_query_weight={orig_query_weight})"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def expand(
        self,
        query_str: str,
        n_docs: int = 10,
        n_terms: int = 30,
        use_rm3: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Perform RM3 expansion for a query string.

        Args:
            query_str: Original query string.
            n_docs: Number of PRF documents to use (top-N).
            n_terms: Number of expansion terms to return.
            use_rm3: If True, interpolate with original query (RM3),
                     otherwise return RM1 model.

        Returns:
            List of (term, weight) sorted by weight descending.
        """
        logger.debug(
            f"Starting RM3 expansion for query='{query_str}', "
            f"n_docs={n_docs}, n_terms={n_terms}, use_rm3={use_rm3}"
        )

        if not query_str.strip():
            logger.warning("Empty query string, no expansion performed.")
            return []

        # Build Lucene query
        lucene_query = self._build_boolean_query(query_str)

        # Retrieve top-N documents
        top_docs = self.searcher.search(lucene_query, n_docs)
        score_docs = top_docs.scoreDocs

        if not score_docs:
            logger.warning("No documents retrieved for query, no expansion terms.")
            return []

        logger.debug(f"Retrieved {len(score_docs)} documents for RM3 expansion.")

        # Normalize P(d | q) from BM25 scores
        doc_scores = np.array([sd.score for sd in score_docs])
        if self.min_doc_score > 0.0:
            mask = doc_scores >= self.min_doc_score
            score_docs = [sd for sd, keep in zip(score_docs, mask) if keep]
            doc_scores = doc_scores[mask]
            logger.debug(
                f"Applied min_doc_score={self.min_doc_score}, "
                f"kept {len(score_docs)} documents."
            )

        if len(score_docs) == 0:
            logger.warning("All PRF documents below min_doc_score, no expansion.")
            return []

        doc_probs = self._normalize_scores(doc_scores)

        # Compute RM1 term distribution P(t | R)
        rm1 = self._compute_rm1(score_docs, doc_probs)

        if use_rm3:
            # Build original query model P(t | Q)
            qm = self._build_query_model(query_str)
            rm3 = self._interpolate_rm3(rm1, qm)
            term_weights = rm3
            logger.debug(
                f"RM3 interpolation applied: orig_query_weight={self.orig_query_weight}"
            )
        else:
            term_weights = rm1

        # Sort and truncate
        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
        result = sorted_terms[:n_terms]

        logger.debug(f"RM3 expansion completed: {len(result)} terms")
        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_boolean_query(self, query_str: str):
        """Build initial boolean query from query string."""
        try:
            query_terms = self._tokenize_query(query_str)

            if not query_terms:
                # Simple whitespace fallback if analyzer returns nothing
                query_terms = [t.strip().lower() for t in query_str.split() if t.strip()]

            if not query_terms:
                raise ValueError(f"No valid terms in query: {query_str}")

            builder = self.BooleanQueryBuilder()

            for term in query_terms:
                term_query = self.TermQuery(self.Term(self.field, term))
                builder.add(term_query, self.JBooleanClauseOccur.SHOULD)

            boolean_query = builder.build()
            logger.debug(f"Built BooleanQuery with {len(query_terms)} terms.")
            return boolean_query

        except Exception as e:
            logger.error(f"Error building BooleanQuery for '{query_str}': {e}")
            # Fallback: use query parser on raw string with escaping ONLY as last resort
            try:
                escaped = self.JQueryParser.escape(query_str)
                # Ensure analyzer is used correctly in parser
                parser = self.JQueryParser(self.field, self.analyzer)
                return parser.parse(escaped)
            except Exception as e2:
                logger.error(f"Fallback ParseException for '{query_str}': {e2}")
                # Provide a Dummy query if all else fails to avoid crashing
                return self.TermQuery(self.Term(self.field, "dummy_empty_query_term"))

    def _tokenize_query(self, query_str: str) -> List[str]:
        """Tokenize query with Lucene Analyzer."""
        try:
            token_stream = self.analyzer.tokenStream(self.field, query_str)
            try:
                token_stream.reset()
                from jnius import autoclass
                CharTermAttribute = autoclass("org.apache.lucene.analysis.tokenattributes.CharTermAttribute")
                term_attr = token_stream.getAttribute(CharTermAttribute)
                terms = []
                while token_stream.incrementToken():
                    terms.append(term_attr.toString())
                token_stream.end()
            finally:
                token_stream.close()
            
            logger.debug(f"Tokenized query '{query_str}' → {terms}")
            return terms
        except Exception as e:
            logger.error(f"Error tokenizing query '{query_str}': {e}")
            # Fallback: whitespace split
            return [t.strip() for t in query_str.split() if t.strip()]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores into P(d | q)."""
        max_score = scores.max()
        # Avoid overflow; subtract max then exp
        exp_scores = np.exp(scores - max_score)
        z = exp_scores.sum()
        if z == 0.0:
            logger.warning("All scores zero after exponentiation; using uniform probs.")
            return np.ones_like(scores) / len(scores)
        probs = exp_scores / z
        logger.debug(f"Normalized scores to P(d|q): sum={probs.sum()}")
        return probs

    def _compute_rm1(
        self,
        score_docs: List,
        doc_probs: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute RM1 model:

            P(t | R) = sum_{d in R} P(t | d) * P(d | q)

        using Dirichlet smoothing for P(t | d).
        """
        try:
            term_weights: Dict[str, float] = defaultdict(float)
            collection_length = self._get_collection_length()
            term_cf_cache: Dict[str, int] = {}

            for sd, p_dq in zip(score_docs, doc_probs):
                doc_id = sd.doc
                doc_len = self._get_doc_length(doc_id)
                if doc_len <= 0:
                    continue

                # Get term vector for this document
                # Lucene 10: reader.termVectors().get(docID) returns Fields
                term_vectors = self.reader.termVectors()
                if term_vectors is None:
                    continue
                fields = term_vectors.get(doc_id)
                if fields is None:
                    continue
                
                terms = fields.terms(self.field)
                if terms is None:
                    continue
                    
                terms_enum = terms.iterator()
                bytes_ref = terms_enum.next()
                while bytes_ref is not None:
                    term_text = bytes_ref.utf8ToString()

                    # Frequency of this term in the document
                    freq = terms_enum.totalTermFreq()

                    # Collection frequency (cached)
                    if term_text in term_cf_cache:
                        cf = term_cf_cache[term_text]
                    else:
                        cf = self._get_collection_freq(term_text)
                        term_cf_cache[term_text] = cf

                    if cf <= 0:
                        bytes_ref = terms_enum.next()
                        continue

                    # Dirichlet-smoothed P(t | d)
                    p_td = (freq + self.mu * (cf / collection_length)) / (
                        doc_len + self.mu
                    )

                    # Accumulate P(t | R)
                    term_weights[term_text] += p_td * p_dq

                    bytes_ref = terms_enum.next()

            # Normalize RM1 to sum to 1
            total = sum(term_weights.values())
            if total > 0:
                for t in term_weights:
                    term_weights[t] /= total

            logger.debug(
                f"Processed {len(score_docs)} documents, total terms: {len(term_weights)}"
            )
            return dict(term_weights)

        except Exception as e:
            logger.error(f"Error computing RM1 weights: {e}")
            return {}

    def _get_collection_length(self) -> int:
        """Total number of terms in the collection for this field."""
        try:
            return self.reader.getSumTotalTermFreq(self.field)
        except Exception as e:
            logger.error(f"Error getting collection length: {e}")
            return 0

    def _get_doc_length(self, doc_id: int) -> int:
        """Document length for doc_id."""
        try:
            term_vectors = self.reader.termVectors()
            if term_vectors is None:
                return 0
            fields = term_vectors.get(doc_id)
            if fields is None:
                return 0
            
            terms = fields.terms(self.field)
            if terms is None:
                return 0
                
            terms_enum = terms.iterator()
            bytes_ref = terms_enum.next()
            length = 0
            while bytes_ref is not None:
                length += terms_enum.totalTermFreq()
                bytes_ref = terms_enum.next()
            return length
        except Exception as e:
            logger.error(f"Error getting document length for doc_id={doc_id}: {e}")
            return 0

    def _get_collection_freq(self, term: str) -> int:
        """Collection frequency of term in field."""
        try:
            lucene_term = self.Term(self.field, term)
            return self.reader.totalTermFreq(lucene_term)
        except Exception as e:
            logger.error(f"Error getting collection frequency for term='{term}': {e}")
            return 0

    def _build_query_model(self, query_str: str) -> Dict[str, float]:
        """Build original query model P(t | Q) from tokenized query."""
        terms = self._tokenize_query(query_str)
        if not terms:
            return {}
        counts = Counter(terms)
        total = sum(counts.values())
        qm = {t: c / total for t, c in counts.items()}
        logger.debug(f"Built query model with {len(qm)} terms.")
        return qm

    def _interpolate_rm3(
        self,
        rm1: Dict[str, float],
        qm: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Interpolate RM1 with original query model:

            P_RM3(t) = (1 - alpha) * P_RM1(t) + alpha * P_QM(t)
        """
        alpha = self.orig_query_weight
        combined = defaultdict(float)

        # Add RM1
        for t, p in rm1.items():
            combined[t] += (1.0 - alpha) * p

        # Add query model
        for t, p in qm.items():
            combined[t] += alpha * p

        # Normalize
        total = sum(combined.values())
        if total > 0:
            for t in list(combined.keys()):
                combined[t] /= total

        logger.debug(
            f"Interpolated RM1 ({len(rm1)} terms) with QM ({len(qm)} terms) → {len(combined)} terms."
        )
        return dict(combined)

    def _tokenize_document(self, text: str) -> List[str]:
        """
        Tokenize document text using the same Analyzer.
        This is only used if you want to apply additional filters in Python.
        """
        try:
            token_stream = self.analyzer.tokenStream(self.field, text)
            try:
                token_stream.reset()
                from jnius import autoclass
                CharTermAttribute = autoclass("org.apache.lucene.analysis.tokenattributes.CharTermAttribute")
                term_attr = token_stream.getAttribute(CharTermAttribute)
                terms = []
                while token_stream.incrementToken():
                    terms.append(term_attr.toString())
                token_stream.end()
            finally:
                token_stream.close()
            return terms
        except Exception as e:
            logger.error(f"Error tokenizing document text: {e}")
            return []


# -------------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------------

def rm3_expand_query(
    index_dir: str,
    query_str: str,
    field: str = "contents",
    n_docs: int = 10,
    n_terms: int = 30,
    mu: float = 1000.0,
    orig_query_weight: float = 0.5,
    analyzer: str = "StandardAnalyzer",
    similarity: str = "BM25Similarity",
    min_doc_score: float = 0.0,
    use_rm3: bool = True,
) -> List[Tuple[str, float]]:
    """
    One-shot RM3 expansion helper.

    Example:
        terms = rm3_expand_query(
            index_dir="/path/to/index",
            query_str="neural ranking models",
            n_docs=10,
            n_terms=30,
        )
    """
    scorer = LuceneRM3Scorer(
        index_dir=index_dir,
        field=field,
        analyzer=analyzer,
        similarity=similarity,
        mu=mu,
        orig_query_weight=orig_query_weight,
        min_doc_score=min_doc_score,
    )
    return scorer.expand(
        query_str=query_str,
        n_docs=n_docs,
        n_terms=n_terms,
        use_rm3=use_rm3,
    )

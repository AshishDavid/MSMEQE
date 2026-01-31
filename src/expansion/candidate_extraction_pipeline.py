# src/expansion/candidate_extraction_pipeline.py

"""
Complete pipeline for extracting multi-source candidates with all required stats.

This module bridges:
  1. Source-specific extractors (RM3, KB, Embeddings)
  2. MS-MEQE expansion model (which needs CandidateTerm objects)

It handles:
  - Multi-source candidate extraction
  - Term statistics (DF, CF) from Lucene
  - Pseudo-document statistics (TF, coverage)
  - Query statistics for budget prediction
  - Pseudo-document centroid computation

Usage:
    from msmeqe.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
    from msmeqe.reranking.semantic_encoder import SemanticEncoder

    encoder = SemanticEncoder()
    extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index",
        encoder=encoder,
    )

    # Extract candidates
    candidates = extractor.extract_all_candidates(
        query_text="neural networks",
        query_id="q123",
    )

    # Compute query stats
    query_stats = extractor.compute_query_stats("neural networks")
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import re
import json
import numpy as np

from src.expansion.rm_expansion import LuceneRM3Scorer
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.msmeqe_expansion import CandidateTerm
from src.utils.lucene_utils import get_lucene_classes

logger = logging.getLogger(__name__)


class MultiSourceCandidateExtractor:
    """
    Extract candidates from all three sources with complete statistics.
    """

    def __init__(
            self,
            index_path: str,
            encoder: SemanticEncoder,
            kb_extractor: Optional[KBCandidateExtractor] = None,
            emb_extractor: Optional[EmbeddingCandidateExtractor] = None,
            n_docs_rm3: int = 30,
            n_kb: int = 30,
            n_emb: int = 30,
            n_pseudo_docs: int = 10,
            llm_candidates_path: Optional[str] = None,
    ):
        """
        Initialize multi-source extractor.
        """
        self.encoder = encoder
        self.index_path = index_path

        # Initialize RM3 scorer
        logger.info(f"Initializing RM3 scorer with index: {index_path}")
        self.rm3_scorer = LuceneRM3Scorer(
            index_dir=index_path,
            field="contents",
            mu=1000.0,
            orig_query_weight=0.5,
            analyzer="EnglishAnalyzer"
        )

        # Store extractors
        self.kb_extractor = kb_extractor
        self.emb_extractor = emb_extractor

        self.n_docs_rm3 = n_docs_rm3
        self.n_kb = n_kb
        self.n_emb = n_emb
        self.n_pseudo_docs = n_pseudo_docs

        # Initialize Lucene for term statistics
        self._init_lucene_stats()

        # Load LLM candidates if provided
        self.llm_cache = {}
        if llm_candidates_path:
            logger.info(f"Loading LLM candidates from {llm_candidates_path}")
            try:
                with open(llm_candidates_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        qid = data.get('qid') or str(data.get('id', ''))
                        if qid:
                            self.llm_cache[str(qid)] = data
                logger.info(f"Loaded {len(self.llm_cache)} LLM candidate records")
            except Exception as e:
                logger.warning(f"Failed to load LLM candidates: {e}")

        # Broadened stopword list
        self.stopwords = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was',
            'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
            'do', 'does', 'doing', 'done', 'am', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'shall', 'will', 'should', 'would', 'can', 'could', 'may', 'might', 'must',
            'also', 'each', 'many', 'much', 'some', 'very', 'too', 'than',
            'just', 'only', 'both', 'each', 'any', 'such', 'this', 'that',
            'these', 'those', 'same', 'different', 'another', 'other',
            'within', 'between', 'during', 'before', 'after', 'above',
            'below', 'around', 'about', 'across', 'through', 'toward',
            'along', 'among', 'behind', 'against', 'near', 'far',
            'common', 'usually', 'typically', 'often', 'frequently',
            'always', 'never', 'sometimes', 'nearly', 'almost',
            'includes', 'including', 'included', 'example', 'examples', 'like',
            'why', 'where', 'whose', 'whom',
        }

        logger.info(
            f"Initialized MultiSourceCandidateExtractor: "
            f"RM3={n_docs_rm3}, KB={n_kb}, Emb={n_emb}"
        )

    def _init_lucene_stats(self):
        """Initialize Lucene index for term statistics (DF, CF)."""
        try:
            classes = get_lucene_classes()

            self.DirectoryReader = classes['IndexReader']
            self.FSDirectory = classes['FSDirectory']
            self.JavaPaths = classes['JavaPaths']
            self.DirectoryReader = classes['DirectoryReader']
            self.IndexSearcher = classes['IndexSearcher']
            self.QueryParser = classes['QueryParser']
            self.EnglishAnalyzer = classes['EnglishAnalyzer']
            self.BooleanQueryBuilder = classes['BooleanQueryBuilder']
            self.BooleanClause = classes['BooleanClause']
            self.Term = classes['Term']
            self.TermQuery = classes['TermQuery']
            self.BytesRef = classes['BytesRef']

            directory = self.FSDirectory.open(self.JavaPaths.get(self.index_path))
            self.lucene_reader = self.DirectoryReader.open(directory)
            self.lucene_searcher = self.IndexSearcher(self.lucene_reader.getContext())
            self.analyzer = self.EnglishAnalyzer()
            self.field_name = "contents"

            # Get collection size
            self.collection_size = self.lucene_reader.numDocs()

            logger.info(
                f"Lucene index opened: {self.collection_size} documents, "
                f"field='{self.field_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Lucene: {e}")
            raise
            
        # Optimization: Cache for extract_all_candidates
        self._cache = {}

    def extract_all_candidates(
            self,
            query_text: str,
            query_id: Optional[str] = None,
            kb_override: Optional[List[Dict]] = None,
    ) -> List[CandidateTerm]:
        """
        Extract candidates from all sources with complete statistics.

        Optimized to pre-tokenize pseudo-documents once to avoid overhead during
        individual term statistics calculation.
        """
        all_candidates = []
        
        # Check cache (only if no override)
        cache_key = (query_text, query_id)
        if kb_override is None and cache_key in self._cache:
            # logger.debug(f"Cache hit for query: {query_text[:30]}...")
            return self._cache[cache_key]

        # 1. Retrieve pseudo-relevant documents once (shared across calculations)
        # Returns (list of text, list of tokenized lists)
        print("DEBUG: Calling _get_pseudo_relevant_docs_and_tokens", flush=True)
        pseudo_docs, pseudo_docs_tokens = self._get_pseudo_relevant_docs_and_tokens(
            query_text, self.n_pseudo_docs
        )
        print(f"DEBUG: Retrieved {len(pseudo_docs)} pseudo docs", flush=True)

        # Prepare query terms for exclusion (stemmed)
        query_terms = set(self._tokenize_and_stem(query_text))

        # === 1. DOCS SOURCE (RM3) ===
        logger.debug(f"Extracting RM3 candidates for: {query_text[:50]}")

        try:
            print(f"DEBUG: Extracting RM3 candidates", flush=True)
            rm3_terms = self.rm3_scorer.expand(
                query_str=query_text,
                n_docs=self.n_pseudo_docs,
                n_terms=self.n_docs_rm3,
                use_rm3=True,
            )

            for rank, (term, rm3_score) in enumerate(rm3_terms, start=1):
                term_l = term.strip().lower()
                if term_l in self.stopwords or term_l in query_terms:
                    continue
                
                df, cf = self._get_term_stats(term)
                
                # Filter out generic high-frequency terms (>3% of collection)
                if df > (self.collection_size * 0.03):
                    continue
                
                # Use pre-tokenized docs for efficiency
                tf_pseudo = self._compute_tf_pseudo_optimized(term, pseudo_docs_tokens)
                coverage = self._compute_coverage_optimized(term, pseudo_docs)

                all_candidates.append(CandidateTerm(
                    term=term,
                    source="docs",
                    rm3_score=rm3_score,
                    tf_pseudo=tf_pseudo,
                    coverage_pseudo=coverage,
                    df=df,
                    cf=cf,
                    native_rank=rank,
                    native_score=rm3_score,
                ))

            logger.debug(f"Extracted {len(rm3_terms)} RM3 candidates")

        except Exception as e:
            logger.warning(f"RM3 extraction failed: {e}")

        # === 2. KB SOURCE (WITH OVERRIDE SUPPORT) ===
        if kb_override is not None:
            # Use precomputed KB candidates
            print(f"DEBUG: Using {len(kb_override)} precomputed KB candidates", flush=True)
            logger.debug(f"Using {len(kb_override)} precomputed KB candidates")

            try:
                for rank, kb_cand_dict in enumerate(kb_override[:self.n_kb], start=1):
                    # Validation: Ensure required keys exist
                    term = kb_cand_dict.get('term')
                    if not term:
                        continue
                    
                    term_l = term.strip().lower()
                    if term_l in query_terms or term_l in self.stopwords:
                        continue
                        
                    confidence = kb_cand_dict.get('confidence', 1.0)
                    # Use provided rank if available, else enumeration
                    cand_rank = kb_cand_dict.get('rank', rank)

                    df, cf = self._get_term_stats(term)

                    if df > (self.collection_size * 0.03):
                        continue
                    
                    all_candidates.append(CandidateTerm(
                        term=term,
                        source="kb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=cand_rank,
                        native_score=confidence,
                    ))
            except Exception as e:
                logger.warning(f"Failed to process precomputed KB candidates: {e}")

        elif self.kb_extractor:
            # Extract KB candidates on-the-fly (original behavior)
            logger.debug(f"Extracting KB candidates on-the-fly")
            try:
                kb_candidates = self.kb_extractor.extract_candidates_with_metadata(
                    query_text=query_text,
                    query_id=query_id,
                )

                for rank, kb_cand in enumerate(kb_candidates[:self.n_kb], start=1):
                    term = kb_cand.term
                    term_l = term.strip().lower()
                    if term_l in self.stopwords or term_l in query_terms:
                        continue
                        
                    df, cf = self._get_term_stats(term)
                    
                    if df > (self.collection_size * 0.03):
                        continue
                        
                    all_candidates.append(CandidateTerm(
                        term=kb_cand.term,
                        source="kb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=rank,
                        native_score=kb_cand.confidence,
                    ))
                logger.debug(f"Extracted {len(kb_candidates)} KB candidates")
            except Exception as e:
                logger.warning(f"KB extraction failed: {e}")

        # === 3. EMBEDDING SOURCE ===
        if self.emb_extractor:
            print(f"DEBUG: Extracting embedding candidates", flush=True)
            logger.debug(f"Extracting embedding candidates for: {query_text[:50]}")
            try:
                emb_candidates = self.emb_extractor.extract_candidates(
                    query_text=query_text,
                    k=self.n_emb,
                )

                for rank, (term, cos_sim) in enumerate(emb_candidates, start=1):
                    term_l = term.strip().lower()
                    if term_l in self.stopwords or term_l in query_terms:
                        continue
                        
                    df, cf = self._get_term_stats(term)
                    
                    if df > (self.collection_size * 0.03):
                        continue
                        
                    all_candidates.append(CandidateTerm(
                        term=term,
                        source="emb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=rank,
                        native_score=cos_sim,
                    ))
                logger.debug(f"Extracted {len(emb_candidates)} embedding candidates")
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")

        # === 4. LLM SOURCE ===
        if query_id and str(query_id) in self.llm_cache:
            # Extract LLM candidates
            # Logic: Entities/Synonyms get boost=5, Passage gets boost=1
            llm_data = self.llm_cache[str(query_id)].get('llm_data', {})
            
            # Helper to tokenize (simple whitespace + lower)
            def tokenize(text):
                return re.findall(r'\w+', text.lower())

            term_counts = Counter()

            # Boost Entities
            for ent in llm_data.get('entities', []):
                for t in tokenize(ent):
                    if t not in self.stopwords: # Use filtered stopwords
                        term_counts[t] += 5

            # Boost Synonyms
            for syn in llm_data.get('synonyms', []):
                for t in tokenize(syn):
                    if t not in self.stopwords:
                        term_counts[t] += 5

            # Standard Passage
            passage = llm_data.get('passage', '')
            for t in tokenize(passage):
                if t not in self.stopwords:
                    term_counts[t] += 1
            
            # Convert to CandidateTerms
            # "Heuristic: If a term is in data['entities'], give it a starting count of 5" -> We did this via Accumulation
            # Rank by total count
            sorted_terms = term_counts.most_common(self.n_docs_rm3) # Reuse n_docs_rm3 limit or define new n_llm

            for rank, (term, count) in enumerate(sorted_terms, start=1):
                 term_l = term.strip().lower()
                 if term_l in query_terms: # Stopwords already filtered in Counter part
                     continue
                     
                 df, cf = self._get_term_stats(term)
                 
                 if df > (self.collection_size * 0.03):
                     continue
                 all_candidates.append(CandidateTerm(
                     term=term,
                     source="llm",
                     rm3_score=0.0,
                     tf_pseudo=0.0,
                     coverage_pseudo=0.0,
                     df=df,
                     cf=cf,
                     native_rank=rank,
                     native_score=float(count), # Score = Frequency/Boost count
                 ))
            logger.debug(f"Extracted {len(sorted_terms)} LLM candidates")

        logger.info(f"Total candidates extracted: {len(all_candidates)}")
        if kb_override is None:
             self._cache[cache_key] = all_candidates
        return all_candidates

    def _perform_lucene_search(self, query: str, n: int):
        """
        Centralized helper to perform Lucene search and return top docs.
        Returns TopDocs object.
        """
        try:
            # Tokenize query to build a BooleanQuery (more robust than raw QueryParser)
            print(f"DEBUG: Tokenizing query for search: {query}", flush=True)
            query_tokens = self._tokenize_for_search(query)
            print(f"DEBUG: Tokens: {query_tokens}", flush=True)
            
            if not query_tokens:
                # If tokenization returns nothing (e.g. all stopwords), 
                # try a simple whitespace split as ultimate fallback
                query_tokens = [t.strip().lower() for t in query.split() if t.strip()]
            
            if not query_tokens:
                return None

            builder = self.BooleanQueryBuilder()
            from jnius import autoclass
            Occur = autoclass('org.apache.lucene.search.BooleanClause$Occur')
            
            for token in query_tokens:
                # Use TermQuery for each token - this avoids QueryParser entirely
                term_query = self.TermQuery(self.Term(self.field_name, token))
                builder.add(term_query, Occur.SHOULD)
            
            lucene_query = builder.build()
            print("DEBUG: Executing Lucene Search...", flush=True)
            top_docs = self.lucene_searcher.search(lucene_query, n)
            print(f"DEBUG: Search returned {top_docs.totalHits} hits", flush=True)
            return top_docs

        except Exception as e:
            logger.warning(f"Lucene search failed for query '{query}': {e}")
            # Final fallback: escape and parse if BooleanQuery build failed for some reason
            try:
                escaped_query = self.QueryParser.escape(query)
                query_parser = self.QueryParser(self.field_name, self.analyzer)
                lucene_query = query_parser.parse(escaped_query)
                return self.lucene_searcher.search(lucene_query, n)
            except Exception as e2:
                logger.error(f"Ultimate Lucene search failure for '{query}': {e2}")
                return None

    def _tokenize_for_search(self, query_text: str) -> List[str]:
        """Tokenize query text using the same analyzer as the index."""
        try:
            token_stream = self.analyzer.tokenStream(self.field_name, query_text)
            token_stream.reset()
            from jnius import autoclass
            CharTermAttribute = autoclass("org.apache.lucene.analysis.tokenattributes.CharTermAttribute")
            term_attr = token_stream.getAttribute(CharTermAttribute)
            terms = []
            while token_stream.incrementToken():
                terms.append(term_attr.toString())
            token_stream.end()
            token_stream.close()
            return terms
        except Exception as e:
            logger.debug(f"Tokenization failed for search: {e}")
            return []

    def _get_pseudo_relevant_docs_and_tokens(
            self,
            query: str,
            n: int = 10
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Get pseudo-relevant docs AND their tokenized versions.
        Optimized to do retrieval once.
        """
        docs_text = []
        docs_tokens = []

        top_docs = self._perform_lucene_search(query, n)
        if top_docs is None:
            return [], []

        for score_doc in top_docs.scoreDocs:
            try:
                doc = self.lucene_searcher.storedFields().document(score_doc.doc)
                doc_text = doc.get(self.field_name)
                if doc_text:
                    docs_text.append(doc_text)
                    # Simple tokenization for stats calculation
                    # (lowercasing done here to save time in loop)
                    docs_tokens.append(doc_text.lower().split())
            except Exception as e:
                continue

        return docs_text, docs_tokens

    def _get_term_stats(self, term: str) -> Tuple[int, int]:
        """
        Get document frequency (DF) and collection frequency (CF) for a term.
        Includes robust UTF-8 handling for JNI.
        """
        if not term or not isinstance(term, str):
            return 1, 1

        try:
            # SAFELY Create Term
            try:
                # Ensure clean UTF-8 encoding
                # term_encoded = term.strip().lower().encode('utf-8')
                # term_bytes = self.BytesRef(term_encoded)
                # Actually IndexReader.docFreq takes a Term object
                lucene_term = self.Term(self.field_name, term.strip().lower())
            except Exception as e:
                logger.debug(f"Failed to create Term '{term}': {e}")
                return 1, 1

            df = self.lucene_reader.docFreq(lucene_term)
            cf = self.lucene_reader.totalTermFreq(lucene_term)
            
            return max(df, 1), max(cf, 1)

        except Exception as e:
            # Fallback for weird JNI errors or index issues
            # Don't log error on every miss to avoid log spam, use debug
            logger.debug(f"Error getting stats for term '{term}': {e}")
            return 1, 1

    def _compute_tf_pseudo_optimized(
            self,
            term: str,
            pseudo_docs_tokens: List[List[str]]
    ) -> float:
        """
        Compute normalized term frequency using pre-tokenized docs.
        """
        if not pseudo_docs_tokens:
            return 0.0

        term_lower = term.lower()
        total_count = 0
        total_words = 0

        for doc_tokens in pseudo_docs_tokens:
            total_words += len(doc_tokens)
            total_count += doc_tokens.count(term_lower)

        if total_words == 0:
            return 0.0

        return total_count / total_words

    def _compute_coverage_optimized(
            self,
            term: str,
            pseudo_docs_text: List[str]
    ) -> float:
        """
        Compute fraction of pseudo-docs containing the term.
        """
        if not pseudo_docs_text:
            return 0.0

        term_lower = term.lower()
        # Fast string check
        count = sum(1 for doc in pseudo_docs_text if term_lower in doc.lower())

        return count / len(pseudo_docs_text)

    def compute_pseudo_centroid(self, query_text: str, precomputed_docs: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute centroid of pseudo-relevant document embeddings.
        Accepts precomputed docs to avoid re-searching Lucene.
        """
        if precomputed_docs is not None:
            pseudo_docs = precomputed_docs
        else:
            pseudo_docs, _ = self._get_pseudo_relevant_docs_and_tokens(query_text, self.n_pseudo_docs)

        if not pseudo_docs:
            return np.zeros(self.encoder.get_dim(), dtype=np.float32)

        # Encode documents
        pseudo_embeddings = self.encoder.encode(pseudo_docs)

        # Compute centroid
        centroid = np.mean(pseudo_embeddings, axis=0)

        return centroid

    def compute_query_stats(
            self,
            query_text: str,
            precomputed_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute query-level statistics for budget prediction.

        Args:
            query_text: The query
            precomputed_scores: Optional numpy array of top BM25 scores if already retrieved.
        """
        # Tokenize query
        query_tokens = query_text.lower().split()
        q_len = len(query_tokens)

        # === IDF STATISTICS ===
        idfs = []
        for token in query_tokens:
            df, _ = self._get_term_stats(token)
            idf = np.log(self.collection_size / max(df, 1))
            idfs.append(idf)

        avg_idf = float(np.mean(idfs)) if idfs else 0.0
        max_idf = float(np.max(idfs)) if idfs else 0.0

        # === CLARITY ===
        clarity = self._compute_query_clarity(query_text, query_tokens)

        # === RETRIEVAL STATS (Shared Search) ===
        # If scores aren't provided, perform ONE search here to get them
        if precomputed_scores is None:
            top_docs = self._perform_lucene_search(query_text, self.n_pseudo_docs)
            if top_docs and top_docs.scoreDocs:
                precomputed_scores = np.array([sd.score for sd in top_docs.scoreDocs])
            else:
                precomputed_scores = np.array([])

        # === ENTROPY & BM25 STATS ===
        entropy = self._compute_retrieval_entropy_from_scores(precomputed_scores)
        avg_bm25, var_bm25 = self._compute_bm25_stats_from_scores(precomputed_scores)

        # === QUERY TYPE ===
        q_type = self._classify_query_type(query_text)

        return {
            'clarity': float(clarity),
            'entropy': float(entropy),
            'avg_idf': float(avg_idf),
            'max_idf': float(max_idf),
            'avg_bm25': float(avg_bm25),
            'var_bm25': float(var_bm25),
            'q_len': int(q_len),
            'q_type': q_type,
        }

    def _compute_query_clarity(
            self,
            query: str,
            query_tokens: List[str]
    ) -> float:
        """
        Compute query clarity score.
        Clarity = sum_t P(t|q) * log(P(t|q) / P(t|C))
        """
        if not query_tokens:
            return 0.0

        clarity = 0.0
        for token in query_tokens:
            df, cf = self._get_term_stats(token)

            # P(t|C) = collection frequency / collection size
            # Assume avg doc length ~100 for normalization if strict count isn't available
            p_t_c = cf / (self.collection_size * 100)

            # P(t|q) = uniform over query terms
            p_t_q = 1.0 / len(query_tokens)

            if p_t_c > 0:
                clarity += p_t_q * np.log(p_t_q / p_t_c)

        return float(clarity)

    def _compute_retrieval_entropy_from_scores(self, scores: np.ndarray) -> float:
        """
        Compute entropy from pre-computed scores to avoid re-searching.
        """
        if scores.size == 0 or scores.sum() == 0:
            return 0.0

        # Normalize to probabilities
        probs = scores / scores.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)

    def _compute_bm25_stats_from_scores(self, scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute BM25 stats from pre-computed scores.
        """
        if scores.size == 0:
            return 0.0, 0.0

        avg_bm25 = float(np.mean(scores))
        var_bm25 = float(np.var(scores))
        return avg_bm25, var_bm25

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type using Regex-based heuristics (Robust).

        Returns: "navigational", "informational", or "transactional"
        """
        query_lower = query.lower()

        def has_word(keywords, text):
            # Creates regex like: \b(buy|purchase|price)\b
            pattern = r'\b(' + '|'.join([re.escape(k) for k in keywords]) + r')\b'
            return bool(re.search(pattern, text))

        # Navigational indicators
        nav_keywords = [
            'homepage', 'website', 'official', 'login', 'site',
            'main page', 'home page', 'portal', 'www', '.com', '.org'
        ]
        if has_word(nav_keywords, query_lower):
            return "navigational"

        # Transactional indicators
        trans_keywords = [
            'buy', 'purchase', 'price', 'download', 'order', 'shop',
            'cheap', 'deal', 'discount', 'sale', 'cost', 'rent',
            'book', 'reserve', 'subscribe', 'coupon', 'review'
        ]
        if has_word(trans_keywords, query_lower):
            return "transactional"

        # Informational indicators
        info_keywords = [
            'what', 'how', 'why', 'who', 'when', 'where',
            'definition', 'explain', 'guide', 'tutorial', 'learn',
            'meaning', 'example', 'history', 'difference', 'vs'
        ]
        if has_word(info_keywords, query_lower):
            return "informational"

        # Default fallback
        return "informational"

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text using Lucene's EnglishAnalyzer."""
        tokens = []
        try:
            token_stream = self.analyzer.tokenStream(self.field_name, text)
            term_attr = token_stream.addAttribute(get_lucene_classes()['CharTermAttribute'])
            token_stream.reset()
            while token_stream.incrementToken():
                tokens.append(term_attr.toString())
            token_stream.end()
            token_stream.close()
        except Exception as e:
            logger.warning(f"Stemming failed: {e}")
            # Fallback to simple tokenization
            tokens = re.findall(r'\w+', text.lower())
        return tokens

    def __del__(self):
        """Clean up Lucene reader."""
        try:
            if hasattr(self, 'lucene_reader'):
                self.lucene_reader.close()
        except Exception as e:
            logger.debug(f"Error closing Lucene reader: {e}")


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def _main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Test multi-source candidate extraction")
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--kb-wat-output", type=str, default=None)
    parser.add_argument("--vocab-embeddings", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("Initializing encoder...")
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    kb_extractor = None
    if args.kb_wat_output:
        kb_extractor = KBCandidateExtractor(wat_output_path=args.kb_wat_output)

    emb_extractor = None
    if args.vocab_embeddings:
        emb_extractor = EmbeddingCandidateExtractor(encoder=encoder, vocab_path=args.vocab_embeddings)

    from src.utils.lucene_utils import initialize_lucene
    
    # Initialize Lucene (adjust path as needed)
    lucene_path = "lucene_jars"
    if not initialize_lucene(lucene_path):
        print("Failed to initialize Lucene")
        return

    logger.info("Initializing candidate extractor...")
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        emb_extractor=emb_extractor,
    )

    logger.info(f"Extracting candidates for: {args.query}")
    candidates = extractor.extract_all_candidates(query_text=args.query)

    print(f"\nQuery: {args.query}")
    print(f"Total candidates: {len(candidates)}")

    # Compute query stats (testing shared retrieval logic internally)
    print("\nQUERY STATISTICS:")
    stats = extractor.compute_query_stats(args.query)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _main_cli()
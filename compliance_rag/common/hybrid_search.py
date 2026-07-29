"""
common/hybrid_search.py

Hybrid Search = BM25 (keyword/lexical) + vector/semantic search (Chroma),
combined by weighted score fusion.

Weighting: 50/50 BM25/vector by default (configurable via BM25_WEIGHT /
VECTOR_WEIGHT in .env).

No LangChain retriever abstractions are used here: both halves (BM25Okapi
and Chroma) are called directly, and the fusion logic is plain Python.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple whitespace/punctuation tokenizer that works for both Arabic and English."""
    return re.findall(r"[\w؀-ۿ]+", text.lower())


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    bm25_score: float
    vector_score: float
    fused_score: float


class HybridIndex:
    """
    Wraps one Chroma collection + one in-memory BM25 index over the same
    set of documents, and exposes a single `search()` method that returns
    fused-and-ranked results.
    """

    def __init__(self, chroma_collection, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self.collection = chroma_collection
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metas: List[dict] = []
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """(Re)build the BM25 index from whatever is currently in the Chroma collection."""
        data = self.collection.get(include=["documents", "metadatas"])
        self._doc_ids = data["ids"]
        self._doc_texts = data["documents"]
        self._doc_metas = data["metadatas"]
        tokenized = [_tokenize(t) for t in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def refresh(self):
        """Call after adding/removing documents from the underlying Chroma collection."""
        self._rebuild_bm25()

    @staticmethod
    def _min_max_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def search(self, query: str, query_embedding: Optional[List[float]] = None,
               top_k: int = 5, where: Optional[dict] = None) -> List[SearchResult]:
        """
        query_embedding: precomputed embedding for `query` (from common.embeddings.embed_query).
        where: optional Chroma metadata filter, e.g. {"language": "en"}.
        """
        n_docs = len(self._doc_ids)
        if n_docs == 0:
            return []

        # --- BM25 side ---
        bm25_scores_by_id = {}
        if self._bm25 is not None:
            raw_bm25 = self._bm25.get_scores(_tokenize(query))
            norm_bm25 = self._min_max_normalize(list(raw_bm25))
            bm25_scores_by_id = dict(zip(self._doc_ids, norm_bm25))

        # --- Vector side (Chroma) ---
        vector_scores_by_id = {}
        vector_docs_by_id = {}
        vector_metas_by_id = {}
        if query_embedding is not None:
            n_results = min(max(top_k * 4, 20), n_docs)
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
            ids = res["ids"][0]
            dists = res["distances"][0]  # Chroma cosine distance: lower = more similar
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            sims = [1.0 - d for d in dists]  # convert distance -> similarity
            norm_sims = self._min_max_normalize(sims)
            for _id, sim, doc, meta in zip(ids, norm_sims, docs, metas):
                vector_scores_by_id[_id] = sim
                vector_docs_by_id[_id] = doc
                vector_metas_by_id[_id] = meta

        # --- fuse ---
        all_ids = set(bm25_scores_by_id) | set(vector_scores_by_id)
        # if a `where` filter was applied on the vector side, restrict BM25-only
        # candidates to the same filter by intersecting with vector-returned ids
        # when a filter is active and vector side actually ran.
        if where is not None and vector_scores_by_id:
            all_ids &= set(vector_scores_by_id)

        id_to_text = {i: t for i, t in zip(self._doc_ids, self._doc_texts)}
        id_to_meta = {i: m for i, m in zip(self._doc_ids, self._doc_metas)}

        results = []
        for _id in all_ids:
            b = bm25_scores_by_id.get(_id, 0.0)
            v = vector_scores_by_id.get(_id, 0.0)
            fused = self.bm25_weight * b + self.vector_weight * v
            results.append(SearchResult(
                id=_id,
                text=vector_docs_by_id.get(_id, id_to_text.get(_id, "")),
                metadata=vector_metas_by_id.get(_id, id_to_meta.get(_id, {})),
                bm25_score=b,
                vector_score=v,
                fused_score=fused,
            ))

        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results[:top_k]

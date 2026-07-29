"""
common/policy_index.py

Builds an ephemeral (in-memory, per-session) Chroma collection + BM25 index
over the chunks of the user's uploaded compliance evidence, using the exact
same embedding model and HybridIndex fusion logic as the static SAMA CSF
index (common/embeddings.py, common/hybrid_search.py). Nothing here is
persisted to disk: a fresh upload = a fresh in-memory collection, discarded
when the process/session ends.

Evidence can be a single file, a list of files, or a directory containing
several files (policies, procedures, reports, screenshots) - see
common/policy_ingest.py for the supported file types and how images are
turned into text.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Union

import chromadb

from common.embeddings import embed_texts
from common.hybrid_search import HybridIndex
from common.policy_ingest import chunk_policy_multi, PolicyChunk
from common.image_ingest import SUPPORTED_IMAGE_SUFFIXES

BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", 0.5))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", 0.5))

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"} | SUPPORTED_IMAGE_SUFFIXES


def _resolve_file_list(file_path: Union[str, List[str]]) -> List[str]:
    """
    Normalizes the accepted input shapes into a flat list of file paths:
      - a single file path (str)
      - a directory path (str) -> every supported file inside it
      - a list of file paths

    Raises FileNotFoundError with a clear message (rather than letting a
    missing/placeholder path silently fall through to a confusing
    "unsupported file type" error later in extract_pages()).
    """
    if isinstance(file_path, (list, tuple)):
        paths = list(file_path)
        missing = [f for f in paths if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(
                "The following evidence file(s) were not found on disk: "
                + ", ".join(missing)
                + ". Check the path(s) and try again."
            )
        return paths

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Evidence path not found: {file_path}\n"
            "This looks like a placeholder/example path rather than a real "
            "one - replace it with the actual path to your evidence file or "
            "folder on disk."
        )
    if p.is_dir():
        files = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not files:
            raise ValueError(
                f"No supported evidence files found in directory: {file_path} "
                f"(supported types: {sorted(SUPPORTED_SUFFIXES)})"
            )
        return files

    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported evidence file type: {p.suffix or '(none)'} for {file_path} "
            f"(supported types: {sorted(SUPPORTED_SUFFIXES)})"
        )
    return [str(p)]


def build_policy_index(file_path: Union[str, List[str]], target_words: int = 220,
                        overlap_words: int = 40, groq_api_key: Optional[str] = None) -> HybridIndex:
    """
    file_path: a single evidence file, a directory of evidence files, or a
    list of evidence file paths. Any mix of PDF/DOCX/TXT/images is supported.
    """
    paths = _resolve_file_list(file_path)
    chunks: List[PolicyChunk] = chunk_policy_multi(
        paths, target_words=target_words, overlap_words=overlap_words, groq_api_key=groq_api_key
    )
    if not chunks:
        raise ValueError("No extractable text found in the uploaded evidence.")

    client = chromadb.EphemeralClient()
    collection_name = f"policy_{uuid.uuid4().hex[:12]}"
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {"source_page": c.source_page, "chunk_id": c.chunk_id, "source_file": c.source_file}
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return HybridIndex(collection, bm25_weight=BM25_WEIGHT, vector_weight=VECTOR_WEIGHT)


if __name__ == "__main__":
    import sys
    idx = build_policy_index(sys.argv[1:] if len(sys.argv) > 2 else sys.argv[1])
    print(f"Built in-memory policy index with {idx.collection.count()} chunks")

"""
common/embeddings.py

Shared embedding function used by both the SAMA CSF static index and the
per-session uploaded company policy index.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - Multilingual pretrained transformer (~50 languages), covers English and
    Arabic control texts and company policies.
  - 384-dim output, ONNX runtime via `fastembed` (no torch/GPU required).
  - Downloaded automatically (~220MB) from Hugging Face on first run and
    cached locally afterwards.
"""

from functools import lru_cache
from typing import List

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents (control chunks / policy chunks)."""
    model = _get_model()
    return [v.tolist() for v in model.embed(texts)]


def embed_query(text: str) -> List[float]:
    """Embed a single search query."""
    model = _get_model()
    return list(model.query_embed(text))[0].tolist()


if __name__ == "__main__":
    # quick smoke test
    vecs = embed_texts(["hello world", "مرحبا بالعالم"])
    print("dim:", len(vecs[0]), "count:", len(vecs))

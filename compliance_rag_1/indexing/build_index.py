"""
indexing/build_index.py

Builds the static SAMA CSF Chroma collection from the parsed control chunks
(../ingestion/controls.jsonl [English] and controls_ar.jsonl [Arabic]).

Run once during setup (or whenever ingestion/*.jsonl changes):
    python indexing/build_index.py

Produces a persistent Chroma DB under ../chroma_db/ containing 72 documents
(36 English + 36 Arabic SAMA CSF controls), each embedded with the real
pretrained multilingual embedding model in common/embeddings.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from common.embeddings import embed_texts

HERE = Path(__file__).resolve().parent
INGESTION_DIR = HERE.parent / "ingestion"
CHROMA_DIR = HERE.parent / "chroma_db"
COLLECTION_NAME = "sama_csf_controls"


def load_controls(path: Path, language: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["language"] = language
            records.append(r)
    return records


def main():
    en = load_controls(INGESTION_DIR / "controls.jsonl", "en")
    ar = load_controls(INGESTION_DIR / "controls_ar.jsonl", "ar")
    all_records = en + ar
    print(f"Loaded {len(en)} EN + {len(ar)} AR = {len(all_records)} total controls")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{r['language']}::{r['control_id']}" for r in all_records]
    documents = [r["text"] for r in all_records]
    metadatas = [{
        "control_id": r["control_id"],
        "title": r["title"],
        "domain": r["domain"],
        "language": r["language"],
        "parent_control_id": r.get("parent_control_id") or "",
    } for r in all_records]

    print("Embedding documents with sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ...")
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Indexed {collection.count()} documents into Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()

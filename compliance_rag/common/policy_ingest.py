"""
common/policy_ingest.py

Extracts and chunks the company's own compliance evidence that the user
uploads at runtime. Evidence is not limited to a single policy document -
it can be several files at once (policies, procedures, reports, and
screenshots of system/security configuration), matching how compliance
evidence is actually gathered in practice.

Supported file types:
  - PDF, DOCX, TXT: extracted as text directly.
  - PNG/JPG/JPEG/BMP/TIFF/WEBP: extracted via common/image_ingest.py, which
    combines OCR (literal on-screen text) with a vision-model description
    (for evidence where the meaning is visual, e.g. a settings toggle or a
    network diagram).

Unlike the SAMA CSF source (a regularly-numbered standard, parsed with
regex in ingestion/ingest_sama*.py), evidence documents have no predictable
structure, so a generic paragraph-aware sliding-window chunker is used
instead.

This produces the *other* half of the comparison the compliance engine
needs: for each SAMA control, we hybrid-search the resulting per-session
index to find the most relevant excerpt(s) of the company's own evidence,
then ask the LLM to judge compliant / partial / missing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from common.image_ingest import is_supported_image, extract_image_evidence


@dataclass
class PolicyChunk:
    chunk_id: str
    text: str
    source_page: int  # 1-indexed page/paragraph-group number, best effort
    source_file: str = ""  # original filename this chunk was extracted from


def _extract_pdf_text_by_page(path: Path) -> List[str]:
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def _extract_docx_text(path: Path) -> List[str]:
    import docx
    d = docx.Document(str(path))
    # DOCX has no native "page" concept in python-docx; treat the whole
    # document as one "page" of paragraphs, chunker below still splits it.
    paragraphs = [p.text for p in d.paragraphs if p.text.strip()]
    return ["\n\n".join(paragraphs)]


def extract_pages(path: str, groq_api_key: Optional[str] = None) -> List[str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text_by_page(p)
    elif suffix in (".docx",):
        return _extract_docx_text(p)
    elif suffix == ".txt":
        return [p.read_text(encoding="utf-8", errors="ignore")]
    elif is_supported_image(str(p)):
        return [extract_image_evidence(str(p), api_key=groq_api_key)]
    else:
        raise ValueError(
            f"Unsupported evidence file type: {suffix} "
            "(use PDF, DOCX, TXT, or an image: PNG/JPG/JPEG/BMP/TIFF/WEBP)"
        )


def _split_paragraphs(page_text: str) -> List[str]:
    # split on blank lines; also treat single newlines as soft breaks we can
    # rejoin, since PDFs often hard-wrap lines mid-sentence.
    raw_paragraphs = re.split(r"\n\s*\n", page_text)
    cleaned = []
    for para in raw_paragraphs:
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if para:
            cleaned.append(para)
    return cleaned


def _chunk_pages(pages: List[str], source_file: str, chunk_index: int,
                 target_words: int, overlap_words: int) -> Tuple[List[PolicyChunk], int]:
    """
    Paragraph-aware sliding-window chunking over already-extracted page texts:
      1. Split each page into paragraphs.
      2. Greedily pack paragraphs into ~target_words chunks, carrying the
         last `overlap_words` words of a chunk into the start of the next
         one so semantic context isn't lost at a chunk boundary.
    `chunk_index` is the running counter across all files so chunk_ids stay
    unique when multiple evidence files are combined into one index.
    """
    chunks: List[PolicyChunk] = []

    for page_num, page_text in enumerate(pages, start=1):
        paragraphs = _split_paragraphs(page_text)
        if not paragraphs:
            continue

        current_words: List[str] = []
        for para in paragraphs:
            para_words = para.split()
            if current_words and len(current_words) + len(para_words) > target_words:
                chunk_text = " ".join(current_words)
                chunk_index += 1
                chunks.append(PolicyChunk(
                    chunk_id=f"policy::{chunk_index}",
                    text=chunk_text,
                    source_page=page_num,
                    source_file=source_file,
                ))
                overlap = current_words[-overlap_words:] if overlap_words else []
                current_words = overlap + para_words
            else:
                current_words.extend(para_words)

        if current_words:
            chunk_text = " ".join(current_words)
            chunk_index += 1
            chunks.append(PolicyChunk(
                chunk_id=f"policy::{chunk_index}",
                text=chunk_text,
                source_page=page_num,
                source_file=source_file,
            ))

    return chunks, chunk_index


def chunk_policy(path: str, target_words: int = 220, overlap_words: int = 40,
                  groq_api_key: Optional[str] = None) -> List[PolicyChunk]:
    """Chunk a single evidence file (PDF, DOCX, TXT, or image)."""
    pages = extract_pages(path, groq_api_key=groq_api_key)
    chunks, _ = _chunk_pages(pages, Path(path).name, 0, target_words, overlap_words)
    return chunks


def chunk_policy_multi(paths: List[str], target_words: int = 220, overlap_words: int = 40,
                        groq_api_key: Optional[str] = None) -> List[PolicyChunk]:
    """
    Chunk multiple evidence files (any mix of PDF/DOCX/TXT/images) into one
    combined, globally-unique-chunk-id list, each chunk tagged with the
    filename it came from.
    """
    all_chunks: List[PolicyChunk] = []
    chunk_index = 0
    for path in paths:
        pages = extract_pages(path, groq_api_key=groq_api_key)
        chunks, chunk_index = _chunk_pages(pages, Path(path).name, chunk_index,
                                           target_words, overlap_words)
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    import sys
    chunks = chunk_policy_multi(sys.argv[1:])
    print(f"Produced {len(chunks)} chunks from {len(sys.argv) - 1} file(s)")
    for c in chunks[:3]:
        print("---")
        print(c.chunk_id, "| file", c.source_file, "| page", c.source_page, "|", len(c.text.split()), "words")
        print(c.text[:200])

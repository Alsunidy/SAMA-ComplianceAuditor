"""
common/image_ingest.py

Extracts usable text from image evidence (screenshots of system settings,
configuration panels, network/architecture diagrams, etc.) so it can flow
into the same chunking and hybrid-search pipeline used for PDF/DOCX policy
documents.

Two extraction paths are combined for each image:
  1. OCR (pytesseract) - picks up literal on-screen text (menu labels,
     toggle states like "Enabled", IP addresses, usernames, etc.). Fast,
     free, no network call.
  2. Vision-language description (Groq multimodal model) - produces a
     semantic description of what the image shows, which matters for
     images where the meaning is visual rather than textual (e.g. a
     network topology diagram with a few labels, or a dashboard where the
     relevant fact is a checkbox/toggle state rather than a text string).

Both outputs are concatenated into one evidence text per image, so the
downstream chunker/retriever/judge never need to know whether a given
excerpt originally came from a text document or an image.
"""

import base64
import os
import re
from pathlib import Path
from typing import Optional

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}

VISION_PROMPT = (
    "This image is evidence submitted as part of a cyber security compliance "
    "review (e.g. a screenshot of a system/security configuration screen, an "
    "admin dashboard, or a network/architecture diagram). Describe factually "
    "what it shows: the system or application involved, any settings, toggles, "
    "or status values visible (e.g. enabled/disabled, policy names, user "
    "counts), and any security-relevant configuration or architecture depicted. "
    "Transcribe any readable on-screen text exactly. Do not speculate about "
    "anything not visible in the image."
)


def is_supported_image(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def ocr_image_text(path: str) -> str:
    """Extract literal on-screen text via Tesseract OCR. Returns '' on failure."""
    try:
        import pytesseract
        from PIL import Image
        with Image.open(path) as img:
            return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def describe_image_with_vision(path: str, api_key: Optional[str] = None,
                                model: str = VISION_MODEL) -> str:
    """
    Ask a Groq vision-capable model to describe the image's security-relevant
    content. Returns '' if no API key is available or the call fails, so
    callers can gracefully fall back to OCR-only extraction.

    Qwen 3.6 27B is a thinking-capable model. reasoning_effort="none" tells
    it to skip its internal reasoning pass entirely (supported natively by
    this model), so the full token budget goes to the visible description
    instead of being spent on an invisible <think>...</think> trace first.
    reasoning_format="hidden" is kept alongside it as a second safety net,
    and the regex strip below is a third one - between the three, a
    reasoning trace should never leak into the evidence text sent to the
    retriever/judge, and the visible answer should never get cut short by
    reasoning eating the token budget.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=key)
        suffix = Path(path).suffix.lower()
        mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
        b64 = _encode_image_base64(path)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            temperature=0.2,
            max_completion_tokens=2000,
            reasoning_format="hidden",
            reasoning_effort="none",
        )
        text = completion.choices[0].message.content or ""
        text = _THINK_TAG_RE.sub("", text).strip()
        return text
    except Exception:
        return ""


def extract_image_evidence(path: str, api_key: Optional[str] = None) -> str:
    """
    Combines OCR text and a vision-model description into a single evidence
    text block for the given image, prefixed with the source filename so the
    origin of the evidence is traceable downstream (matched_policy_excerpt).
    """
    filename = Path(path).name
    ocr_text = ocr_image_text(path)
    description = describe_image_with_vision(path, api_key=api_key)

    parts = [f"[Image evidence: {filename}]"]
    if description:
        parts.append(description)
    if ocr_text:
        parts.append(f"On-screen text (OCR): {ocr_text}")
    if not description and not ocr_text:
        parts.append("(No text or description could be extracted from this image.)")

    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()  # picks up GROQ_API_KEY from .env when run standalone
    result = extract_image_evidence(sys.argv[1])
    print(result)

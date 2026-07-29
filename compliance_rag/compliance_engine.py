"""
compliance_engine.py

The core sequential pipeline (no agents/LangGraph - a plain retrieve-then-judge
loop, per control):

    for each SAMA CSF control:
        1. embed the control's own text
        2. hybrid-search (BM25 + vector) the uploaded evidence's per-session
           index for the most relevant excerpt(s)
        3. ask the LLM (Groq) to judge: Compliant / Partial / Missing,
           with a gap description + recommendation
        4. collect the verdict

Evidence is not limited to one file: a single policy document, a list of
files, or a whole directory of evidence (policies, procedures, reports, and
screenshots of system/security configuration) are all accepted - see
common/policy_index.py and common/policy_ingest.py for how each type is
turned into text and indexed together.

Run directly for a quick CLI gap analysis (one or more evidence files/dirs):
    python compliance_engine.py path/to/company_policy.pdf --language en
    python compliance_engine.py policy.pdf screenshot1.png screenshot2.png --language en
    python compliance_engine.py path/to/evidence_folder/ --language en

Or import `run_gap_analysis()` from the FastAPI backend (api/main.py, task #8).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from common.embeddings import embed_query
from common.policy_index import build_policy_index
from common.llm_judge import judge_control, Verdict, STATUS_CODE, STATUS_LABEL

HERE = Path(__file__).resolve().parent
INGESTION_DIR = HERE / "ingestion"

TOP_K_EXCERPTS = int(os.environ.get("RETRIEVER_TOP_K", 3))


def load_controls(language: str = "en") -> List[dict]:
    filename = "controls.jsonl" if language == "en" else "controls_ar.jsonl"
    path = INGESTION_DIR / filename
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sort_key(control_id: str):
    return [int(p) for p in control_id.split(".")]


def run_gap_analysis(policy_file_path: Union[str, List[str]], language: str = "en",
                      top_k: int = TOP_K_EXCERPTS,
                      groq_api_key: Optional[str] = None,
                      progress_callback=None) -> List[Verdict]:
    """
    policy_file_path: a single evidence file, a directory of evidence files,
    or a list of evidence file paths. Any mix of PDF/DOCX/TXT/images is
    supported (see common/policy_index.py).

    progress_callback(done: int, total: int, control_id: str) is called after
    each control is judged, useful for a Streamlit/FastAPI progress bar.
    """
    from groq import Groq

    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Groq API key found. Set GROQ_API_KEY in your environment/.env "
            "file, or pass groq_api_key= explicitly."
        )
    client = Groq(api_key=api_key)

    controls = load_controls(language)
    controls.sort(key=lambda r: _sort_key(r["control_id"]))

    print(f"Building in-memory index for uploaded evidence: {policy_file_path}")
    policy_index = build_policy_index(policy_file_path, groq_api_key=api_key)
    print(f"Evidence index ready: {policy_index.collection.count()} chunks")

    verdicts: List[Verdict] = []
    total = len(controls)

    for i, control in enumerate(controls, start=1):
        query_text = f"{control['title']}. {control['principle']} {control['objective']}"
        query_embedding = embed_query(query_text)

        results = policy_index.search(query_text, query_embedding=query_embedding, top_k=top_k)
        # tag each excerpt with its source filename so the LLM's judgment (and
        # the resulting matched_policy_excerpt in the report) stays traceable
        # to a specific evidence document/screenshot.
        excerpts = [
            f"[Source: {r.metadata.get('source_file', 'unknown')}] {r.text}"
            for r in results
        ]
        chunk_ids = [r.id for r in results]

        verdict = judge_control(client, control, excerpts, chunk_ids)
        verdicts.append(verdict)

        if progress_callback:
            progress_callback(i, total, control["control_id"])
        else:
            print(f"[{i}/{total}] {control['control_id']} {control['title']}: {verdict.status}")

    return verdicts


def verdicts_to_dicts(verdicts: List[Verdict]) -> List[dict]:
    """
    Serializes verdicts to the report JSON shape:
        {
          "control_id": "3.1.1",
          "control_domain": "Cybersecurity Governance and Leadership",
          "control_text": "...",
          "status_code": "PASS" | "PARTIAL" | "FAIL",
          "status_label": "Compliant" | "Partially Compliant" | "Non-Compliant",
          "matched_policy_excerpt": "...",
          "justification": "...",
          "recommendation": "..."
        }
    `evidence_chunk_ids` is included as an extra traceability field beyond
    the core schema (harmless extra key, useful for the evidence/ writeup).
    """
    results = []
    for v in verdicts:
        results.append({
            "control_id": v.control_id,
            "control_domain": v.control_domain,
            "control_text": v.control_text,
            "status_code": STATUS_CODE[v.status],
            "status_label": STATUS_LABEL[v.status],
            "matched_policy_excerpt": v.matched_policy_excerpt,
            "justification": v.justification,
            "recommendation": v.recommendation,
            "evidence_chunk_ids": v.evidence_chunk_ids,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run SAMA CSF gap analysis against one or more company evidence files."
    )
    parser.add_argument(
        "policy_files", nargs="+",
        help="One or more evidence files (PDF, DOCX, TXT, or PNG/JPG/JPEG/BMP/TIFF/WEBP "
             "screenshots), or a single directory containing them.",
    )
    parser.add_argument("--language", choices=["en", "ar"], default="en")
    parser.add_argument("--top-k", type=int, default=TOP_K_EXCERPTS)
    parser.add_argument("--out", default="gap_analysis_result.json")
    args = parser.parse_args()

    # a single positional arg that is a directory is passed through as-is;
    # multiple positional args are passed through as a list of files.
    policy_input = args.policy_files[0] if len(args.policy_files) == 1 else args.policy_files

    start = time.time()
    verdicts = run_gap_analysis(policy_input, language=args.language, top_k=args.top_k)
    elapsed = time.time() - start

    results = verdicts_to_dicts(verdicts)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    counts = {"Compliant": 0, "Partial": 0, "Missing": 0}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    print(f"\nDone in {elapsed:.1f}s. Compliant={counts['Compliant']} "
          f"Partial={counts['Partial']} Missing={counts['Missing']}")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

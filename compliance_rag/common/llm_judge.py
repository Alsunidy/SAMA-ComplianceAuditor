"""
common/llm_judge.py

Wraps the Groq LLM call that judges one SAMA CSF control against the most
relevant excerpt(s) of the company's own policy document, returning a
structured verdict: status (Compliant / Partial / Missing), a justification,
the matched policy excerpt, and a recommendation.

Model: configurable via the GROQ_MODEL environment variable (defaults to
openai/gpt-oss-120b if unset), via the Groq API.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List

import groq

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Groq's free/on-demand tier has a low tokens-per-minute (TPM) limit, easily
# hit when judging 36 controls back to back (each call runs a few thousand
# tokens of control text + policy excerpts). Retry with backoff instead of
# letting one 429 kill the whole batch.
MAX_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 5.0

_RETRY_MS_RE = re.compile(r"try again in ([\d.]+)ms", re.IGNORECASE)
_RETRY_S_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _seconds_to_wait(error: "groq.RateLimitError") -> float:
    """Best-effort extraction of how long to wait before retrying."""
    try:
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            return float(retry_after)
    except Exception:
        pass

    message = str(error)
    m = _RETRY_MS_RE.search(message)
    if m:
        return max(float(m.group(1)) / 1000.0, 0.5)
    m = _RETRY_S_RE.search(message)
    if m:
        return float(m.group(1))

    return DEFAULT_BACKOFF_SECONDS

VALID_STATUSES = {"Compliant", "Partial", "Missing"}

# These display mappings are applied only at the final report-serialization
# boundary (see compliance_engine.verdicts_to_dicts), never used for internal
# status comparisons/tests - internal code always compares against the plain
# VALID_STATUSES strings ("Compliant" / "Partial" / "Missing").
STATUS_CODE = {"Compliant": "PASS", "Partial": "PARTIAL", "Missing": "FAIL"}
STATUS_LABEL = {"Compliant": "Compliant", "Partial": "Partially Compliant", "Missing": "Non-Compliant"}

SYSTEM_PROMPT = """You are a cyber security compliance auditor for Saudi \
financial institutions. You are given ONE control from the SAMA Cyber \
Security Framework (SAMA CSF) and one or more excerpts from a financial \
institution's own internal security policy document (retrieved because \
they are the most relevant passages found for this control).

Your job: judge whether the company's policy, as shown in the excerpts, \
satisfies the SAMA control's Principle/Objective/Control considerations.

Judge strictly based on the excerpts given — do not assume the company \
does something just because it would be reasonable to do so; if the \
excerpts don't mention it, treat it as not addressed. If no excerpt is \
provided or the excerpts are clearly unrelated to the control's topic, \
the status must be "Missing".

Respond with ONLY a JSON object with exactly these fields:
{
  "status": "Compliant" | "Partial" | "Missing",
  "justification": "<one or two sentences: why you chose this status, referencing what the excerpts do or don't show.>",
  "recommendation": "<one or two sentences: a concrete, actionable recommendation to close the gap. Empty string if status is Compliant.>"
}

Definitions:
- Compliant: the excerpts clearly and specifically address the control's requirements.
- Partial: the excerpts address the general topic but miss specific requirements, details, or formalization (e.g. mentioned informally but not "defined, approved and documented").
- Missing: the excerpts do not address the control's topic at all, or no relevant excerpt was found.
"""


@dataclass
class Verdict:
    control_id: str
    control_domain: str
    control_text: str
    status: str  # plain "Compliant" | "Partial" | "Missing" - see STATUS_CODE/STATUS_LABEL for display
    matched_policy_excerpt: str
    justification: str
    recommendation: str
    evidence_chunk_ids: List[str]  # extra: all top-k chunk ids used as context, for traceability


def _build_user_prompt(control: dict, policy_excerpts: List[str]) -> str:
    control_block = (
        f"SAMA CSF Control {control['control_id']} - {control['title']}\n"
        f"Domain: {control['domain']}\n"
        f"Principle: {control['principle']}\n"
        f"Objective: {control['objective']}\n"
        f"Control considerations:\n{control['considerations_text']}"
    )
    if policy_excerpts:
        excerpts_block = "\n\n".join(
            f"[Excerpt {i+1}]\n{ex}" for i, ex in enumerate(policy_excerpts)
        )
    else:
        excerpts_block = "(No relevant excerpt was found in the company's policy document.)"

    return (
        f"=== SAMA CSF CONTROL ===\n{control_block}\n\n"
        f"=== COMPANY POLICY EXCERPTS (retrieved via hybrid search) ===\n{excerpts_block}\n\n"
        "Judge this control now and respond with the JSON object only."
    )


def judge_control(client, control: dict, policy_excerpts: List[str],
                   evidence_chunk_ids: List[str]) -> Verdict:
    """
    client: a groq.Groq client instance (created once by the caller).
    control: one record from controls.jsonl / controls_ar.jsonl (must include
             the combined 'text' field written by ingestion/ingest_sama*.py).
    policy_excerpts: list of retrieved policy chunk texts (top-k from HybridIndex.search),
                     ranked best-first.
    evidence_chunk_ids: matching list of chunk ids, for traceability in the report.
    """
    user_prompt = _build_user_prompt(control, policy_excerpts)

    completion = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            completion = raw.parse()
            remaining = raw.headers.get("x-ratelimit-remaining-tokens")
            limit = raw.headers.get("x-ratelimit-limit-tokens")
            reset = raw.headers.get("x-ratelimit-reset-tokens")
            if remaining is not None and limit is not None:
                print(f"  [rate limit] control {control['control_id']}: "
                      f"{remaining}/{limit} tokens remaining this minute "
                      f"(resets in {reset})")
            break
        except groq.RateLimitError as e:
            last_error = e
            wait = _seconds_to_wait(e)
            print(f"  [rate limit] control {control['control_id']}: waiting {wait:.1f}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError) as e:
            last_error = e
            wait = DEFAULT_BACKOFF_SECONDS * attempt
            print(f"  [transient error] control {control['control_id']}: {e} "
                  f"- retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

    if completion is None:
        # all retries exhausted: fail this one control gracefully, don't crash the batch
        return Verdict(
            control_id=control["control_id"],
            control_domain=control.get("domain", ""),
            control_text=control.get("text", f"{control['title']}\n{control['principle']}"),
            status="Missing",
            matched_policy_excerpt=policy_excerpts[0] if policy_excerpts else "",
            justification=f"LLM call failed after {MAX_RETRIES} retries: {last_error}",
            recommendation="Re-run this control manually once the rate limit / API issue clears.",
            evidence_chunk_ids=evidence_chunk_ids,
        )

    raw = completion.choices[0].message.content
    try:
        parsed = json.loads(raw)
        status = parsed.get("status", "").strip()
        if status not in VALID_STATUSES:
            status = "Missing"
        justification = parsed.get("justification", "").strip()
        recommendation = parsed.get("recommendation", "").strip()
    except (json.JSONDecodeError, AttributeError):
        # fail safe: never crash the whole batch run over one bad LLM response
        status = "Missing"
        justification = "LLM response could not be parsed."
        recommendation = raw[:300] if raw else ""

    return Verdict(
        control_id=control["control_id"],
        control_domain=control.get("domain", ""),
        control_text=control.get("text", f"{control['title']}\n{control['principle']}"),
        status=status,
        matched_policy_excerpt=policy_excerpts[0] if policy_excerpts else "",
        justification=justification,
        recommendation=recommendation,
        evidence_chunk_ids=evidence_chunk_ids,
    )

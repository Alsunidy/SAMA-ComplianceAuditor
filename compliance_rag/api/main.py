"""
api/main.py

FastAPI backend exposing the compliance engine over HTTP, for the Streamlit
UI (task #9) or any other client to consume.

Because a full 36-control gap analysis involves one LLM call per control and
takes several minutes, POST /api/analyze does not block the request: it
saves the uploaded evidence, starts the analysis in a background task, and
immediately returns a job_id. Clients poll GET /api/analyze/{job_id} for
progress and, once status is "done", the results.

Endpoints:
    GET  /api/health                 - liveness check
    GET  /api/controls               - list the SAMA CSF controls being checked
    POST /api/analyze                - upload evidence, start a gap analysis job
    GET  /api/analyze/{job_id}       - poll job status/progress/results
    POST /api/report                 - render a finished job's results as a PDF
"""

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from compliance_engine import load_controls, run_gap_analysis, verdicts_to_dicts, TOP_K_EXCERPTS
from common.policy_index import SUPPORTED_SUFFIXES
from report_generator import generate_pdf_report

app = FastAPI(
    title="SAMA Compliance Auditor API",
    description="Compares a financial institution's security evidence against the SAMA Cyber Security Framework.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. A job's uploaded files and any generated report PDF
# live under its own temp directory, removed once the job's results have
# been fetched and, separately, once its report has been downloaded.
JOBS: dict = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    done: int = 0
    total: int = 0
    current_control: Optional[str] = None
    error: Optional[str] = None
    results: Optional[list] = None


class ReportRequest(BaseModel):
    job_id: Optional[str] = None
    results: Optional[list] = None
    company_name: Optional[str] = None


def _run_job(job_id: str, paths: List[str], language: str, top_k: int, work_dir: str):
    job = JOBS[job_id]

    def progress_callback(done, total, control_id):
        job["done"] = done
        job["total"] = total
        job["current_control"] = control_id

    try:
        verdicts = run_gap_analysis(
            paths, language=language, top_k=top_k, progress_callback=progress_callback
        )
        job["results"] = verdicts_to_dicts(verdicts)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/controls")
def get_controls(language: str = "en"):
    if language not in ("en", "ar"):
        raise HTTPException(400, "language must be 'en' or 'ar'")
    controls = load_controls(language)
    return [
        {"control_id": c["control_id"], "title": c["title"], "domain": c["domain"]}
        for c in controls
    ]


@app.post("/api/analyze", response_model=JobStatus)
async def analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more evidence files (PDF, DOCX, TXT, or images)"),
    language: str = Form("en"),
    top_k: int = Form(TOP_K_EXCERPTS),
):
    if language not in ("en", "ar"):
        raise HTTPException(400, "language must be 'en' or 'ar'")
    if not files:
        raise HTTPException(400, "No files uploaded")

    work_dir = tempfile.mkdtemp(prefix="sama_evidence_")
    saved_paths = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                400,
                f"Unsupported file type '{suffix}' for '{upload.filename}'. "
                f"Supported types: {sorted(SUPPORTED_SUFFIXES)}",
            )
        dest = Path(work_dir) / upload.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest))

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "done": 0,
        "total": 0,
        "current_control": None,
        "error": None,
        "results": None,
    }

    background_tasks.add_task(_run_job, job_id, saved_paths, language, top_k, work_dir)

    return JOBS[job_id]


@app.get("/api/analyze/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job


@app.post("/api/report")
def report(req: ReportRequest):
    if req.job_id:
        job = JOBS.get(req.job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        if job["status"] != "done":
            raise HTTPException(409, f"Job is not finished yet (status: {job['status']})")
        results = job["results"]
    elif req.results:
        results = req.results
    else:
        raise HTTPException(400, "Provide either job_id or results")

    out_path = os.path.join(tempfile.gettempdir(), f"gap_analysis_report_{uuid.uuid4().hex}.pdf")
    generate_pdf_report(results, out_path, company_name=req.company_name)

    return FileResponse(
        out_path,
        media_type="application/pdf",
        filename="gap_analysis_report.pdf",
        background=None,
    )

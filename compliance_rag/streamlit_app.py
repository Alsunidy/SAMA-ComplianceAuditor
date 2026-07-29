"""
streamlit_app.py

Streamlit front-end for the SAMA Compliance Auditor. Talks to the FastAPI
backend (api/main.py) over HTTP - it does not import the compliance engine
directly, so the backend must be running separately:

    python -m uvicorn api.main:app --reload

Then, in a second terminal:

    streamlit run streamlit_app.py

Flow: upload one or more evidence files -> start a gap analysis job on the
backend -> poll for progress -> show the results as a color-coded table ->
optionally download a PDF gap analysis report.
"""

import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SAMA Compliance Auditor", page_icon="🛡️", layout="wide")

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

STATUS_COLORS = {
    "PASS": "#d4edda",
    "PARTIAL": "#fff3cd",
    "FAIL": "#f8d7da",
}


def _style_by_status(df: pd.DataFrame, status_col: pd.Series):
    """Row-wise background color based on a status_code series aligned by index."""
    def _row_style(row):
        color = STATUS_COLORS.get(status_col.loc[row.name], "")
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(_row_style, axis=1)


with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL).rstrip("/")
    language = st.selectbox("SAMA CSF language", options=["en", "ar"], index=0)
    top_k = st.number_input("Retrieved excerpts per control (top_k)", min_value=1, max_value=10, value=3)
    company_name = st.text_input("Company name (for the PDF report)", value="")

    st.divider()
    if st.button("Check backend connection"):
        try:
            r = requests.get(f"{backend_url}/api/health", timeout=5)
            if r.ok:
                st.success("Backend is reachable.")
            else:
                st.error(f"Backend responded with status {r.status_code}.")
        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")

st.title("🛡️ SAMA Compliance Auditor")
st.caption("Compares your uploaded security evidence against the SAMA Cyber Security Framework (CSF).")

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "results" not in st.session_state:
    st.session_state.results = None

uploaded_files = st.file_uploader(
    "Upload evidence (policies, procedures, reports, or screenshots)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    accept_multiple_files=True,
)

run_clicked = st.button("Run Gap Analysis", type="primary", disabled=not uploaded_files)

if run_clicked and uploaded_files:
    files_payload = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    data = {"language": language, "top_k": str(top_k)}
    try:
        resp = requests.post(f"{backend_url}/api/analyze", files=files_payload, data=data, timeout=30)
        resp.raise_for_status()
        st.session_state.job_id = resp.json()["job_id"]
        st.session_state.results = None
    except requests.RequestException as e:
        st.error(f"Could not start analysis: {e}")
        st.session_state.job_id = None

if st.session_state.job_id and st.session_state.results is None:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    while True:
        try:
            r = requests.get(f"{backend_url}/api/analyze/{st.session_state.job_id}", timeout=10)
            r.raise_for_status()
            job = r.json()
        except requests.RequestException as e:
            st.error(f"Lost connection to backend while polling: {e}")
            break

        if job["status"] == "error":
            st.error(f"Analysis failed: {job['error']}")
            st.session_state.job_id = None
            break

        if job["status"] == "done":
            progress_bar.progress(1.0)
            status_text.text("Done.")
            st.session_state.results = job["results"]
            break

        total = job["total"] or 1
        done = job["done"]
        progress_bar.progress(min(done / total, 1.0))
        status_text.text(f"Judging control {job['current_control'] or '...'} ({done}/{total})")
        time.sleep(2)

if st.session_state.results:
    results = st.session_state.results
    df = pd.DataFrame(results)

    counts = df["status_code"].value_counts().to_dict()
    total = len(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Compliant", counts.get("PASS", 0), f"{counts.get('PASS', 0) / total:.0%}")
    col2.metric("Partially Compliant", counts.get("PARTIAL", 0), f"{counts.get('PARTIAL', 0) / total:.0%}")
    col3.metric("Non-Compliant", counts.get("FAIL", 0), f"{counts.get('FAIL', 0) / total:.0%}")

    st.subheader("Detailed Findings")
    display_cols = ["control_id", "control_domain", "status_label", "justification", "recommendation"]
    df_display = df[display_cols]
    styled = _style_by_status(df_display, df["status_code"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Report")
    if st.button("Generate PDF Report"):
        try:
            payload = {"job_id": st.session_state.job_id, "results": results, "company_name": company_name or None}
            r = requests.post(f"{backend_url}/api/report", json=payload, timeout=60)
            r.raise_for_status()
            st.download_button(
                "Download gap_analysis_report.pdf",
                data=r.content,
                file_name="gap_analysis_report.pdf",
                mime="application/pdf",
            )
        except requests.RequestException as e:
            st.error(f"Could not generate report: {e}")

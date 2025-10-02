

from __future__ import annotations
import io
import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List

import streamlit as st

st.set_page_config(page_title="PDF → JSON (via extract_debates.py)", page_icon="📄", layout="centered")
st.title("📄 PDF → JSON Extractor")
st.caption("pdf file from hansard.")


HERE = Path(__file__).resolve().parent
EXTRACTOR_SRC = HERE / "extract_debates.py"

if not EXTRACTOR_SRC.exists():
    st.error("Could not find `extract_debates.py` next to this app. "
             "Please ensure both files are in the same folder.")
    st.stop()

uploaded = st.file_uploader(
    "Drag & drop a PDF here, or click to browse",
    type=["pdf"],
    accept_multiple_files=False,
    help="Your file will be processed in a temporary sandbox folder."
)

st.markdown(
    "> runs `python extract_debates.py` in a temporary working folder. "
    "JSON script will appear below for download or in the main folder of the document."
)

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

st.write(f"**Selected file:** {uploaded.name}")


workdir = Path(tempfile.mkdtemp(prefix="extract_"))
st.write(f"Working in: `{workdir}`")


pdf_bytes = uploaded.read()
(doc_path := workdir / "document").write_bytes(pdf_bytes)
(doc_pdf_path := workdir / "document.pdf").write_bytes(pdf_bytes)


shutil.copy2(EXTRACTOR_SRC, workdir / "extract_debates.py")


before_files = {p.name for p in workdir.iterdir()}


py_exe = sys.executable or "python"
cmd = [py_exe, "-u", "extract_debates.py"]
st.code(" ".join(cmd), language="bash")

with st.spinner("Running your extractor…"):
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True
    )


st.subheader("Logs")
logs = ""
if proc.stdout:
    logs += f"STDOUT:\n{proc.stdout}\n"
if proc.stderr:
    logs += f"\nSTDERR:\n{proc.stderr}\n"
if not logs.strip():
    logs = "(No output)"
st.text_area("Process output", value=logs, height=240)


after_files = {p.name for p in workdir.iterdir()}
new_files = [f for f in (after_files - before_files) if f.lower().endswith(".json")]
new_files_paths = [workdir / f for f in sorted(new_files)]

if proc.returncode != 0:
    st.error(f"Extractor exited with code {proc.returncode}. "
             "If no JSON was produced, check the logs above.")
elif not new_files_paths:
    st.warning("The script finished, but no new .json files were found in the working folder. "
               "Check the logs above to see where it saved output (if any).")
else:
    st.success(f"Done! Found {len(new_files_paths)} JSON file(s).")


for path in new_files_paths:
    st.markdown(f"**Output:** `{path.name}`")
    try:
        data = path.read_bytes()
        st.download_button(
            label=f"⬇️ Download {path.name}",
            data=data,
            file_name=path.name,
            mime="application/json"
        )
        
        try:
            preview = path.read_text(encoding="utf-8", errors="ignore")
            if len(preview) > 1000:
                preview = preview[:1000] + "…"
            with st.expander(f"Preview {path.name}"):
                st.code(preview, language="json")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")

st.caption("Temporary working folder will be cleaned up automatically by the OS.")

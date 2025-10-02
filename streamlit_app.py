
import os, io, json, importlib, tempfile, time, inspect
import streamlit as st


os.environ.setdefault("PYTORCH_SDP_DISABLE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

st.set_page_config(page_title="PDF → JSON Extractor", layout="wide")
st.title("PDF → JSON Extractor")
st.caption("Drag & drop a PDF. This wraps your existing `extract_debates.py` without modifying it.")

@st.cache_resource(show_spinner=False)
def load_core():
    core = importlib.import_module("extract_debates")  
    
    if hasattr(core, "process_pdf_to_json"):
        return core, "to_json_file"
    if hasattr(core, "process_pdf"):
        sig = inspect.signature(core.process_pdf)
        
        n_pos = sum(1 for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty)
        if n_pos >= 2:
            return core, "process_pdf_writes_file"
        return core, "process_pdf_returns_dict"
    raise RuntimeError(
        "Expected `extract_debates.py` to expose either:\n"
        "  - process_pdf_to_json(pdf_path, out_path)\n"
        "  - process_pdf(pdf_path, out_path)    # writes JSON file\n"
        "  - process_pdf(pdf_path)              # returns a dict"
    )

core, mode = load_core()

file = st.file_uploader("Drop a PDF here", type=["pdf"])
go = st.button("Extract", use_container_width=True, disabled=not file)

if go and file:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, file.name)
        with open(pdf_path, "wb") as f:
            f.write(file.read())

        t0 = time.time()
        with st.spinner("Extracting… (first run downloads models)"):
            if mode == "to_json_file":
                out_path = os.path.join(td, "output.json")
                core.process_pdf_to_json(pdf_path, out_path)
                json_bytes = open(out_path, "rb").read()
            elif mode == "process_pdf_writes_file":
                out_path = os.path.join(td, "output.json")
                core.process_pdf(pdf_path, out_path)
                json_bytes = open(out_path, "rb").read()
            else:  
                data = core.process_pdf(pdf_path)
                json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        dt = time.time() - t0

    st.success(f"Done in {dt:.1f}s")

    c1, c2 = st.columns([2,1], gap="large")
    with c1:
        st.subheader("Preview")
        try:
            st.json(json.loads(json_bytes.decode("utf-8", errors="ignore")), expanded=False)
        except Exception as e:
            st.warning(f"Could not render JSON preview: {e}")
        st.caption(f"Size: ~{len(json_bytes)/1024:.1f} KB")

    with c2:
        st.subheader("Download")
        st.download_button(
            "Download output.json",
            data=json_bytes,
            file_name="output.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown("---")
with st.expander("Notes"):
    st.write(
        "- First run is slower while models download.\n"
        "- This UI **does not modify** your core script; it just imports it."
    )

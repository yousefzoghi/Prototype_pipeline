"""
EEG Analysis Dashboard — main entry point.

Run with:  streamlit run Dashboard.py
"""

import streamlit as st
import base64
from pathlib import Path

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared sidebar (CSS + footer) ─────────────────────────────────────────
from utils.sidebar import render_sidebar, render_sidebar_footer
from utils.signal_processing import DEFAULT_MAPPING, apply_standard_montage

render_sidebar()

# ── Sidebar: file uploader ─────────────────────────────────────────────────
from utils.data_loader import load_file, EEGRecording, SurveyData

uploaded_files = st.sidebar.file_uploader(
    "Upload EEG / Survey files",
    type=["csv", "txt", "xlsx"],
    accept_multiple_files=True,
    help="Supported: OpenBCI raw CSV/TXT, BrainFlow CSV, Participant survey XLSX",
)

# Process uploads
if uploaded_files:
    if "recordings" not in st.session_state:
        st.session_state.recordings = {}
    if "surveys" not in st.session_state:
        st.session_state.surveys = {}

    for uf in uploaded_files:
        if uf.name not in st.session_state.recordings and uf.name not in st.session_state.surveys:
            import tempfile, os
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            try:
                result = load_file(tmp_path)
                if isinstance(result, EEGRecording):
                    st.session_state.recordings[uf.name] = result
                elif isinstance(result, SurveyData):
                    st.session_state.surveys[uf.name] = result
            except Exception as e:
                st.sidebar.error(f"Error loading {uf.name}: {e}")

# Active recording selector
recordings = st.session_state.get("recordings", {})
surveys = st.session_state.get("surveys", {})

active_rec = None
if recordings:
    st.sidebar.markdown(
        '<p style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; '
        'letter-spacing:0.06em; font-weight:600; margin-bottom:4px;">Loaded Recordings</p>',
        unsafe_allow_html=True,
    )
    rec_name = st.sidebar.selectbox("Active recording", list(recordings.keys()), label_visibility="collapsed")
    
    # If the recording has changed, clear the processed data from session state
    if st.session_state.get("active_rec_name") != rec_name:
        st.session_state["active_rec_name"] = rec_name
        if "raw_filtered" in st.session_state:
            del st.session_state["raw_filtered"]
        if "ica" in st.session_state:
            del st.session_state["ica"]
        if "ica_source" in st.session_state:
            del st.session_state["ica_source"]
            
    active_rec = recordings[rec_name]
    st.session_state["active_rec_name"] = rec_name

if surveys:
    st.sidebar.markdown(
        '<p style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; '
        'letter-spacing:0.06em; font-weight:600; margin-bottom:4px;">Loaded Surveys</p>',
        unsafe_allow_html=True,
    )
    for sname in surveys:
        st.sidebar.markdown(
            f'<div style="background:#0d1f0d; border:1px solid #238636; border-radius:6px; '
            f'padding:6px 10px; margin-bottom:4px; font-size:0.85rem; color:#3fb950;">'
            f'✓ {sname}</div>',
            unsafe_allow_html=True,
        )

# Store in session for pages
st.session_state["active_rec"] = active_rec

# ── Sidebar footer — credit (shown on all pages) ──────────────────────────
render_sidebar_footer()


# ── Main page: Logo + Title + Overview ─────────────────────────────────────

# Load logo from assets if it exists, otherwise fallback to simple emoji or text
logo_path = Path("assets/eeg2.png")
if logo_path.exists():
    img_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    img_src = f"data:image/png;base64,{img_b64}"
else:
    # Fallback to a placeholder or simple graphic if file missing
    img_src = ""

# Reduce Streamlit's default top padding
st.markdown(
    "<style>.block-container { padding-top: 1rem !important; }</style>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="text-align:center; padding: 2.5rem 0 0.8rem;">
        <img src="{img_src}" width="100"
             alt="brain icon" style="display:inline-block; margin-bottom: 8px;" />
        <h1 style="font-size: 2.2rem; font-weight: 700; color: #58a6ff;
                   letter-spacing: 0.02em; margin: 0;">EEG Analysis Dashboard</h1>
        <p style="font-size: 1rem; color: #6e7681; margin-top: 4px;
                  font-style: italic; margin-bottom: 0;">
            Interactive EEG Visualization &amp; Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    A comprehensive platform for exploring, preprocessing, and analyzing raw EEG
    recordings. Upload OpenBCI or BrainFlow data files alongside participant surveys
    to visualize multi-channel signals, compute spectral power, generate topographic
    maps, detect anomalies with ML/DL models, and produce AI-driven insights — all
    from a single interactive dashboard.
    """
)

st.markdown("---")

if active_rec is None and not surveys:
    st.info("Upload one or more EEG data files or survey files using the sidebar to get started.")
    st.markdown(
        """
        **Supported formats:**
        - OpenBCI raw CSV / TXT
        - BrainFlow raw CSV
        - Participant survey XLSX
        """
    )
else:
    if active_rec:
        st.header("Recording Overview")
        meta = active_rec.metadata

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Format", meta.get("format", "N/A").upper())
        col2.metric("Channels", len(active_rec.ch_names))
        col3.metric("Sample Rate", f"{active_rec.sfreq:.0f} Hz")
        col4.metric("Duration", f"{meta.get('duration_s', 0):.1f} s")

        st.markdown("---")

        # Channel list
        with st.expander("Channel Names", expanded=False):
            cols = st.columns(4)
            for i, ch in enumerate(active_rec.ch_names):
                cols[i % 4].code(ch)

        # Data preview
        with st.expander("Data Preview (first 20 rows)", expanded=False):
            st.dataframe(active_rec.data.head(20), use_container_width=True)

        # Quick stats
        with st.expander("Quick Statistics", expanded=True):
            stats = active_rec.data[active_rec.ch_names].describe().T
            st.dataframe(stats, use_container_width=True)

    if surveys:
        st.header("Survey Overview")
        for sname, survey in surveys.items():
            st.subheader(survey.participant_id)
            if not survey.raw_df.empty:
                st.dataframe(survey.raw_df, use_container_width=True)

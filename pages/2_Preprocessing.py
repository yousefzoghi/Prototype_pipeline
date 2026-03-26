"""
Preprocessing — Filtering, ICA artifact rejection, and before/after comparison.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Preprocessing", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Preprocessing")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from utils.signal_processing import apply_bandpass, apply_notch

# Build MNE raw
raw = rec.build_mne_raw()

# ── Filtering Options ──────────────────────────────────────────────────────
st.header("Filtering")
col1, col2, col3 = st.columns(3)

with col1:
    low_freq = st.number_input("Bandpass low cutoff (Hz)", min_value=0.1, max_value=50.0, value=1.0, step=0.5)
with col2:
    high_freq = st.number_input("Bandpass high cutoff (Hz)", min_value=1.0, max_value=200.0, value=50.0, step=1.0)
with col3:
    notch_freq = st.selectbox("Notch filter", options=["None", "50 Hz", "60 Hz", "50 + 60 Hz"])

apply_filter = st.button("Apply Filters", type="primary")

if apply_filter:
    with st.spinner("Applying bandpass filter..."):
        raw_filtered = apply_bandpass(raw, l_freq=low_freq, h_freq=high_freq)

    if notch_freq != "None":
        with st.spinner("Applying notch filter..."):
            freqs = {"50 Hz": 50.0, "60 Hz": 60.0, "50 + 60 Hz": [50.0, 60.0]}
            raw_filtered = apply_notch(raw_filtered, freqs=freqs[notch_freq])

    st.session_state["raw_filtered"] = raw_filtered
    st.success("Filters applied successfully!")

# ── Before / After Comparison ──────────────────────────────────────────────
raw_filtered = st.session_state.get("raw_filtered")

if raw_filtered is not None:
    st.header("Before / After Comparison")

    ch_compare = st.selectbox("Channel", options=rec.ch_names, index=0)
    view_seconds = st.slider("View window (s)", 1, 10, 5)

    ch_idx = rec.ch_names.index(ch_compare)
    samples = int(view_seconds * rec.sfreq)

    data_before = raw.get_data(picks=ch_idx)[0, :samples] * 1e6  # V → µV
    data_after = raw_filtered.get_data(picks=ch_idx)[0, :samples] * 1e6
    time = np.arange(samples) / rec.sfreq

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Before Filtering", "After Filtering"],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scattergl(x=time, y=data_before, mode="lines",
                               line=dict(color="#ff6b6b", width=0.8), name="Before"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=time, y=data_after, mode="lines",
                               line=dict(color="#51cf66", width=0.8), name="After"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500,
                      xaxis2_title="Time (s)", margin=dict(l=60, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # ── PSD Comparison ───────────────────────────────────────────────────
    with st.expander("📊 Power Spectral Density Comparison", expanded=True):
        st.markdown("Verify that line noise (50/60 Hz) and DC offsets have been removed.")
        
        from utils.signal_processing import compute_psd
        
        psd_col1, psd_col2 = st.columns(2)
        with psd_col1:
            fmax_view = st.slider("Max frequency to view (Hz)", 10, 100, 60)
        
        with st.spinner("Computing PSD comparison..."):
            # Compute PSD for original and filtered
            # Use a slightly larger n_fft for better resolution in this view
            psds_before, freqs_before = compute_psd(raw.copy().pick(ch_idx), fmin=0.5, fmax=fmax_view, n_fft=512)
            psds_after, freqs_after = compute_psd(raw_filtered.copy().pick(ch_idx), fmin=0.5, fmax=fmax_view, n_fft=512)
            
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(
                x=freqs_before, y=10 * np.log10(psds_before[0] + 1e-20),
                mode="lines", name="Before (Raw)", line=dict(color="#ff6b6b", width=1.5)
            ))
            fig_psd.add_trace(go.Scatter(
                x=freqs_after, y=10 * np.log10(psds_after[0] + 1e-20),
                mode="lines", name="After (Filtered)", line=dict(color="#51cf66", width=2)
            ))
            
            fig_psd.update_layout(
                template="plotly_dark", height=400,
                xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
                title=f"PSD Comparison — {ch_compare}",
                margin=dict(l=60, r=20, t=40, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_psd, use_container_width=True)

# ── ICA ────────────────────────────────────────────────────────────────────
st.header("ICA Artifact Rejection")
st.markdown("Decompose the signal into independent components to identify and remove artifacts.")

source = raw_filtered if raw_filtered is not None else raw

n_components = st.slider("Number of ICA components", min_value=2, max_value=min(16, len(rec.ch_names)), value=min(10, len(rec.ch_names)))

if st.button("Run ICA", type="primary"):
    import mne
    with st.spinner("Running ICA decomposition (this may take a moment)..."):
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
        ica.fit(source, verbose=False)
        st.session_state["ica"] = ica
        st.session_state["ica_source"] = source
    st.success(f"ICA fit complete with {n_components} components.")

ica = st.session_state.get("ica")
if ica is not None:
    source_for_ica = st.session_state.get("ica_source", source)

    # Show component time series
    sources = ica.get_sources(source_for_ica)
    source_data = sources.get_data() * 1e6
    n_show = min(n_components, source_data.shape[0])
    view_s = min(5, int(source_for_ica.times[-1]))
    n_samp = int(view_s * rec.sfreq)
    t = np.arange(n_samp) / rec.sfreq

    fig_ica = go.Figure()
    for i in range(n_show):
        fig_ica.add_trace(go.Scattergl(
            x=t, y=source_data[i, :n_samp] + i * 200,
            mode="lines", name=f"IC {i}", line=dict(width=0.7)
        ))
    fig_ica.update_layout(template="plotly_dark", height=max(300, n_show * 50),
                          xaxis_title="Time (s)", yaxis_title="Component amplitude",
                          margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig_ica, use_container_width=True)

    # Exclude components
    exclude = st.multiselect("Components to exclude (artifacts)", options=list(range(n_show)))

    if st.button("Apply ICA Exclusion"):
        ica.exclude = exclude
        raw_cleaned = ica.apply(source_for_ica.copy(), verbose=False)
        st.session_state["raw_filtered"] = raw_cleaned
        st.success(f"Excluded components {exclude}. Cleaned data stored.")

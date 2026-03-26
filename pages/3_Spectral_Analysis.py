"""
Spectral Analysis — PSD, band power, spectrograms, and wavelet scalograms.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import signal as sp_signal

st.set_page_config(page_title="Spectral Analysis", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Spectral Analysis")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from utils.signal_processing import compute_psd, compute_band_powers, FREQ_BANDS

# ── Data Selection ────────────────────────────────────────────────────────
raw_filtered = st.session_state.get("raw_filtered")

if raw_filtered is not None:
    raw = raw_filtered
    st.success("✨ Currently using **post-filtered** signals (from Preprocessing page).")
else:
    raw = rec.build_mne_raw()
    st.warning("⚠️ Currently using **raw** EEG signals. It is highly recommended to apply filters on the [Preprocessing](./Preprocessing) page first to remove noise and DC offsets.")

# ── PSD ────────────────────────────────────────────────────────────────────
st.header("Power Spectral Density")

col1, col2, col3 = st.columns(3)
with col1:
    n_fft = st.select_slider("FFT size", options=[64, 128, 256, 512, 1024], value=256)
with col2:
    fmin_psd = st.number_input("Min freq (Hz)", min_value=0.1, value=0.5, step=0.5)
with col3:
    fmax_psd = st.number_input("Max freq (Hz)", min_value=1.0, value=60.0, step=5.0)

ch_psd = st.multiselect("Channels for PSD", options=rec.ch_names, default=rec.ch_names[:4])

if ch_psd:
    with st.spinner("Computing PSD..."):
        raw_pick = raw.copy().pick(ch_psd)
        psds, freqs = compute_psd(raw_pick, fmin=fmin_psd, fmax=fmax_psd, n_fft=n_fft)

    fig_psd = go.Figure()
    for i, ch in enumerate(ch_psd):
        fig_psd.add_trace(go.Scatter(
            x=freqs, y=10 * np.log10(psds[i] + 1e-20),
            mode="lines", name=ch, line=dict(width=1.5),
        ))

    # Add band shading
    colors = ["rgba(100,149,237,0.1)", "rgba(144,238,144,0.1)", "rgba(255,215,0,0.1)",
              "rgba(255,165,0,0.1)", "rgba(255,99,71,0.1)"]
    for idx, (band_name, (f1, f2)) in enumerate(FREQ_BANDS.items()):
        if f1 < fmax_psd:
            fig_psd.add_vrect(x0=f1, x1=min(f2, fmax_psd), fillcolor=colors[idx],
                              layer="below", line_width=0,
                              annotation_text=band_name.split(" ")[0], annotation_position="top left")

    fig_psd.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)",
        margin=dict(l=60, r=20, t=40, b=60),
    )
    st.plotly_chart(fig_psd, use_container_width=True)

    # ── Band Power Bar Chart ───────────────────────────────────────────────
    st.header("Band Power Distribution")
    bp = compute_band_powers(psds, freqs)
    bp.index = ch_psd

    fig_bp = go.Figure()
    band_colors = ["#6495ED", "#90EE90", "#FFD700", "#FFA500", "#FF6347"]
    for i, band in enumerate(bp.columns):
        fig_bp.add_trace(go.Bar(
            x=bp.index, y=bp[band], name=band,
            marker_color=band_colors[i],
        ))

    fig_bp.update_layout(
        template="plotly_dark", height=400,
        barmode="group",
        xaxis_title="Channel", yaxis_title="Mean Power (V²/Hz)",
        margin=dict(l=60, r=20, t=40, b=60),
    )
    st.plotly_chart(fig_bp, use_container_width=True)

# ── Spectrogram ────────────────────────────────────────────────────────────
st.header("Time-Frequency Spectrogram")

ch_spec = st.selectbox("Channel for spectrogram", options=rec.ch_names, index=0)

spec_col1, spec_col2 = st.columns(2)
with spec_col1:
    nperseg = st.select_slider("Segment length", options=[64, 128, 256, 512], value=128)
with spec_col2:
    noverlap_pct = st.slider("Overlap %", 0, 90, 75, step=5)

ch_idx = rec.ch_names.index(ch_spec)
data_ch = raw.get_data(picks=ch_idx)[0]
noverlap = int(nperseg * noverlap_pct / 100)

with st.spinner("Computing spectrogram..."):
    f_spec, t_spec, Sxx = sp_signal.spectrogram(
        data_ch, fs=rec.sfreq, nperseg=nperseg, noverlap=noverlap
    )

# Limit to 60 Hz for display
freq_mask = f_spec <= 60
fig_spec = go.Figure(data=go.Heatmap(
    z=10 * np.log10(Sxx[freq_mask, :] + 1e-20),
    x=t_spec,
    y=f_spec[freq_mask],
    colorscale="Viridis",
    colorbar=dict(title="dB"),
))

fig_spec.update_layout(
    template="plotly_dark", height=400,
    xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
    title=f"Spectrogram — {ch_spec}",
    margin=dict(l=60, r=20, t=60, b=60),
)
st.plotly_chart(fig_spec, use_container_width=True)

# ── Wavelet Scalogram ─────────────────────────────────────────────────────
st.header("Morlet Wavelet Scalogram")

wav_col1, wav_col2 = st.columns(2)
with wav_col1:
    wav_fmin = st.number_input("Min freq (Hz)", min_value=0.5, value=1.0, step=0.5, key="wav_fmin")
with wav_col2:
    wav_fmax = st.number_input("Max freq (Hz)", min_value=1.0, value=40.0, step=1.0, key="wav_fmax")

wav_freqs = np.linspace(wav_fmin, wav_fmax, 50)
# Use first 10 seconds for scalogram (to keep it manageable)
max_samps = int(10 * rec.sfreq)
sig_short = data_ch[:max_samps]
t_wav = np.arange(len(sig_short)) / rec.sfreq

with st.spinner("Computing wavelet transform..."):
    # Manual Morlet CWT (scipy.signal.cwt was removed in newer scipy)
    n = len(sig_short)
    cwt_matrix = np.zeros((len(wav_freqs), n))
    for i, freq in enumerate(wav_freqs):
        # Morlet wavelet parameters
        w = 6  # omega0
        s = w * rec.sfreq / (2 * np.pi * freq)
        # Create wavelet in time domain
        t_wav_kernel = np.arange(-int(4 * s), int(4 * s) + 1) / rec.sfreq
        wavelet = np.exp(1j * 2 * np.pi * freq * t_wav_kernel) * np.exp(-t_wav_kernel**2 * freq**2 / (2 * (w / (2 * np.pi))**2))
        wavelet = wavelet / np.sqrt(s)
        # Convolve via FFT
        conv = np.convolve(sig_short, wavelet, mode="same")
        cwt_matrix[i, :] = np.abs(conv)

fig_wav = go.Figure(data=go.Heatmap(
    z=10 * np.log10(cwt_matrix ** 2 + 1e-20),
    x=t_wav,
    y=wav_freqs,
    colorscale="Magma",
    colorbar=dict(title="dB"),
))

fig_wav.update_layout(
    template="plotly_dark", height=400,
    xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
    title=f"Wavelet Scalogram — {ch_spec} (first {len(sig_short)/rec.sfreq:.0f} s)",
    margin=dict(l=60, r=20, t=60, b=60),
)
st.plotly_chart(fig_wav, use_container_width=True)

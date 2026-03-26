"""
Topography — MNE topographic maps of band power.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

st.set_page_config(page_title="Topography", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Topography")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from utils.signal_processing import FREQ_BANDS

# ── Data Selection ────────────────────────────────────────────────────────
raw_filtered = st.session_state.get("raw_filtered")

if raw_filtered is not None:
    raw = raw_filtered
    st.success("✨ Currently using **post-filtered** signals (from Preprocessing page).")
else:
    raw = rec.build_mne_raw()
    st.warning("⚠️ Currently using **raw** EEG signals. Topomaps may be noisy without bandpass filtering.")

# ── Montage Selection ──────────────────────────────────────────────────────
st.header("Montage Configuration")

# Standard 10-20 channel mapping for OpenBCI Cyton+Daisy 16-channel
DEFAULT_MAPPING = {
    "EXG Channel 0": "Fp1", "EXG Channel 1": "Fp2",
    "EXG Channel 2": "C3",  "EXG Channel 3": "C4",
    "EXG Channel 4": "P7",  "EXG Channel 5": "P8",
    "EXG Channel 6": "O1",  "EXG Channel 7": "O2",
    "EXG Channel 8": "F7",  "EXG Channel 9": "F8",
    "EXG Channel 10": "F3", "EXG Channel 11": "F4",
    "EXG Channel 12": "T7", "EXG Channel 13": "T8",
    "EXG Channel 14": "P3", "EXG Channel 15": "P4",
}

use_custom = st.checkbox("Use custom 10-20 mapping", value=True)

if use_custom:
    st.info("Default mapping assumes OpenBCI Cyton+Daisy. Adjust as needed.")

    # Let user edit the mapping
    mapping = {}
    cols = st.columns(4)
    for i, ch in enumerate(rec.ch_names):
        default_name = DEFAULT_MAPPING.get(ch, ch)
        new_name = cols[i % 4].text_input(f"{ch} →", value=default_name, key=f"map_{ch}")
        mapping[ch] = new_name

    # Apply rename and montage
    try:
        raw_topo = raw.copy()
        rename_dict = {old: new for old, new in mapping.items() if old in raw_topo.ch_names}
        raw_topo.rename_channels(rename_dict)

        montage = mne.channels.make_standard_montage("standard_1020")
        # Keep only channels that exist in the montage
        valid_chs = [ch for ch in raw_topo.ch_names if ch in montage.ch_names]
        if valid_chs:
            raw_topo.pick(valid_chs)
            raw_topo.set_montage(montage, on_missing="ignore")
        else:
            st.error("No channels match the standard 10-20 montage. Check your mapping.")
            st.stop()
    except Exception as e:
        st.error(f"Error setting montage: {e}")
        st.stop()
else:
    raw_topo = raw.copy()
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_topo.set_montage(montage, on_missing="ignore")
    except Exception:
        st.warning("Could not auto-apply montage. Channel names may not match standard 10-20.")
        st.stop()

# ── Band Power Topomaps ────────────────────────────────────────────────────
st.header("Band Power Topomaps")

topo_col1, topo_col2 = st.columns(2)
with topo_col1:
    topo_tmin = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
with topo_col2:
    topo_tmax = st.number_input("End time (s)", min_value=0.1,
                                value=min(10.0, raw_topo.times[-1]), step=1.0)

# Crop to selected window
raw_crop = raw_topo.copy().crop(tmin=topo_tmin, tmax=min(topo_tmax, raw_topo.times[-1]))

# Compute PSD for each band
try:
    nyquist = raw_crop.info["sfreq"] / 2.0
    spectrum = raw_crop.compute_psd(method="welch", fmin=0.5, fmax=nyquist - 0.5, n_fft=256, verbose=False)
    psds = spectrum.get_data()
    freqs = spectrum.freqs

    fig, axes = plt.subplots(1, len(FREQ_BANDS), figsize=(4 * len(FREQ_BANDS), 4))
    fig.patch.set_facecolor("#0d1117")

    for idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psds[:, freq_idx], axis=1)

        ax = axes[idx] if len(FREQ_BANDS) > 1 else axes
        mne.viz.plot_topomap(
            band_power, raw_crop.info, axes=ax, show=False,
            cmap="RdYlBu_r", contours=4,
        )
        short_name = band_name.split(" ")[0]
        ax.set_title(short_name, color="white", fontsize=12, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.error(f"Error computing topomaps: {e}")
    st.info("Ensure channel names match a standard montage for topomap visualization.")

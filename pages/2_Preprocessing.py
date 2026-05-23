"""
Preprocessing — Filtering, Artifact Rejection (ICA, WPT, WAT, ASR), and before/after comparison.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import mne
import matplotlib
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import kurtosis

from utils.sidebar import render_sidebar, render_sidebar_footer
from utils.signal_processing import (
    apply_custom_filter, 
    apply_notch, 
    apply_asr,
    compute_psd, 
    apply_wpt_denoising, 
    apply_wat_denoising,
    DEFAULT_MAPPING,
    apply_standard_montage,
    extract_3d_features,
    make_fixed_epochs,
    get_epoch_tags
)

st.set_page_config(page_title="Preprocessing", layout="wide")
render_sidebar()
render_sidebar_footer()

st.title("Preprocessing")

# ── Reset Button ──────────────────────────────────────────────────────────
if st.button("🔄 Reset All Preprocessing", type="secondary", use_container_width=True):
    for key in ["raw_filtered", "raw_asr", "ica_obj", "ica_labels", "filter_comparison"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

# Build MNE raw if not already in session
if "raw_base" not in st.session_state:
    st.session_state["raw_base"] = rec.build_mne_raw()

raw = st.session_state["raw_base"]

# ── Visual Verification ────────────────────────────────────────────────────
st.divider()
st.header("Visual Verification")
st.markdown("Compare the **Original** data with the **Processed** state after current steps.")

raw_final = st.session_state.get("raw_filtered", raw)

vcol1, vcol2 = st.columns([1, 2])
with vcol1:
    ch_compare = st.selectbox("Channel to inspect", options=rec.ch_names, index=0, key="inspect_ch", 
                            format_func=lambda x: DEFAULT_MAPPING.get(x, x))

with vcol2:
    t_max_limit = float(rec.metadata.get("duration_s", len(rec.data)/rec.sfreq))
    # Range slider for start and end time
    time_selection = st.slider("Select time range (s)", 0.0, t_max_limit, (0.0, min(5.0, t_max_limit)), step=0.1, key="inspect_time_range")
    start_time, end_time = time_selection

ch_idx = rec.ch_names.index(ch_compare)

# Calculate indices
start_sample = int(start_time * rec.sfreq)
end_sample = int(end_time * rec.sfreq)
# Ensure at least some samples are selected
if end_sample <= start_sample:
    end_sample = start_sample + 1
end_sample = min(end_sample, int(t_max_limit * rec.sfreq))

# Slice the data
t = np.arange(start_sample, end_sample) / rec.sfreq
data_orig = raw.get_data(picks=ch_idx)[0, start_sample:end_sample] * 1e6
data_filt = raw_final.get_data(picks=ch_idx)[0, start_sample:end_sample] * 1e6

ch_display_name = DEFAULT_MAPPING.get(ch_compare, ch_compare)

fig = go.Figure()

# Original trace with lower opacity for background comparison
fig.add_trace(go.Scattergl(
    x=t, y=data_orig, 
    mode="lines", 
    name="Original", 
    line=dict(color="#ff6b6b", width=1.2),
    opacity=0.5
))

# Processed trace on top
fig.add_trace(go.Scattergl(
    x=t, y=data_filt, 
    mode="lines", 
    name="Processed", 
    line=dict(color="#51cf66", width=1.5)
))

fig.update_layout(
    title=f"Verification: Original vs Processed ({ch_display_name})",
    template="plotly_dark", 
    height=500, 
    xaxis_title="Time (s)", 
    yaxis_title="Amplitude (µV)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ── 1. Bandpass Filtering ────────────────────────────────────────────────
with st.expander("1. Bandpass Filtering", expanded=True):
    st.subheader("Filter Options")
    
    # Header row
    hcol1, hcol2, hcol3, hcol4 = st.columns([2, 1, 1, 3])
    hcol1.write("**Filter Type**")
    hcol2.write("**Low (Hz)**")
    hcol3.write("**High (Hz)**")
    hcol4.write("**Order**")

    # FIR Row
    r1c1, r1c2, r1c3, r1c4 = st.columns([2, 1, 1, 3])
    r1c1.markdown("FIR (Finite Impulse Response)")
    fir_low = r1c2.number_input("Low", value=1.0, key="fir_l", label_visibility="collapsed")
    fir_high = r1c3.number_input("High", value=50.0, key="fir_h", label_visibility="collapsed")
    fir_order = r1c4.slider("FIR Order", 0, 10, 0, help="0 for MNE auto-selection", key="fir_o_s", label_visibility="collapsed")

    # Butterworth Row
    r2c1, r2c2, r2c3, r2c4 = st.columns([2, 1, 1, 3])
    r2c1.markdown("Butterworth (IIR)")
    butt_low = r2c2.number_input("Low", value=1.0, key="butt_l", label_visibility="collapsed")
    butt_high = r2c3.number_input("High", value=50.0, key="butt_h", label_visibility="collapsed")
    butt_order = r2c4.slider("Butt Order", 1, 12, 4, key="butt_o_s", label_visibility="collapsed")

    # Bessel Row
    r3c1, r3c2, r3c3, r3c4 = st.columns([2, 1, 1, 3])
    r3c1.markdown("Bessel (IIR)")
    bess_low = r3c2.number_input("Low", value=1.0, key="bess_l", label_visibility="collapsed")
    bess_high = r3c3.number_input("High", value=50.0, key="bess_h", label_visibility="collapsed")
    bess_order = r3c4.slider("Bess Order", 1, 12, 4, key="bess_o_s", label_visibility="collapsed")

    # Chebyshev Row
    r4c1, r4c2, r4c3, r4c4 = st.columns([2, 1, 1, 3])
    r4c1.markdown("Chebyshev Type I (IIR)")
    cheb_low = r4c2.number_input("Low", value=1.0, key="cheb_l", label_visibility="collapsed")
    cheb_high = r4c3.number_input("High", value=50.0, key="cheb_h", label_visibility="collapsed")
    cheb_order = r4c4.slider("Cheb Order", 1, 12, 4, key="cheb_o_s", label_visibility="collapsed")

    st.divider()
    st.subheader("Comparison Chart Controls")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        comp_channel = st.selectbox("Channel to compare", options=raw.ch_names, format_func=lambda x: DEFAULT_MAPPING.get(x, x))
    with ccol2:
        t_max_limit = float(raw.times[-1])
        comp_time_range = st.slider("Time window (s)", 0.0, t_max_limit, (0.0, min(5.0, t_max_limit)))

    if st.button("Generate Comparison Chart", use_container_width=True):
        with st.spinner("Processing filter comparisons..."):
            ch_idx = raw.ch_names.index(comp_channel)
            t_start, t_end = comp_time_range
            if t_end <= t_start: t_end = t_start + 0.1
            
            raw_crop = raw.copy().crop(tmin=t_start, tmax=t_end).pick(ch_idx)
            
            results = {
                "Original": raw_crop.get_data()[0] * 1e6,
                "FIR": apply_custom_filter(raw_crop, fir_low, fir_high, "fir", fir_order).get_data()[0] * 1e6,
                "Butterworth": apply_custom_filter(raw_crop, butt_low, butt_high, "iir", butt_order, "butter").get_data()[0] * 1e6,
                "Bessel": apply_custom_filter(raw_crop, bess_low, bess_high, "iir", bess_order, "bessel").get_data()[0] * 1e6,
                "Chebyshev": apply_custom_filter(raw_crop, cheb_low, cheb_high, "iir", cheb_order, "cheby1").get_data()[0] * 1e6
            }
            st.session_state["filter_comparison"] = (results, raw_crop.times)

    if "filter_comparison" in st.session_state:
        results, times = st.session_state["filter_comparison"]
        fig = go.Figure()
        for name, data in results.items():
            fig.add_trace(go.Scattergl(x=times, y=data, mode="lines", name=name, 
                                       line=dict(width=1 if name=="Original" else 1.5),
                                       opacity=0.5 if name=="Original" else 1.0))
        fig.update_layout(title=f"Bandpass Comparison - {comp_channel}", xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
                          template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Show Frequency Response (PSD) Comparison"):
            fig_psd = go.Figure()
            for name, data in results.items():
                f, p = sp_signal.welch(data, fs=raw.info['sfreq'], nperseg=min(len(data), 1024))
                # data is already in microvolts, so p is in microvolts^2/Hz. 10*log10(p) is ref 1 microvolt.
                fig_psd.add_trace(go.Scatter(x=f, y=10*np.log10(p + 1e-20), name=name))
            fig_psd.update_layout(title="PSD Comparison", xaxis_title="Frequency (Hz)", yaxis_title="dB ref 1µV²/Hz",
                                  xaxis_range=[0, 100], template="plotly_dark")
            st.plotly_chart(fig_psd, use_container_width=True)

        st.divider()
        st.subheader("Final Bandpass Choice")
        final_choice = st.selectbox("Which filter would you like to apply to the entire dataset?", 
                                   options=["FIR", "Butterworth", "Bessel", "Chebyshev"])
        
        if st.button("Apply Bandpass Filter", type="primary", use_container_width=True):
            with st.spinner(f"Applying {final_choice} filter..."):
                if final_choice == "FIR": raw_filt = apply_custom_filter(raw, fir_low, fir_high, "fir", fir_order)
                elif final_choice == "Butterworth": raw_filt = apply_custom_filter(raw, butt_low, butt_high, "iir", butt_order, "butter")
                elif final_choice == "Bessel": raw_filt = apply_custom_filter(raw, bess_low, bess_high, "iir", bess_order, "bessel")
                elif final_choice == "Chebyshev": raw_filt = apply_custom_filter(raw, cheb_low, cheb_high, "iir", cheb_order, "cheby1")
                
                st.session_state["raw_filtered"] = raw_filt
                # Clear downstream states to ensure pipeline integrity
                for key in ["ica_obj", "ica_labels"]:
                    if key in st.session_state:
                        del st.session_state[key]
                        
                st.success(f"{final_choice} bandpass filter applied successfully!")
                st.rerun()

# ── 2. Notch Filtering ───────────────────────────────────────────────────
with st.expander("2. Notch Filtering", expanded=False):
    raw_working = st.session_state.get("raw_filtered", raw)
    
    st.subheader("Notch Filter")
    notch_freq = st.number_input("Frequency to remove (Hz)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
    
    if st.button("Apply Notch Filter", type="primary", use_container_width=True):
        with st.spinner(f"Applying {notch_freq}Hz notch filter..."):
            raw_notched = apply_notch(raw_working, freqs=notch_freq)
            st.session_state["raw_filtered"] = raw_notched
            st.success(f"Notch filter at {notch_freq}Hz applied successfully!")
            st.rerun()

    st.divider()
    st.subheader("📊 Spectral Impact Comparison")
    
    psd_col1, psd_col2 = st.columns(2)
    with psd_col1:
        psd_ch = st.selectbox("Channel to inspect", options=raw.ch_names, key="psd_inspect_ch")
    with psd_col2:
        t_max_limit = float(raw.times[-1])
        psd_time_range = st.slider("Time window for PSD (s)", 0.0, t_max_limit, (0.0, min(10.0, t_max_limit)), key="psd_inspect_time")

    # Generate PSD Comparison
    t_start, t_end = psd_time_range
    if t_end <= t_start: t_end = t_start + 0.1
    
    # 1. Original (Raw Base)
    raw_orig_crop = raw.copy().crop(tmin=t_start, tmax=t_end).pick(psd_ch)
    # 2. Processed (Raw Working)
    raw_filt_crop = raw_working.copy().crop(tmin=t_start, tmax=t_end).pick(psd_ch)
    
    nyquist = raw.info['sfreq'] / 2.0
    psds_orig, freqs_orig = compute_psd(raw_orig_crop, fmin=0.5, fmax=min(100.0, nyquist-0.5))
    psds_filt, freqs_filt = compute_psd(raw_filt_crop, fmin=0.5, fmax=min(100.0, nyquist-0.5))
    
    fig_psd_comp = go.Figure()
    fig_psd_comp.add_trace(go.Scatter(x=freqs_orig, y=120 + 10*np.log10(psds_orig[0] + 1e-20), name="Original", line=dict(color="#ff6b6b", width=1.5)))
    fig_psd_comp.add_trace(go.Scatter(x=freqs_filt, y=120 + 10*np.log10(psds_filt[0] + 1e-20), name="Processed", line=dict(color="#51cf66", width=1.5)))
    
    fig_psd_comp.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Frequency (Hz)", yaxis_title="dB ref 1µV²/Hz",
        margin=dict(l=60, r=20, t=40, b=60),
    )
    st.plotly_chart(fig_psd_comp, use_container_width=True)

# ── 3. Statistical Epoch Tagging (1s) ────────────────────────────────────
with st.expander("3. Statistical Epoch Tagging (1s)", expanded=False):
    raw_working = st.session_state.get("raw_filtered", raw)
    
    st.subheader("Automated Segment Tagging (Kurtosis)")
    st.markdown("Identify 1-second segments based on **Kurtosis** and tag them as 'bad' for downstream processing (like ICA).")

    # 1. Segment the data into 1s windows
    sfreq = raw_working.info['sfreq']
    data = raw_working.get_data() * 1e6 # µV
    n_channels, n_samples = data.shape
    win_samples = int(sfreq) # 1 second
    n_epochs = n_samples // win_samples
    
    # Reshape to (n_epochs, n_channels, win_samples)
    epoch_data = data[:, :n_epochs*win_samples].reshape(n_channels, n_epochs, win_samples).transpose(1, 0, 2)
    
    # Calculate Kurtosis per epoch per channel
    kurt_per_epoch = kurtosis(epoch_data, axis=2) # (n_epochs, n_channels)
    max_kurt_per_epoch = np.max(kurt_per_epoch, axis=1) # (n_epochs,)
    min_kurt_per_epoch = np.min(kurt_per_epoch, axis=1) # (n_epochs,)
    
    st.write("**Threshold Tuning**")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        high_kurt_thresh = st.slider("High Kurtosis (Spikes)", 0.0, 50.0, 10.0, 0.5, help="Epochs exceeding this will be tagged.")
    with tcol2:
        low_kurt_thresh = st.slider("Low Kurtosis (Flatline)", -3.0, 0.0, -1.5, 0.1, help="Epochs below this will be tagged.")

    # Flagging logic
    bad_high = np.where(max_kurt_per_epoch > high_kurt_thresh)[0]
    bad_low = np.where(min_kurt_per_epoch < low_kurt_thresh)[0]
    bad_epochs = np.unique(np.concatenate([bad_high, bad_low]))
    
    st.metric("Flagged Segments", f"{len(bad_epochs)} / {n_epochs}", delta=f"{len(bad_high)} spikes, {len(bad_low)} flat")

    # 2. Visualize
    epoch_times = np.arange(n_epochs)
    fig_stats = go.Figure()
    fig_stats.add_trace(go.Bar(
        x=epoch_times, 
        y=max_kurt_per_epoch,
        marker_color=['#ff6b6b' if i in bad_epochs else '#51cf66' for i in range(n_epochs)],
        name="Max Kurtosis"
    ))
    fig_stats.add_hline(y=high_kurt_thresh, line_dash="dash", line_color="orange")
    fig_stats.add_hline(y=low_kurt_thresh, line_dash="dash", line_color="cyan")
    
    fig_stats.update_layout(title="Kurtosis Distribution (1s Windows)", template="plotly_dark", height=350)
    st.plotly_chart(fig_stats, use_container_width=True)

    if st.button("Apply Kurtosis-based Tagging", type="primary", use_container_width=True):
        if len(bad_epochs) == 0:
            st.warning("No segments flagged.")
        else:
            with st.spinner("Adding 'bad_kurtosis' annotations..."):
                raw_modified = raw_working.copy()
                # Create annotations for bad epochs
                onsets = bad_epochs.astype(float)
                durations = np.ones(len(bad_epochs))
                descriptions = ['bad_kurtosis'] * len(bad_epochs)
                
                new_ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=raw_modified.info['meas_date'])
                raw_modified.set_annotations(raw_modified.annotations + new_ann)
                
                st.session_state["raw_filtered"] = raw_modified
                st.success(f"Successfully tagged {len(bad_epochs)} segments as bad!")
                st.rerun()

# ── 4. Artifact Rejection (ICA) ──────────────────────────────────────────────────
with st.expander("4. Artifact Rejection (ICA)", expanded=False):
    raw_working = st.session_state.get("raw_filtered", raw)
    
    # ── Step 1: ICA Fitting ───────────────────────────────────────────────
    st.subheader("Independent Component Analysis (ICA) - Fitting")
    
    # Check for bad segments to inform user
    bad_anns = [a for a in raw_working.annotations if a['description'].lower().startswith('bad')]
    if bad_anns:
        st.info(f"✅ Found {len(bad_anns)} segments tagged as 'bad'. These will be **dismissed** during ICA fitting to ensure clean decomposition.")
    else:
        st.warning("⚠️ No 'bad' segments found. We highly recommend running **Section 3** first to tag noisy segments, otherwise ICA might fail to converge or capture artifacts properly.")
    
    ica_input_data = raw_working
    n_components = st.slider("Number of ICA components", 2, min(20, len(ica_input_data.ch_names)), min(10, len(ica_input_data.ch_names)), key="ica_n")
    
    if st.button("Run ICA Decomposition", type="primary", use_container_width=True):
        with st.spinner("Fitting ICA (excluding segments marked as 'bad')..."):
            # Initialize ICA
            ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
            
            # fit ICA - reject_by_annotation=True ensures bad segments are ignored
            ica.fit(ica_input_data, reject_by_annotation=True, verbose=False)
            
            # Store in session state
            st.session_state["ica_obj"] = ica
            # Store the data used for ICA to allow reliable comparison later
            st.session_state["raw_pre_ica"] = ica_input_data.copy()
            
            # Clear previous labels if any
            if "ica_labels" in st.session_state:
                del st.session_state["ica_labels"]
                
            st.success("ICA decomposition complete. Please proceed to **Section 5** for analysis.")
            st.rerun()

# ── 5. IC Analysis & Component Removal ──────────────────────────────────────
with st.expander("5. IC Analysis & Component Removal", expanded=False):
    if "ica_obj" not in st.session_state:
        st.info("Please run ICA decomposition in **Section 4** first.")
    else:
        # Use the pre-ICA data for all analysis and comparison
        raw_pre_ica = st.session_state.get("raw_pre_ica", st.session_state.get("raw_filtered", raw))
        ica_obj = st.session_state["ica_obj"]
        
        st.subheader("Independent Components (Time Series)")
        
        t_max_limit = float(raw.times[-1])
        ica_view_t = st.slider("ICA View window (s)", 0.0, t_max_limit, (0.0, min(5.0, t_max_limit)), key="ica_v_t")
        
        t_start, t_end = ica_view_t
        if t_end <= t_start: t_end = t_start + 0.1
        
        # Ensure times are within bounds
        t_end = min(t_end, raw_pre_ica.times[-1])
        
        # Extract component activations
        sources_raw = ica_obj.get_sources(raw_pre_ica.copy().crop(tmin=t_start, tmax=t_end))
        sources_vals = sources_raw.get_data()
        sources_times = sources_raw.times + t_start
        
        fig_ica = go.Figure()
        # Scale components for better stacking
        scale = np.max(np.abs(sources_vals)) * 1.5 if sources_vals.size > 0 else 1.0
        for i in range(sources_vals.shape[0]):
            fig_ica.add_trace(go.Scattergl(x=sources_times, y=sources_vals[i] + i*scale, name=f"ICA{i:03d}", line=dict(width=1)))
        
        fig_ica.update_layout(title="Independent Component Activations (Stacked)", 
                                template="plotly_dark", height=600, 
                                yaxis_title="Component + Offset",
                                hovermode="x unified")
        st.plotly_chart(fig_ica, use_container_width=True)

        # ── ICA Topographies (Grid) ──────────────────────────────────
        st.divider()
        st.subheader("Component Topographies (Spatial Maps)")
        
        with st.expander("ℹ️ How to interpret these maps", expanded=False):
            st.markdown("""
            **Colors & Polarity (RdBu_r Colormap):**
            *   **Red (Positive Weights):** Regions where the component has a strong positive influence on the scalp electrodes.
            *   **Blue (Negative Weights):** Regions where the component has a strong negative influence.
            *   *Note:* The absolute polarity is arbitrary; focus on the **spatial distribution**.
            
            **Common Patterns:**
            *   **Eye Blinks:** Strong concentration (Red or Blue) at the very front (near the 'nose' triangle).
            *   **Lateral Eye Movements:** High weights at the front-left and front-right with opposite polarities.
            *   **Muscle Noise (EMG):** High-frequency patches at the very edges of the map (temporal or occipital).
            *   **Brain Activity:** Smooth, centralized gradients.
            """)
        
        # Check if montage is available
        has_montage = False
        try:
            montage = raw_pre_ica.get_montage()
            if montage:
                ch_pos = montage.get_positions()['ch_pos']
                has_montage = all(ch in ch_pos and ch_pos[ch] is not None for ch in raw_pre_ica.ch_names)
        except:
            pass
        
        if not has_montage:
            st.warning("Electrode positions missing. Apply a 10-20 Montage in the sidebar to view topomaps.")
        else:
            n_comp = ica_obj.n_components_
            cols_per_row = 4
            n_rows = (n_comp + cols_per_row - 1) // cols_per_row
            
            # Using Matplotlib to generate the individual maps and then displaying them in Streamlit columns
            # This is more stable for MNE topomaps than raw Plotly conversion
            for r in range(n_rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx = r * cols_per_row + c
                    if idx < n_comp:
                        with cols[c]:
                            ica_data = ica_obj.get_components()[:, idx]
                            fig, ax = plt.subplots(figsize=(3, 3))
                            fig.patch.set_facecolor("#0d1117")
                            
                            mne.viz.plot_topomap(
                                ica_data, raw_pre_ica.info, axes=ax, show=False,
                                cmap="RdBu_r", contours=4, names=None # Hide names for cleaner grid
                            )
                            ax.set_title(f"ICA{idx:03d}", color="white", fontsize=10, fontweight="bold")
                            st.pyplot(fig)
                            plt.close(fig)

        # ── Auto-Detection with mne-icalabel ──────────────────────────
        st.divider()
        st.subheader("Auto-Detection (mne-icalabel)")
        if st.button("Run Auto-Labeling", type="secondary", use_container_width=True):
            try:
                from mne_icalabel import label_components
                with st.spinner("Analyzing components..."):
                    labels = label_components(raw_pre_ica, ica_obj, method='iclabel')
                    st.session_state["ica_labels"] = labels
                    st.success("Auto-labeling complete!")
                    st.rerun()
            except ImportError:
                st.error("mne-icalabel is not installed. Please install it using 'pip install mne-icalabel'.")
            except Exception as e:
                st.error(f"Error during auto-labeling: {e}")

        ica_labels = st.session_state.get("ica_labels")
        recommended_exclude = []
        if ica_labels:
            st.write("**Detected Labels:**")
            df_labels = pd.DataFrame({
                "Component": range(len(ica_labels['labels'])),
                "Label": ica_labels['labels'],
                "Probability": ica_labels['y_pred_proba']
            })
            st.dataframe(df_labels, hide_index=True, use_container_width=True)
            recommended_exclude = [i for i, l in enumerate(ica_labels['labels']) if l != 'brain']
            if recommended_exclude:
                st.info(f"Recommended to exclude (non-brain): {recommended_exclude}")

        st.divider()
        exclude = st.multiselect("Select components to exclude (artifacts)", 
                                    options=list(range(ica_obj.n_components_)),
                                    default=recommended_exclude if recommended_exclude else [])
        
        # ── ICA Reconstruction Verification (Live Preview) ──────────────────
        with st.expander("ICA Cleaning Diagram (Live Preview)", expanded=True):
            st.subheader("Visual Verification: Before vs After (Current Selection)")
            
            iccol1, iccol2 = st.columns([1, 2])
            with iccol1:
                ica_comp_ch = st.selectbox("Channel to compare", options=raw_pre_ica.ch_names, index=0, key="ica_v_ch",
                                         format_func=lambda x: DEFAULT_MAPPING.get(x, x))
            with iccol2:
                t_max_limit = float(raw.times[-1])
                ica_comp_t = st.slider("Comparison Time window (s)", 0.0, t_max_limit, (0.0, min(5.0, t_max_limit)), key="ica_v_t_comp")

            t_start, t_end = ica_comp_t
            if t_end <= t_start: t_end = t_start + 0.1
            
            # 1. Before Trace (Original filtered data)
            data_pre = raw_pre_ica.copy().crop(tmin=t_start, tmax=t_end).pick(ica_comp_ch)
            pre_vals = data_pre.get_data()[0] * 1e6
            plot_times = data_pre.times + t_start
            
            # 2. After Trace (Apply current exclusion list on the fly)
            ica_obj.exclude = exclude
            raw_post_temp = ica_obj.apply(raw_pre_ica.copy().crop(tmin=t_start, tmax=t_end), verbose=False)
            post_vals = raw_post_temp.pick(ica_comp_ch).get_data()[0] * 1e6
            
            fig_ica_comp = go.Figure()
            fig_ica_comp.add_trace(go.Scattergl(x=plot_times, y=pre_vals, name="Before ICA", 
                                               line=dict(color="#ff6b6b", width=1.2), opacity=0.5))
            fig_ica_comp.add_trace(go.Scattergl(x=plot_times, y=post_vals, name="After ICA Exclusion", 
                                               line=dict(color="#51cf66", width=1.5)))
            
            fig_ica_comp.update_layout(title=f"Live Effect: {DEFAULT_MAPPING.get(ica_comp_ch, ica_comp_ch)}",
                                      template="plotly_dark", height=450, xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
                                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ica_comp, use_container_width=True)

        if st.button("Apply ICA Exclusion Permanently", type="primary", use_container_width=True):
            ica_obj.exclude = exclude
            # Apply to the full pre-ICA dataset
            raw_cleaned = ica_obj.apply(raw_pre_ica.copy(), verbose=False)
            st.session_state["raw_filtered"] = raw_cleaned
            st.success("ICA exclusion applied. Final data updated.")
            st.rerun()

        # ── Master Investigation Export Tool ──────────────────────────────────
        st.divider()
        with st.expander("📥 Export Master Investigation Report (Internal Channel Switching)", expanded=False):
            st.markdown("""
            Generate a single, comprehensive HTML report containing **all channels**.
            Inside the report, you can use a **dropdown menu** to switch between channels and 
            toggle between Raw, Filtered, and ICA states.
            """)
            
            t_max = float(raw_pre_ica.times[-1])
            export_t_range = st.slider("Export Time Range (s)", 0.0, t_max, (0.0, min(30.0, t_max)), key="ica_master_exp_t")
            
            if st.button("🚀 Prepare Master Report", use_container_width=True):
                with st.spinner("Embedding all channels into interactive report..."):
                    t_start, t_end = export_t_range
                    if t_end <= t_start: t_end = t_start + 1.0
                    
                    # Prepare the 3 states
                    raw_base = st.session_state.get("raw_base", raw)
                    d_raw_full = raw_base.copy().crop(tmin=t_start, tmax=t_end)
                    d_filt_full = raw_pre_ica.copy().crop(tmin=t_start, tmax=t_end)
                    
                    temp_ica = ica_obj.copy()
                    temp_ica.exclude = exclude
                    d_ica_full = temp_ica.apply(raw_pre_ica.copy().crop(tmin=t_start, tmax=t_end), verbose=False)
                    
                    p_times = d_raw_full.times + t_start
                    channels = d_raw_full.ch_names
                    
                    fig_master = go.Figure()
                    
                    # Add 3 traces per channel (Raw, Filtered, ICA)
                    # We store them in a predictable order to make toggling easy
                    for i, ch in enumerate(channels):
                        ch_label = DEFAULT_MAPPING.get(ch, ch)
                        is_visible = (i == 0) # Only first channel visible by default
                        
                        # Trace 0: Raw
                        fig_master.add_trace(go.Scatter(x=p_times, y=d_raw_full.get_data(picks=ch)[0] * 1e6, 
                                                        name=f"{ch_label} - Raw", visible=is_visible,
                                                        line=dict(color="#636efa", width=1), opacity=0.3))
                        # Trace 1: Filtered
                        fig_master.add_trace(go.Scatter(x=p_times, y=d_filt_full.get_data(picks=ch)[0] * 1e6, 
                                                        name=f"{ch_label} - Filtered", visible=is_visible,
                                                        line=dict(color="#ff6b6b", width=1.2), opacity=0.5))
                        # Trace 2: ICA
                        fig_master.add_trace(go.Scatter(x=p_times, y=d_ica_full.get_data(picks=ch)[0] * 1e6, 
                                                        name=f"{ch_label} - ICA Cleaned", visible=is_visible,
                                                        line=dict(color="#51cf66", width=1.8)))
                    
                    # ── Internal Dropdown for Channel Selection ──
                    dropdown_buttons = []
                    for i, ch in enumerate(channels):
                        ch_label = DEFAULT_MAPPING.get(ch, ch)
                        
                        # Create a visibility mask: only the 3 traces for this channel are True
                        visibility_mask = [False] * (len(channels) * 3)
                        visibility_mask[i*3 : (i+1)*3] = [True, True, True]
                        
                        dropdown_buttons.append(dict(
                            label=ch_label,
                            method="update",
                            args=[{"visible": visibility_mask},
                                  {"title": f"Investigation: {ch_label} ({t_start:.1f}s - {t_end:.1f}s)"}]
                        ))

                    fig_master.update_layout(
                        updatemenus=[
                            # Dropdown for Channels
                            dict(
                                buttons=dropdown_buttons,
                                direction="down",
                                pad={"r": 10, "t": 10},
                                showactive=True,
                                x=0.05,
                                xanchor="left",
                                y=1.15,
                                yanchor="top"
                            ),
                            # Buttons for Processing State Comparison
                            dict(
                                type="buttons",
                                direction="right",
                                x=0.95,
                                xanchor="right",
                                y=1.15,
                                yanchor="top",
                                buttons=[
                                    dict(label="Show All (Raw/Filt/ICA)", method="restyle", 
                                         args=[{"opacity": [0.3, 0.5, 1.0] * len(channels)}]),
                                    dict(label="Compare Filter/ICA", method="restyle", 
                                         args=[{"opacity": [0.0, 0.6, 1.0] * len(channels)}]),
                                    dict(label="Only Final Result", method="restyle", 
                                         args=[{"opacity": [0.0, 0.0, 1.0] * len(channels)}])
                                ]
                            )
                        ],
                        title=f"Investigation: {DEFAULT_MAPPING.get(channels[0], channels[0])} ({t_start:.1f}s - {t_end:.1f}s)",
                        template="plotly_dark",
                        height=700,
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude (µV)",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        hovermode="x unified"
                    )
                    
                    html_bytes = fig_master.to_html(full_html=True, include_plotlyjs='cdn').encode('utf-8')
                    
                    st.success("✅ Master Investigation Report ready!")
                    st.download_button(
                        label="📥 Download Master Investigation HTML",
                        data=html_bytes,
                        file_name="master_eeg_investigation.html",
                        mime="text/html",
                        type="primary",
                        use_container_width=True
                    )


# ── 6. Feature Extraction ────────────────────────────────────────────────
st.divider()
with st.expander("6. Extract Features (ML Classification Dataset)", expanded=False):
    raw_working = st.session_state.get("raw_filtered", raw).copy()
    
    # Sync manual annotations from session state into raw_working
    rec_name = st.session_state.get("active_rec_name", "default")
    if "annotations_dict" in st.session_state and rec_name in st.session_state["annotations_dict"]:
        new_anns = []
        for ann in st.session_state["annotations_dict"][rec_name]:
            new_anns.append({
                'onset': ann['start'],
                'duration': ann['end'] - ann['start'],
                'description': ann['label']
            })
        if new_anns:
            # Convert to MNE Annotations
            mne_anns = mne.Annotations(
                onset=[a['onset'] for a in new_anns],
                duration=[a['duration'] for a in new_anns],
                description=[a['description'] for a in new_anns],
                orig_time=raw_working.info['meas_date']
            )
            raw_working.set_annotations(raw_working.annotations + mne_anns)

    st.subheader("Extract Wide Feature Dataset")
    st.markdown("""
    Generate a dataset suitable for machine learning classification.
    - **Window:** 1.0s (Fixed, matching Section 3)
    - **Dimensions:** (N_epochs, (16 channels × 6 features) + 1 tag)
    - **Features:** Band Powers (dB) & RMS (µV)
    - **Tag:** Combined Manual + Statistical Tags
    """)
    
    if st.button("Extract Features", type="primary", use_container_width=True):
        with st.spinner("Extracting features and tags..."):
            # 1. Create epochs (Fixed 1.0s to match Section 3)
            epochs = make_fixed_epochs(raw_working, duration=1.0, overlap=0.0)
            
            # 2. Extract 3D features (N, Ch, Feat)
            features_3d, feat_names = extract_3d_features(epochs)
            
            # 3. Get Tags for each epoch
            epoch_tags = get_epoch_tags(epochs, raw_working)
            
            # 4. Store in session state
            st.session_state["extracted_features"] = {
                "data": features_3d,
                "feature_names": feat_names,
                "ch_names": epochs.info['ch_names'],
                "epoch_tags": epoch_tags
            }
            st.success(f"Extracted {features_3d.shape[0]} epochs.")

    if "extracted_features" in st.session_state:
        feat_info = st.session_state["extracted_features"]
        
        # Defensive check for stale session state
        if "epoch_tags" not in feat_info:
            st.info("Please click 'Extract Features' to generate the dataset with tags.")
        else:
            f3d = feat_info["data"]
            tags = feat_info["epoch_tags"]
            ch_names = feat_info["ch_names"]
            f_names = feat_info["feature_names"]
            
            n_ep, n_ch, n_feat = f3d.shape
            
            # Build Wide DataFrame
            # Columns: [Ch1_Delta, Ch1_Theta, ..., Ch1_RMS, Ch2_Delta, ..., Tag]
            all_rows = []
            for e in range(n_ep):
                row = []
                for c in range(n_ch):
                    row.extend(list(f3d[e, c, :]))
                row.append(tags[e])
                all_rows.append(row)
                
            col_names = []
            for ch in ch_names:
                ch_label = DEFAULT_MAPPING.get(ch, ch)
                for fn in f_names:
                    col_names.append(f"{ch_label}_{fn}")
            col_names.append("Tag")
            
            df_wide = pd.DataFrame(all_rows, columns=col_names)
            
            st.write(f"**Dataset Shape:** {df_wide.shape}")
            st.dataframe(df_wide.head(10), use_container_width=True)
            
            csv_feats = df_wide.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download ML Dataset (CSV)",
                data=csv_feats,
                file_name="eeg_ml_dataset.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_feat_csv"
            )


# ── Export & Comparison ────────────────────────────────────────────────────
if "raw_filtered" in st.session_state:
    st.header("Save Preprocessed Data")
    if st.button("Prepare CSV for Download"):
        with st.spinner("Exporting..."):
            # 1. Convert MNE to DataFrame
            df_export = raw_final.to_data_frame()

            # 2. Add Epoch Number and Kurtosis (1s windows)
            sfreq = raw_final.info['sfreq']
            data = raw_final.get_data() * 1e6 # µV
            n_channels, n_samples = data.shape
            win_samples = int(sfreq) # 1 second window
            
            # Calculate epoch numbers for every sample
            indices = np.arange(len(df_export))
            df_export["epoch_number"] = indices // win_samples
            
            # Calculate Kurtosis per epoch
            n_epochs = n_samples // win_samples
            epoch_data = data[:, :n_epochs*win_samples].reshape(n_channels, n_epochs, win_samples).transpose(1, 0, 2)
            kurt_per_epoch = kurtosis(epoch_data, axis=2) 
            max_kurt_per_epoch = np.max(kurt_per_epoch, axis=1) # (n_epochs,)
            
            kurt_mapping = {i: max_kurt_per_epoch[i] for i in range(n_epochs)}
            
            # Handle tail samples if any
            if n_samples > n_epochs * win_samples:
                tail_data = data[:, n_epochs*win_samples:]
                if tail_data.shape[1] > 1: # Kurtosis needs at least 2 samples, ideally more
                    tail_kurt = np.max(kurtosis(tail_data, axis=1))
                else:
                    tail_kurt = 0.0
                kurt_mapping[n_epochs] = tail_kurt
            
            df_export["epoch_kurtosis"] = df_export["epoch_number"].map(kurt_mapping)
            
            # 3. Sync Annotations from Session State
            rec_name = st.session_state.get("active_rec_name", "default")
            if "tag" not in rec.data.columns:
                rec.data["tag"] = "none"
            
            # Ensure the tag column reflects all annotations in the session state
            if "annotations_dict" in st.session_state and rec_name in st.session_state["annotations_dict"]:
                for ann in st.session_state["annotations_dict"][rec_name]:
                    s_idx = int(ann["start"] * rec.sfreq)
                    e_idx = int(ann["end"] * rec.sfreq)
                    # Clip indices to safe bounds
                    s_idx = max(0, min(s_idx, len(rec.data)-1))
                    e_idx = max(0, min(e_idx, len(rec.data)))
                    rec.data.iloc[s_idx:e_idx, rec.data.columns.get_loc("tag")] = ann["label"]
            
            # 4. Add the tag column to the exported dataframe
            # MNE to_data_frame() should have the same number of rows as rec.data 
            # as long as no global cropping or resampling was applied.
            if len(df_export) == len(rec.data):
                df_export["tag"] = rec.data["tag"].values
            else:
                # Fallback: fill with 'none' if length mismatch occurs (unexpected)
                df_export["tag"] = "none"
                st.warning("Length mismatch between MNE data and original recording. Tags could not be aligned perfectly.")

            # 5. Convert to CSV
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            orig_filename = rec.metadata.get("file", "eeg_recording.csv")
            base_name = orig_filename.split('.')[0]
            st.download_button(label="📥 Download Post-Filtered EEG (CSV)", data=csv_data, 
                               file_name=f"preprocessed_{base_name}.csv", mime="text/csv", type="primary")

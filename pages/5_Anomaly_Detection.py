"""
Anomaly Detection — EEG-specific and ML/DL anomaly scoring with
interactive threshold, heatmap, epoch gallery, and feature importance.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

st.set_page_config(page_title="Anomaly Detection", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Anomaly Detection")

# ── Descriptive Introduction ───────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
                border: 1px solid #30363d; border-radius: 12px; padding: 20px 24px;
                margin-bottom: 1.5rem; line-height: 1.7;">
        <h3 style="color: #58a6ff; margin-top: 0; font-size: 1.1rem;">
            🔍 What does EEG anomaly detection do?
        </h3>
        <p style="color: #c9d1d9; font-size: 0.9rem; margin-bottom: 10px;">
            EEG anomaly detection automatically identifies unusual segments in brain
            recordings — artifacts from muscle movement, eye blinks, electrode pops,
            or genuinely abnormal neural patterns (e.g., epileptiform spikes). By
            flagging these epochs, researchers can clean data before analysis or
            discover clinically relevant events.
        </p>
        <p style="color: #8b949e; font-size: 0.82rem; margin-bottom: 0;">
            <strong style="color: #c9d1d9;">This page runs two categories of detectors:</strong><br/>
            <span style="color: #3fb950;">■</span> <strong>EEG-Specific</strong> —
            Statistical Z-score thresholding, spectral band-ratio outliers (θ/β, α/β),
            and kurtosis/entropy analysis for muscle artifacts and flat-line detection.<br/>
            <span style="color: #58a6ff;">■</span> <strong>General ML/DL</strong> —
            Isolation Forest, Local Outlier Factor, One-Class SVM, and an optional
            Autoencoder — all applied to a rich feature space (band powers, Hjorth
            parameters, line length) extracted from each epoch.<br/>
            Results are ensembled into a single score and visualised as time-series,
            a channel×epoch heatmap, and an epoch gallery for inspection.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from utils.signal_processing import make_fixed_epochs, extract_epoch_features
from analysis.anomaly import (
    run_isolation_forest, run_lof, run_ocsvm,
    run_autoencoder, ensemble_scores, _check_torch,
    run_zscore_detector, run_spectral_ratio_detector,
    run_kurtosis_entropy_detector,
    parse_trigger_csv, get_trigger_info,
    compute_trigger_locked_scores, pre_post_trigger_test,
    trigger_coincidence_test, compute_trigger_band_shift,
)
from analysis.video_analysis import (
    analyze_video_with_gemini, parse_gemini_response,
    clamp_events_to_duration,
    compute_event_anomaly_coincidence, compute_event_locked_erp,
    compute_event_anomaly_stats, add_video_event_overlays,
    build_event_strip_trace, events_to_dataframe, get_mime_type,
    EVENT_COLORS, EVENT_ICONS, SEVERITY_SIZE,
)

# ── Optional Trigger File Upload ───────────────────────────────────────────
with st.expander("⚡ VR Experiment Trigger Events (optional)", expanded=False):
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem; margin-bottom: 8px;">'
        'Upload the <strong>Trigger Time CSV</strong> from your VR evacuation experiment '
        'to overlay trigger events on anomaly plots and run trigger–anomaly '
        'relationship analyses.</p>',
        unsafe_allow_html=True,
    )
    trigger_file = st.file_uploader(
        "Upload Trigger Time CSV",
        type=["csv"],
        key="trigger_csv_upload",
        help="CSV with columns: Participant Number, User Group, Trigger Time, ...",
    )

    trigger_info = st.session_state.get("trigger_info")

    if trigger_file is not None:
        trigger_df = parse_trigger_csv(trigger_file.getvalue())
        st.session_state["trigger_df"] = trigger_df

        participants = trigger_df["Participant Number"].dropna().tolist()
        selected_participant = st.selectbox(
            "Select Participant",
            options=participants,
            index=0,
            help="Choose the participant whose trigger event to overlay.",
        )

        trigger_info = get_trigger_info(trigger_df, selected_participant)

        if trigger_info is not None:
            st.session_state["trigger_info"] = trigger_info
            grp = trigger_info["group"]
            icon = "🔊" if grp == "Auditory" else "👁️"
            color = "#f0883e" if grp == "Auditory" else "#58a6ff"
            t_s = trigger_info["trigger_time_s"]
            t_min = int(t_s // 60)
            t_sec = t_s % 60

            _cur_offset = float(st.session_state.get("video_time_offset_s", 0.0))
            _eeg_note = ""
            if _cur_offset != 0.0:
                _eeg_t = t_s + _cur_offset
                _eeg_min = int(_eeg_t // 60)
                _eeg_sec = _eeg_t % 60
                _eeg_note = (
                    f'<br><span style="color:#8b949e; font-size:0.85rem;">'
                    f'&#128280; Analysed at EEG position&nbsp;'
                    f'<strong style="color:#ffa657;">{_eeg_min}:{_eeg_sec:04.1f}</strong>'
                    f'&nbsp;&mdash;&nbsp;VR trigger time ({t_s:.1f} s)'
                    f' shifted forward by sync offset'
                    f' <strong style="color:#ffa657;">{_cur_offset:+.1f} s</strong>'
                    f'</span>'
                )
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #0d1f0d 0%, #1a2a1a 100%); '
                f'border: 1px solid {color}; border-radius: 8px; padding: 12px 16px; '
                f'margin-top: 8px;">'
                f'<span style="font-size: 1.4rem;">{icon}</span> '
                f'<strong style="color: {color}; font-size: 1rem;"> '
                f'{selected_participant}</strong> &nbsp;—&nbsp; '
                f'<span style="color: #c9d1d9;">{grp} trigger at '
                f'<strong>{t_min}:{t_sec:04.1f}</strong> ({t_s:.1f}s VR time)</span>'
                f'{_eeg_note}</div>',
                unsafe_allow_html=True,
            )
            if trigger_info["explanations"]:
                for note in trigger_info["explanations"]:
                    st.caption(f"📝 {note}")
        else:
            st.warning(
                f"No valid trigger time found for **{selected_participant}**. "
                "Trigger analysis will be disabled."
            )
            st.session_state.pop("trigger_info", None)
    elif trigger_info is not None:
        # Keep previous trigger info if file was already uploaded
        grp = trigger_info["group"]
        icon = "🔊" if grp == "Auditory" else "👁️"
        st.info(
            f"{icon} Using previously loaded trigger: "
            f"**{trigger_info['participant']}** ({grp}, "
            f"{trigger_info['trigger_time_s']:.1f}s)"
        )

# ── Optional VR Screen Recording Analysis ─────────────────────────────────
with st.expander("🎥 VR Screen Recording Analysis (optional)", expanded=False):
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem; margin-bottom: 8px;">'
        'Upload a <strong>VR screen recording</strong> to automatically extract '
        'behavioural events (head movements, gaze shifts, sudden actions) using '
        'Google Gemini Vision-Language Model. Events are overlaid on anomaly plots '
        'and used for multi-modal EEG–behaviour correlation analysis.</p>',
        unsafe_allow_html=True,
    )

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your Google AI Studio API key for Gemini. Get one at https://aistudio.google.com/",
        key="gemini_api_key_input",
    )

    vid_col1, vid_col2 = st.columns(2)
    with vid_col1:
        video_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "webm", "mkv"],
            key="video_upload",
            help="VR screen recording file (up to 200 MB via upload).",
        )
    with vid_col2:
        video_path = st.text_input(
            "Or enter local file path",
            help="Full path to a video file on disk (for larger files).",
            key="video_path_input",
        )

    model_col, _ = st.columns([1, 1])
    with model_col:
        gemini_model = st.selectbox(
            "Gemini Model",
            options=["gemini-flash-latest", "gemini-flash-lite-latest"],
            index=0,
            help="Flash provides full analysis; Flash Lite is faster and lighter.",
        )

    custom_prompt = st.text_area(
        "Additional context for the AI (optional)",
        value="This is an indoor office/building evacuation scenario in VR. "
              "The participant may encounter fire alarms, smoke, exit signs, "
              "and emergency lighting.",
        height=80,
        help="Add experiment-specific context to improve event extraction quality.",
    )

    video_events = st.session_state.get("video_events")

    if st.button("🔍 Analyse Video", type="secondary"):
        if not gemini_api_key:
            st.error("Please provide a Gemini API key.")
        else:
            # Resolve video bytes
            video_bytes = None
            video_filename = "video.mp4"
            if video_file is not None:
                video_bytes = video_file.getvalue()
                video_filename = video_file.name
            elif video_path.strip():
                import os
                p = video_path.strip()
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        video_bytes = f.read()
                    video_filename = os.path.basename(p)
                else:
                    st.error(f"File not found: {p}")

            if video_bytes is not None:
                mime = get_mime_type(video_filename)
                with st.spinner("Sending video to Gemini for analysis (this may take 1-2 minutes)…"):
                    try:
                        result = analyze_video_with_gemini(
                            video_bytes=video_bytes,
                            api_key=gemini_api_key,
                            model=gemini_model,
                            custom_prompt=custom_prompt,
                            mime_type=mime,
                        )
                        events = parse_gemini_response(result["events_text"])
                        # Clamp timestamps to actual video duration
                        vid_dur = result.get("video_duration_s")
                        if vid_dur is not None:
                            n_before = len(events)
                            events = clamp_events_to_duration(events, vid_dur)
                            n_dropped = n_before - len(events)
                            if n_dropped > 0:
                                st.warning(
                                    f"Dropped **{n_dropped}** events with timestamps "
                                    f"beyond video duration ({vid_dur:.1f}s)."
                                )
                        st.session_state["video_events"] = events
                        st.session_state["video_raw_response"] = result["events_text"]
                        st.session_state["video_scene_summary"] = result.get("scene_summary", "")
                        st.session_state["video_duration_s"] = result.get("video_duration_s")
                        video_events = events
                        st.success(f"Extracted **{len(events)}** behavioural events from video.")
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")
            elif video_file is None and not video_path.strip():
                st.warning("Please upload a video file or enter a file path.")

    # Display extracted events if available
    if video_events:
        # Summary badges
        type_counts = {}
        for ev in video_events:
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1

        badge_html = ""
        for etype, count in sorted(type_counts.items()):
            color = EVENT_COLORS.get(etype, "#8b949e")
            icon_e = EVENT_ICONS.get(etype, "📌")
            badge_html += (
                f'<span style="background: {color}22; color: {color}; '
                f'border: 1px solid {color}; border-radius: 12px; '
                f'padding: 3px 10px; margin-right: 6px; font-size: 0.85rem;">'
                f'{icon_e} {etype}: <strong>{count}</strong></span>'
            )
        st.markdown(badge_html, unsafe_allow_html=True)

        # Event table
        ev_df = events_to_dataframe(video_events, time_offset_s=float(st.session_state.get("video_time_offset_s", 0.0)))
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

# Scene summary & raw response OUTSIDE the expander (Streamlit forbids nested expanders)
if st.session_state.get("video_events"):
    scene = st.session_state.get("video_scene_summary", "")
    if scene:
        with st.expander("📝 Scene Narrative Summary", expanded=False):
            st.markdown(scene)

    with st.expander("🔧 Raw Gemini Response", expanded=False):
        st.code(st.session_state.get("video_raw_response", ""), language="json")

# ── Data Selection ────────────────────────────────────────────────────────
raw_filtered = st.session_state.get("raw_filtered")

if raw_filtered is not None:
    raw = raw_filtered
    st.success("✨ Currently using **post-filtered** signals (from Preprocessing page).")
else:
    raw = rec.build_mne_raw()
    st.warning("⚠️ Currently using **raw** EEG signals. It is highly recommended to apply filters on the [Preprocessing](./Preprocessing) page first.")

# ── EEG ↔ VR Video Time Synchronisation ───────────────────────────────────
if st.session_state.get("video_events"):
    _eeg_dur = raw.times[-1] if hasattr(raw, "times") and len(raw.times) > 0 else rec.metadata.get("duration_s", 0)
    _vid_dur = st.session_state.get("video_duration_s")  # set during analysis if ffprobe available
    with st.expander("⏱️ EEG ↔ Video Time Synchronisation", expanded=True):
        st.markdown(
            '<p style="color:#8b949e; font-size:0.88rem; margin-bottom:10px;">'
            'The EEG and VR screen recording may have started at different times. '
            'Set the <strong>video start offset</strong>: the EEG timestamp (in seconds) '
            'that corresponds to the first frame of the video. All video event annotations '
            'will be shifted by this amount on every plot and in all analyses.<br>'
            '<em>Example: EEG = 360 s, Video = 243 s → if the video started 60 s into '
            'the EEG, set offset = 60.</em></p>',
            unsafe_allow_html=True,
        )
        sync_col1, sync_col2, sync_col3 = st.columns([2, 1, 1])
        with sync_col1:
            _offset_default = float(st.session_state.get("video_time_offset_s", 0.0))
            _offset_max = max(float(_eeg_dur), 600.0)
            video_time_offset = st.slider(
                "Video start offset (s) — EEG time when recording began",
                min_value=0.0,
                max_value=_offset_max,
                value=_offset_default,
                step=0.5,
                key="video_offset_slider",
                help="Drag to shift all video event markers along the EEG timeline.",
            )
        with sync_col2:
            video_time_offset = st.number_input(
                "Exact offset (s)",
                min_value=0.0,
                max_value=_offset_max,
                value=video_time_offset,
                step=0.1,
                format="%.1f",
                key="video_offset_input",
            )
        with sync_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            eeg_end = video_time_offset + (_vid_dur if _vid_dur else 0)
            _dur_str = f"{_vid_dur:.0f} s" if _vid_dur else "unknown"
            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d; '
                f'border-radius:8px; padding:8px 12px; font-size:0.82rem; color:#c9d1d9;">'
                f'📹 Video: <strong>{_dur_str}</strong><br>'
                f'🧠 EEG: <strong>{_eeg_dur:.0f} s</strong><br>'
                f'🎯 Video ends at EEG: <strong>{eeg_end:.0f} s</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.session_state["video_time_offset_s"] = video_time_offset
else:
    video_time_offset = 0.0

# ── Epoch settings ─────────────────────────────────────────────────────────
st.header("⚙️ Detection Settings")
col1, col2 = st.columns(2)
with col1:
    epoch_dur = st.slider("Epoch duration (s)", 1.0, 5.0, 2.0, 0.5)
with col2:
    contamination = st.slider("Expected anomaly rate", 0.01, 0.20, 0.05, 0.01)

# ── Run Detection ──────────────────────────────────────────────────────────
if st.button("Run Anomaly Detection", type="primary"):
    with st.spinner("Creating epochs and extracting features..."):
        epochs = make_fixed_epochs(raw, duration=epoch_dur)
        features = extract_epoch_features(epochs)
        st.session_state["anomaly_features"] = features
        st.session_state["anomaly_epoch_dur"] = epoch_dur
        st.session_state["anomaly_epochs_obj"] = epochs

    eeg_scores = {}
    ml_scores = {}

    # ── EEG-Specific Detectors ─────────────────────────────────────────
    with st.spinner("Running Z-Score Threshold Detector..."):
        eeg_scores["Z-Score Threshold"] = run_zscore_detector(epochs)

    with st.spinner("Running Spectral Ratio Detector (θ/β, α/β)..."):
        eeg_scores["Spectral Ratio"] = run_spectral_ratio_detector(epochs)

    with st.spinner("Running Kurtosis & Entropy Detector..."):
        eeg_scores["Kurtosis & Entropy"] = run_kurtosis_entropy_detector(epochs)

    # ── General ML Detectors ───────────────────────────────────────────
    with st.spinner("Running Isolation Forest..."):
        ml_scores["Isolation Forest"] = run_isolation_forest(features, contamination=contamination)

    with st.spinner("Running Local Outlier Factor..."):
        ml_scores["LOF"] = run_lof(features, contamination=contamination)

    with st.spinner("Running One-Class SVM..."):
        ml_scores["OCSVM"] = run_ocsvm(features, nu=contamination)

    # DL (optional)
    if _check_torch():
        with st.spinner("Training Autoencoder..."):
            ae_scores = run_autoencoder(features, epochs=30)
            if ae_scores is not None:
                ml_scores["Autoencoder"] = ae_scores
    else:
        st.markdown(
            '<div class="dep-warning">'
            '<strong>Note:</strong> Deep learning features require PyTorch. '
            'Install with <code>pip install torch</code>.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Ensemble ───────────────────────────────────────────────────────
    all_scores = {**eeg_scores, **ml_scores}
    all_scores["Ensemble"] = ensemble_scores(all_scores)

    st.session_state["anomaly_scores"] = all_scores
    st.session_state["anomaly_eeg_scores"] = eeg_scores
    st.session_state["anomaly_ml_scores"] = ml_scores
    st.success(
        f"Detection complete! **{len(all_scores) - 1} detectors** run on "
        f"**{len(features)} epochs** ({epoch_dur}s each)."
    )

# ── Visualization ──────────────────────────────────────────────────────────
all_scores = st.session_state.get("anomaly_scores")
if all_scores is None:
    st.info("Configure settings above and click **Run Anomaly Detection** to begin.")
    st.stop()

eeg_scores = st.session_state.get("anomaly_eeg_scores", {})
ml_scores = st.session_state.get("anomaly_ml_scores", {})
epoch_dur = st.session_state.get("anomaly_epoch_dur", 2.0)
n_epochs = len(list(all_scores.values())[0])
time_axis = np.arange(n_epochs) * epoch_dur

# Resolve video events for overlays
_video_events = st.session_state.get("video_events", [])
_video_offset = float(st.session_state.get("video_time_offset_s", 0.0))

# Resolve trigger info for overlays
# trigger_time_s is in the VR/video experiment timeline; apply the same
# sync offset so it aligns with the EEG time axis.
trigger_info = st.session_state.get("trigger_info")
_trigger_time_raw = trigger_info["trigger_time_s"] if trigger_info else None
_trigger_time = (_trigger_time_raw + _video_offset) if _trigger_time_raw is not None else None
_trigger_group = trigger_info["group"] if trigger_info else None
_trigger_color = (
    "#f0883e" if _trigger_group == "Auditory" else "#58a6ff"
) if _trigger_group else None
_trigger_icon = (
    "🔊" if _trigger_group == "Auditory" else "👁️"
) if _trigger_group else None

# Threshold slider
threshold = st.slider(
    "Anomaly threshold (percentile)",
    min_value=80, max_value=99, value=95, step=1,
)

# ── Tabbed Results ─────────────────────────────────────────────────────────
tab_list = [
    "🧠 EEG-Specific Detectors",
    "🤖 General ML Detectors",
    "🗺️ Channel × Epoch Heatmap",
    "📋 Anomalous Epoch Gallery",
]
if _trigger_time is not None:
    tab_list.append("⚡ Trigger Analysis")
if _video_events:
    tab_list.append("🎥 Video-Anomaly Analysis")
tab_list.append("📊 Feature Importance")

tabs = st.tabs(tab_list)

# Map tab names to tab objects for clarity
tab_eeg = tabs[0]
tab_ml = tabs[1]
tab_heatmap = tabs[2]
tab_gallery = tabs[3]
_tab_idx = 4
tab_trigger = None
tab_video = None
if _trigger_time is not None:
    tab_trigger = tabs[_tab_idx]
    _tab_idx += 1
if _video_events:
    tab_video = tabs[_tab_idx]
    _tab_idx += 1
tab_importance = tabs[-1]


def _plot_detector_group(score_dict, colors, threshold_pct):
    """Plot a group of detector scores as subplots with optional trigger overlay."""
    names = list(score_dict.keys())
    n = len(names)
    if n == 0:
        st.info("No detectors in this group.")
        return

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=names,
        vertical_spacing=max(0.03, 0.15 / n),
    )
    for i, (name, sc) in enumerate(score_dict.items()):
        if sc is None or len(sc) == 0:
            continue
        thresh_val = np.percentile(sc, threshold_pct)
        is_anomaly = sc > thresh_val

        fig.add_trace(go.Scatter(
            x=time_axis, y=sc, mode="lines",
            name=name, line=dict(color=colors[i % len(colors)], width=1.2),
            showlegend=False,
        ), row=i + 1, col=1)

        # Highlight anomalies
        fig.add_trace(go.Scatter(
            x=time_axis[is_anomaly], y=sc[is_anomaly], mode="markers",
            name=f"{name} anomalies",
            marker=dict(color="#ff7b72", size=6, symbol="x"),
            showlegend=False,
        ), row=i + 1, col=1)

        fig.add_hline(
            y=thresh_val, line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            row=i + 1, col=1,
        )

        # Trigger overlay
        if _trigger_time is not None:
            fig.add_vline(
                x=_trigger_time, line_dash="dash",
                line_color=_trigger_color, line_width=1.5,
                row=i + 1, col=1,
            )
            # Shaded ±5s window around trigger
            fig.add_vrect(
                x0=_trigger_time - 5, x1=_trigger_time + 5,
                fillcolor=_trigger_color, opacity=0.07,
                line_width=0, row=i + 1, col=1,
            )
            # Annotation only on the first subplot to avoid clutter
            if i == 0:
                fig.add_annotation(
                    x=_trigger_time, y=1.05, yref="paper",
                    text=f"{_trigger_icon} Trigger ({_trigger_group})",
                    showarrow=False, font=dict(color=_trigger_color, size=13),
                )

        # Video event overlays
        if _video_events:
            add_video_event_overlays(
                fig, _video_events,
                row=i + 1, col=1,
                show_labels=(i == 0),  # labels only on first subplot
                max_labels=12,
                time_offset_s=_video_offset,
            )

    # Add colour-coded event strip along the top (only once for the figure)
    if _video_events:
        strip = build_event_strip_trace(_video_events, y_position=0, time_offset_s=_video_offset)
        if strip:
            # Use a secondary y-axis pinned to paper coordinates via an
            # invisible subplot is complex; simpler to add as annotations.
            # Instead, add coloured markers at the top of each subplot using
            # the first subplot's y range.
            fig.add_trace(go.Scattergl(
                x=strip["x"],
                y=[np.max(list(score_dict.values())[0]) * 1.15] * len(strip["x"]),
                mode="markers",
                marker=strip["marker"],
                text=strip["text"],
                hoverinfo="text",
                name="Video Events",
                showlegend=True,
            ), row=1, col=1)

    top_margin = 110 if _video_events else 60
    fig.update_layout(
        template="plotly_dark",
        height=max(350, n * 180),
        margin=dict(l=60, r=20, t=top_margin, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab: EEG-Specific ─────────────────────────────────────────────────────
with tab_eeg:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        These detectors use <strong>EEG domain knowledge</strong> — amplitude statistics,
        spectral band ratios, and signal complexity measures — to identify
        physiologically implausible segments.
        </p>
        """,
        unsafe_allow_html=True,
    )
    _plot_detector_group(eeg_scores, ["#3fb950", "#f0883e", "#a371f7"], threshold)

# ── Tab: General ML ───────────────────────────────────────────────────────
with tab_ml:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        These detectors apply <strong>unsupervised machine learning</strong> to a
        multi-dimensional feature space (band powers, Hjorth parameters, line length)
        extracted from each epoch. They learn the "normal" distribution and flag outliers.
        </p>
        """,
        unsafe_allow_html=True,
    )
    _plot_detector_group(ml_scores, ["#58a6ff", "#f0883e", "#a371f7", "#3fb950"], threshold)

# ── Tab: Heatmap ──────────────────────────────────────────────────────────
with tab_heatmap:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        This heatmap shows the <strong>Z-score of peak amplitude</strong> for each
        channel in each epoch. Hot spots reveal which channels and time segments are
        most anomalous.
        </p>
        """,
        unsafe_allow_html=True,
    )
    epochs_obj = st.session_state.get("anomaly_epochs_obj")
    if epochs_obj is not None:
        data = epochs_obj.get_data()  # (n_epochs, n_channels, n_times)
        peak_amp = np.max(np.abs(data), axis=2)  # (n_epochs, n_channels)
        mu = np.mean(peak_amp, axis=0, keepdims=True)
        sigma = np.std(peak_amp, axis=0, keepdims=True) + 1e-12
        z_matrix = np.abs((peak_amp - mu) / sigma)  # (n_epochs, n_channels)

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_matrix.T,
            x=[f"{t:.1f}s" for t in time_axis],
            y=rec.ch_names[:data.shape[1]],
            colorscale="YlOrRd",
            colorbar=dict(title="|Z-score|"),
        ))
        fig_hm.update_layout(
            template="plotly_dark",
            height=max(350, data.shape[1] * 30),
            xaxis_title="Epoch start time",
            yaxis_title="Channel",
            margin=dict(l=100, r=20, t=20, b=60),
        )
        # Trigger overlay on heatmap
        if _trigger_time is not None:
            # Find the x-axis tick label closest to trigger
            closest_idx = int(np.argmin(np.abs(time_axis - _trigger_time)))
            fig_hm.add_vline(
                x=closest_idx, line_dash="dash",
                line_color=_trigger_color, line_width=2,
                annotation_text=f"{_trigger_icon} Trigger",
                annotation_font_color=_trigger_color,
                annotation_font_size=11,
            )
        # Video event overlays on heatmap
        if _video_events:
            for ev in _video_events:
                eeg_ts_hm = ev.timestamp_s + _video_offset
                closest_idx_v = int(np.argmin(np.abs(time_axis - eeg_ts_hm)))
                color_v = EVENT_COLORS.get(ev.event_type, "#8b949e")
                fig_hm.add_vline(
                    x=closest_idx_v, line_dash="dot",
                    line_color=color_v, line_width=1.2, opacity=0.7,
                )
            # Single legend annotation
            fig_hm.add_annotation(
                x=0.5, y=-0.12, xref="paper", yref="paper",
                text=" | ".join(
                    f'<span style="color:{c}">{EVENT_ICONS[t]} {t}</span>'
                    for t, c in EVENT_COLORS.items()
                ),
                showarrow=False, font=dict(size=10),
            )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No epoch data available.")

# ── Tab: Epoch Gallery ────────────────────────────────────────────────────
with tab_gallery:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        Raw EEG waveforms for the <strong>most anomalous epochs</strong> so you can
        visually inspect what was flagged. Red epochs = highest ensemble anomaly score.
        </p>
        """,
        unsafe_allow_html=True,
    )
    epochs_obj = st.session_state.get("anomaly_epochs_obj")
    ensemble_sc = all_scores.get("Ensemble", np.array([]))

    if epochs_obj is not None and len(ensemble_sc) > 0:
        n_show = st.slider("Number of top anomalous epochs to display", 3, 10, 5)
        top_idx = np.argsort(ensemble_sc)[-n_show:][::-1]

        data = epochs_obj.get_data()
        ch_names = rec.ch_names[:data.shape[1]]
        t_epoch = np.arange(data.shape[2]) / rec.sfreq

        for rank, ep_idx in enumerate(top_idx):
            # Check if this epoch contains or neighbours the trigger
            ep_start = ep_idx * epoch_dur
            ep_end = ep_start + epoch_dur
            near_trigger = False
            trigger_label = ""
            if _trigger_time is not None:
                if ep_start - 5 <= _trigger_time <= ep_end + 5:
                    near_trigger = True
                    trigger_label = f"  {_trigger_icon} Near trigger"

            with st.expander(
                f"#{rank+1}  —  Epoch {ep_idx}  "
                f"(t = {ep_idx * epoch_dur:.1f}s,  "
                f"score = {ensemble_sc[ep_idx]:.3f}){trigger_label}",
                expanded=(rank == 0),
            ):
                fig_ep = go.Figure()
                n_ch_show = min(8, len(ch_names))
                offset = 0
                offsets = []
                for ch_i in range(n_ch_show):
                    sig = data[ep_idx, ch_i, :] * 1e6  # V → µV
                    fig_ep.add_trace(go.Scattergl(
                        x=t_epoch, y=sig + offset,
                        mode="lines", name=ch_names[ch_i],
                        line=dict(width=0.8),
                    ))
                    offsets.append(offset)
                    offset += np.ptp(sig) * 1.3 + 50

                fig_ep.update_layout(
                    template="plotly_dark",
                    height=max(250, n_ch_show * 45),
                    xaxis_title="Time (s)",
                    yaxis=dict(
                        tickvals=offsets,
                        ticktext=ch_names[:n_ch_show],
                    ),
                    showlegend=False,
                    margin=dict(l=80, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_ep, use_container_width=True, key=f"gallery_{ep_idx}")
    else:
        st.info("No ensemble scores available.")

# ── Tab: Trigger Analysis ─────────────────────────────────────────────────
if tab_trigger is not None:
  with tab_trigger:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
                    border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px;
                    margin-bottom: 1.2rem; line-height: 1.6;">
            <h4 style="color: #58a6ff; margin: 0 0 6px 0; font-size: 1rem;">
                ⚡ Trigger–Anomaly Relationship Analysis
            </h4>
            <p style="color: #8b949e; font-size: 0.84rem; margin: 0;">
                This section analyses whether detected anomalies are temporally
                associated with the VR experiment trigger (auditory alarm or visual
                cue). It includes: <strong>trigger-locked anomaly response curves</strong>,
                <strong>pre/post statistical comparisons</strong> (Mann-Whitney U,
                Cohen's d), <strong>permutation-based coincidence tests</strong>,
                <strong>spectral band power shifts</strong>, and an
                <strong>anomaly proximity distribution</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Settings for trigger analysis
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        tlar_pre = st.number_input("Pre-trigger window (s)", 5.0, 60.0, 15.0, 5.0, key="tlar_pre")
    with t_col2:
        tlar_post = st.number_input("Post-trigger window (s)", 5.0, 120.0, 30.0, 5.0, key="tlar_post")
    with t_col3:
        stat_window = st.number_input("Stats comparison window (s)", 10.0, 120.0, 30.0, 5.0, key="stat_window")

    st.markdown("---")

    # ── 1. Trigger-Locked Anomaly Response (TLAR) ──────────────────────
    st.subheader("1. Trigger-Locked Anomaly Response (TLAR)")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Anomaly scores aligned '
        'to trigger onset (time 0). Analogous to Event-Related Potential (ERP) '
        'analysis but applied to anomaly scores instead of raw voltage.</p>',
        unsafe_allow_html=True,
    )

    tlar = compute_trigger_locked_scores(
        all_scores, epoch_dur, _trigger_time,
        window_pre=tlar_pre, window_post=tlar_post,
    )
    tlar_time = tlar["time_axis"]
    tlar_scores = tlar["scores"]

    if len(tlar_time) > 0:
        # Plot ensemble + individual detectors
        fig_tlar = go.Figure()

        # Individual detectors (thin, low opacity)
        det_colors = {
            "Z-Score Threshold": "#3fb950",
            "Spectral Ratio": "#f0883e",
            "Kurtosis & Entropy": "#a371f7",
            "Isolation Forest": "#58a6ff",
            "LOF": "#d2a8ff",
            "OCSVM": "#79c0ff",
            "Autoencoder": "#ffa657",
        }
        for name, sc in tlar_scores.items():
            if name == "Ensemble":
                continue
            fig_tlar.add_trace(go.Scatter(
                x=tlar_time, y=sc, mode="lines",
                name=name, line=dict(
                    color=det_colors.get(name, "#8b949e"),
                    width=0.8, dash="dot",
                ),
                opacity=0.5,
            ))

        # Ensemble (bold)
        if "Ensemble" in tlar_scores:
            ens = tlar_scores["Ensemble"]
            fig_tlar.add_trace(go.Scatter(
                x=tlar_time, y=ens, mode="lines",
                name="Ensemble", line=dict(color="#ff7b72", width=2.5),
            ))

            # Bootstrap 95% CI band (fast: resample from nearby epochs)
            if len(ens) >= 5:
                n_boot = 200
                rng = np.random.default_rng(42)
                boot_means = np.zeros((n_boot, len(ens)))
                for b in range(n_boot):
                    idx = rng.choice(len(ens), size=len(ens), replace=True)
                    boot_means[b] = ens[idx]
                ci_lo = np.percentile(boot_means, 2.5, axis=0)
                ci_hi = np.percentile(boot_means, 97.5, axis=0)
                fig_tlar.add_trace(go.Scatter(
                    x=np.concatenate([tlar_time, tlar_time[::-1]]),
                    y=np.concatenate([ci_hi, ci_lo[::-1]]),
                    fill="toself", fillcolor="rgba(255,123,114,0.12)",
                    line=dict(width=0), showlegend=False, name="95% CI",
                ))

        # Trigger line at time 0
        fig_tlar.add_vline(
            x=0, line_dash="dash", line_color=_trigger_color, line_width=2,
            annotation_text=f"{_trigger_icon} Trigger",
            annotation_font_color=_trigger_color,
        )
        fig_tlar.add_vrect(
            x0=-5, x1=5, fillcolor=_trigger_color, opacity=0.06, line_width=0,
        )

        fig_tlar.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="Time relative to trigger (s)",
            yaxis_title="Anomaly score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        st.plotly_chart(fig_tlar, use_container_width=True)
    else:
        st.warning("Trigger falls outside the recorded epoch range.")

    st.markdown("---")

    # ── 2. Pre / Post Trigger Statistical Comparison ───────────────────
    st.subheader("2. Pre / Post Trigger Statistical Comparison")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Mann-Whitney U test '
        'compares anomaly score distributions before vs. after trigger onset. '
        'Cohen\'s d quantifies effect size. '
        f'Window: <strong>{stat_window:.0f}s</strong> pre and post trigger.</p>',
        unsafe_allow_html=True,
    )

    stats_df = pre_post_trigger_test(
        all_scores, epoch_dur, _trigger_time,
        pre_window=stat_window, post_window=stat_window,
    )

    if not stats_df.empty:
        # Bar chart: pre vs post mean per detector
        fig_prepost = go.Figure()
        fig_prepost.add_trace(go.Bar(
            name="Pre-trigger", x=stats_df["Detector"], y=stats_df["Pre Mean"],
            marker_color="#58a6ff", opacity=0.8,
        ))
        fig_prepost.add_trace(go.Bar(
            name="Post-trigger", x=stats_df["Detector"], y=stats_df["Post Mean"],
            marker_color="#ff7b72", opacity=0.8,
        ))

        # Add significance stars
        for i, row in stats_df.iterrows():
            pv = row["p-value"]
            if isinstance(pv, str):
                continue
            y_pos = max(row["Pre Mean"], row["Post Mean"]) * 1.08
            star = ""
            if pv < 0.001:
                star = "***"
            elif pv < 0.01:
                star = "**"
            elif pv < 0.05:
                star = "*"
            if star:
                fig_prepost.add_annotation(
                    x=row["Detector"], y=y_pos, text=star,
                    showarrow=False, font=dict(color="#ffa657", size=14),
                )

        fig_prepost.update_layout(
            template="plotly_dark", barmode="group", height=380,
            yaxis_title="Mean anomaly score",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=20, t=40, b=80),
        )
        st.plotly_chart(fig_prepost, use_container_width=True)

        # Stats table
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Interpretation
        sig_count = sum(
            1 for _, r in stats_df.iterrows()
            if isinstance(r["p-value"], (int, float)) and r["p-value"] < 0.05
        )
        total = len(stats_df)
        if sig_count > 0:
            st.success(
                f"**{sig_count}/{total}** detectors show statistically significant "
                f"(p < 0.05) differences in anomaly scores between pre- and post-trigger windows."
            )
        else:
            st.info(
                "No detectors show statistically significant pre/post trigger differences "
                "at p < 0.05."
            )
    else:
        st.warning("Insufficient epochs in the pre/post trigger windows for statistical testing.")

    st.markdown("---")

    # ── 3. Trigger–Anomaly Temporal Coincidence ────────────────────────
    st.subheader("3. Trigger–Anomaly Temporal Coincidence")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Permutation test '
        '(n=1000) assessing whether flagged anomalies cluster near the trigger '
        'more than expected by chance. Fold enrichment &gt; 1 indicates '
        'spatial clustering; p-value indicates statistical significance.</p>',
        unsafe_allow_html=True,
    )

    coin_df = trigger_coincidence_test(
        all_scores, epoch_dur, _trigger_time,
        threshold_pct=float(threshold),
        windows=[5.0, 10.0, 15.0],
    )

    if not coin_df.empty:
        st.dataframe(coin_df, use_container_width=True, hide_index=True)

        # Highlight significant coincidences
        sig_coins = coin_df[coin_df["p-value (perm)"] < 0.05]
        if not sig_coins.empty:
            st.success(
                f"**{len(sig_coins)}** detector–window combinations show "
                "statistically significant clustering of anomalies near the trigger "
                "(permutation p < 0.05)."
            )
        else:
            st.info("No significant trigger–anomaly clustering detected at p < 0.05.")
    else:
        st.warning("No flagged anomalies to test for coincidence.")

    st.markdown("---")

    # ── 4. Spectral Band Power Shift ───────────────────────────────────
    st.subheader("4. Spectral Band Power Shift at Trigger")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Comparison of EEG '
        'frequency band powers before vs. after trigger. In VR evacuation '
        'paradigms, expect <strong>Alpha suppression</strong> (increased '
        'alertness), <strong>Beta increase</strong> (cognitive load), and '
        'potential <strong>Theta changes</strong> (stress/anxiety response). '
        'Mann-Whitney U test per band.</p>',
        unsafe_allow_html=True,
    )

    epochs_obj = st.session_state.get("anomaly_epochs_obj")
    if epochs_obj is not None:
        band_shift = compute_trigger_band_shift(
            epochs_obj, epoch_dur, _trigger_time,
            pre_window=stat_window, post_window=stat_window,
        )

        if band_shift["pre_epoch_powers"].shape[0] > 0 and band_shift["post_epoch_powers"].shape[0] > 0:
            band_labels = [b.split(" (")[0] for b in band_shift["bands"]]

            fig_bands = go.Figure()
            fig_bands.add_trace(go.Bar(
                name="Pre-trigger", x=band_labels, y=band_shift["pre_powers"],
                marker_color="#58a6ff", opacity=0.8,
            ))
            fig_bands.add_trace(go.Bar(
                name="Post-trigger", x=band_labels, y=band_shift["post_powers"],
                marker_color="#ff7b72", opacity=0.8,
            ))

            # Significance stars
            for i, (bp, pv) in enumerate(zip(band_labels, band_shift["p_values"])):
                y_pos = max(band_shift["pre_powers"][i], band_shift["post_powers"][i]) * 1.1
                star = ""
                if pv < 0.001:
                    star = "***"
                elif pv < 0.01:
                    star = "**"
                elif pv < 0.05:
                    star = "*"
                if star:
                    fig_bands.add_annotation(
                        x=bp, y=y_pos, text=star,
                        showarrow=False, font=dict(color="#ffa657", size=14),
                    )

            fig_bands.update_layout(
                template="plotly_dark", barmode="group", height=380,
                yaxis_title="Mean band power (µV²/Hz)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=20, t=40, b=60),
            )
            st.plotly_chart(fig_bands, use_container_width=True)

            # Table of p-values
            band_table = pd.DataFrame({
                "Band": band_shift["bands"],
                "Pre Mean Power": [f"{v:.6f}" for v in band_shift["pre_powers"]],
                "Post Mean Power": [f"{v:.6f}" for v in band_shift["post_powers"]],
                "Change %": [
                    f"{((post - pre) / (pre + 1e-12)) * 100:.1f}"
                    for pre, post in zip(band_shift["pre_powers"], band_shift["post_powers"])
                ],
                "p-value": [f"{p:.4f}" for p in band_shift["p_values"]],
            })
            st.dataframe(band_table, use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient epochs in the pre/post trigger windows for spectral analysis.")
    else:
        st.warning("No epoch data available for spectral analysis.")

    st.markdown("---")

    # ── 5. Anomaly Proximity Distribution ──────────────────────────────
    st.subheader("5. Anomaly Proximity Distribution")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Distribution of '
        'temporal distances from flagged anomaly epochs to the trigger event. '
        'A concentration near zero indicates anomalies cluster around the trigger. '
        'Kolmogorov-Smirnov test compares against a uniform baseline.</p>',
        unsafe_allow_html=True,
    )

    ensemble_sc_trig = all_scores.get("Ensemble", np.array([]))
    if len(ensemble_sc_trig) > 0:
        thresh_val_trig = np.percentile(ensemble_sc_trig, threshold)
        flagged_mask = ensemble_sc_trig > thresh_val_trig
        flagged_times = time_axis[flagged_mask]

        if len(flagged_times) > 2:
            distances = flagged_times - _trigger_time

            fig_prox = go.Figure()
            fig_prox.add_trace(go.Histogram(
                x=distances, nbinsx=30, name="Anomaly distances",
                marker_color="#ff7b72", opacity=0.7,
            ))

            # KDE overlay
            from scipy.stats import gaussian_kde, kstest
            try:
                kde = gaussian_kde(distances)
                x_kde = np.linspace(distances.min(), distances.max(), 200)
                y_kde = kde(x_kde) * len(distances) * (distances.max() - distances.min()) / 30
                fig_prox.add_trace(go.Scatter(
                    x=x_kde, y=y_kde, mode="lines", name="KDE",
                    line=dict(color="#ffa657", width=2),
                ))
            except Exception:
                pass

            # Trigger at 0
            fig_prox.add_vline(
                x=0, line_dash="dash", line_color=_trigger_color, line_width=2,
                annotation_text=f"{_trigger_icon} Trigger",
                annotation_font_color=_trigger_color,
            )

            fig_prox.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Time relative to trigger (s)",
                yaxis_title="Count",
                margin=dict(l=60, r=20, t=30, b=60),
            )
            st.plotly_chart(fig_prox, use_container_width=True)

            # KS test: flagged distances vs. uniform over recording duration
            from scipy.stats import kstest as ks_test
            total_dur = time_axis[-1] + epoch_dur
            normed_distances = (flagged_times) / total_dur
            ks_stat, ks_p = ks_test(normed_distances, "uniform")

            ks_col1, ks_col2 = st.columns(2)
            ks_col1.metric("KS statistic", f"{ks_stat:.4f}")
            ks_col2.metric("KS p-value", f"{ks_p:.4f}")

            if ks_p < 0.05:
                st.success(
                    "The distribution of anomaly positions is **significantly non-uniform** "
                    f"(KS p = {ks_p:.4f}), suggesting anomalies are not randomly distributed "
                    "across the recording."
                )
            else:
                st.info(
                    "The distribution of anomaly positions is **not significantly different** "
                    "from uniform (randomly distributed)."
                )
        else:
            st.info("Too few flagged anomalies to compute a proximity distribution.")

# ── Tab: Video-Anomaly Analysis ───────────────────────────────────────────
if tab_video is not None:
  with tab_video:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
                    border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px;
                    margin-bottom: 1.2rem; line-height: 1.6;">
            <h4 style="color: #f0883e; margin: 0 0 6px 0; font-size: 1rem;">
                🎥 Multi-Modal Video–EEG Anomaly Analysis
            </h4>
            <p style="color: #8b949e; font-size: 0.84rem; margin: 0;">
                This section correlates <strong>behavioural events</strong> extracted from the
                VR screen recording with <strong>EEG anomaly scores</strong>. It answers the
                key research question: <em>Are detected EEG anomalies temporally associated with
                observable behavioural events (head movements, gaze shifts, sudden actions)?</em>
                Analyses include a unified multi-modal timeline, temporal coincidence testing,
                event-locked ERP averaging, and pre/post event anomaly comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Settings
    va_col1, va_col2 = st.columns(2)
    with va_col1:
        coincidence_window = st.number_input(
            "Coincidence window (±s)", 1.0, 15.0, 3.0, 1.0, key="va_coinc_win",
            help="Time window around each video event to check for anomaly epochs.",
        )
    with va_col2:
        erp_pre = st.number_input("ERP pre-event (s)", 0.5, 5.0, 1.0, 0.5, key="va_erp_pre")
        erp_post = st.number_input("ERP post-event (s)", 0.5, 10.0, 2.0, 0.5, key="va_erp_post")

    st.markdown("---")

    # ── 1. Multi-Modal Timeline ────────────────────────────────────────
    st.subheader("1. Multi-Modal Timeline")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Unified visualisation '
        'showing ensemble anomaly scores (top) and video events (bottom), aligned '
        'on the same time axis. Coloured vertical lines connect behavioural events '
        'to their temporal position in the EEG anomaly landscape.</p>',
        unsafe_allow_html=True,
    )

    ensemble_sc = all_scores.get("Ensemble", np.array([]))
    if len(ensemble_sc) > 0:
        fig_timeline = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["Ensemble Anomaly Score", "Behavioural Events"],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
        )

        # Top: Ensemble score line
        thresh_val_tl = np.percentile(ensemble_sc, threshold)
        is_anom_tl = ensemble_sc > thresh_val_tl

        fig_timeline.add_trace(go.Scatter(
            x=time_axis, y=ensemble_sc, mode="lines",
            name="Ensemble Score", line=dict(color="#ff7b72", width=1.5),
        ), row=1, col=1)

        fig_timeline.add_trace(go.Scatter(
            x=time_axis[is_anom_tl], y=ensemble_sc[is_anom_tl],
            mode="markers", name="Anomalies",
            marker=dict(color="#ff7b72", size=7, symbol="x"),
        ), row=1, col=1)

        fig_timeline.add_hline(
            y=thresh_val_tl, line_dash="dash",
            line_color="rgba(255,255,255,0.3)", row=1, col=1,
        )

        # Bottom: Event strip
        for ev in _video_events:
            color = EVENT_COLORS.get(ev.event_type, "#8b949e")
            icon_e = EVENT_ICONS.get(ev.event_type, "📌")
            sev_h = SEVERITY_SIZE.get(ev.severity, 8)
            eeg_ts = ev.timestamp_s + _video_offset  # align to EEG time

            fig_timeline.add_trace(go.Scatter(
                x=[eeg_ts], y=[ev.event_type],
                mode="markers+text",
                marker=dict(color=color, size=sev_h, symbol="diamond"),
                text=[icon_e],
                textposition="top center",
                textfont=dict(size=10),
                hovertext=f"{icon_e} {ev.description[:60]}<br>Severity: {ev.severity}<br>Video t={ev.timestamp_s:.1f}s  EEG t={eeg_ts:.1f}s",
                hoverinfo="text",
                showlegend=False,
            ), row=2, col=1)

            # Connecting vertical line spanning both subplots
            fig_timeline.add_vline(
                x=eeg_ts, line_dash="dot",
                line_color=color, line_width=0.8, opacity=0.5,
            )

        # Trigger overlay if available
        if _trigger_time is not None:
            fig_timeline.add_vline(
                x=_trigger_time, line_dash="dash",
                line_color=_trigger_color, line_width=2,
            )
            fig_timeline.add_annotation(
                x=_trigger_time, y=1.05, yref="paper",
                text=f"{_trigger_icon} Trigger",
                showarrow=False, font=dict(color=_trigger_color, size=13),
            )

        fig_timeline.update_layout(
            template="plotly_dark", height=500,
            margin=dict(l=60, r=20, t=90, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_timeline.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_timeline.update_yaxes(title_text="Score", row=1, col=1)
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # ── 2. Temporal Coincidence Analysis ───────────────────────────────
    st.subheader("2. Video Event–Anomaly Temporal Coincidence")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">For each video event, '
        'checks whether an EEG anomaly epoch falls within the specified time window. '
        'A permutation test (n=1000) assesses if the observed coincidence rate '
        'exceeds chance expectation.</p>',
        unsafe_allow_html=True,
    )

    if len(ensemble_sc) > 0:
        coincidence = compute_event_anomaly_coincidence(
            _video_events, ensemble_sc, epoch_dur,
            window_s=coincidence_window,
            threshold_pct=float(threshold),
            time_offset_s=_video_offset,
        )

        # Summary metrics
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Total Events", len(_video_events))
        c_col2.metric("Coincident", coincidence["observed_count"])
        c_col3.metric(
            "Coincidence Rate",
            f"{coincidence['overall_rate']:.1%}",
            delta=f"vs {coincidence['expected_rate']:.1%} expected",
            delta_color="normal",
        )
        c_col4.metric("p-value", f"{coincidence['p_value']:.4f}")

        if coincidence["p_value"] < 0.05:
            fold = coincidence["overall_rate"] / max(coincidence["expected_rate"], 1e-6)
            st.success(
                f"EEG anomalies cluster significantly around video events "
                f"(p = {coincidence['p_value']:.4f}, {fold:.1f}× enrichment). "
                f"This suggests detected anomalies are **behaviourally driven**."
            )
        else:
            st.info(
                "No significant temporal clustering of anomalies around video events "
                f"(p = {coincidence['p_value']:.4f})."
            )

        # Per-event coincidence table
        pe_df = pd.DataFrame(coincidence["per_event"])
        pe_df["coincides"] = pe_df["coincides"].map({True: "✅ Yes", False: "❌ No"})
        st.dataframe(pe_df, use_container_width=True, hide_index=True)

        # By event type bar chart
        if coincidence["by_type"]:
            bt = coincidence["by_type"]
            types_list = list(bt.keys())
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Bar(
                x=[f"{EVENT_ICONS.get(t, '📌')} {t}" for t in types_list],
                y=[bt[t]["rate"] * 100 for t in types_list],
                marker_color=[EVENT_COLORS.get(t, "#8b949e") for t in types_list],
                text=[f"{bt[t]['n_coincident']}/{bt[t]['n_events']}" for t in types_list],
                textposition="auto",
            ))
            fig_bt.update_layout(
                template="plotly_dark", height=350,
                yaxis_title="Coincidence Rate (%)",
                xaxis_title="Event Type",
                margin=dict(l=60, r=20, t=20, b=80),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("---")

    # ── 3. Event-Locked ERP Analysis ───────────────────────────────────
    st.subheader("3. Event-Locked EEG Response (ERP)")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Grand-average EEG '
        'response time-locked to each type of video event, analogous to '
        'Event-Related Potential (ERP) analysis. Shows the average neural '
        'response pattern around behavioural events. Shaded regions indicate '
        '±1 SD across events. Requires ≥2 events of a given type.</p>',
        unsafe_allow_html=True,
    )

    erp_data = compute_event_locked_erp(
        _video_events, raw,
        pre_s=erp_pre, post_s=erp_post,
        time_offset_s=_video_offset,
    )

    if erp_data:
        for etype, erpd in erp_data.items():
            icon_e = EVENT_ICONS.get(etype, "📌")
            color_e = EVENT_COLORS.get(etype, "#8b949e")
            n_ev = erpd["n_events"]
            caveat = " ⚠️ *Low N — interpret with caution*" if n_ev < 5 else ""

            st.markdown(
                f"#### {icon_e} {etype} (N = {n_ev}){caveat}"
            )

            grand_avg = erpd["grand_avg"] * 1e6  # V → µV
            std_dev = erpd["std"] * 1e6
            t_erp = erpd["time_axis"]
            ch_names_erp = erpd["ch_names"]
            n_ch_show = min(8, len(ch_names_erp))

            fig_erp = go.Figure()
            for ch_i in range(n_ch_show):
                fig_erp.add_trace(go.Scatter(
                    x=t_erp, y=grand_avg[ch_i],
                    mode="lines", name=ch_names_erp[ch_i],
                    line=dict(width=1.2),
                ))
                # ±1 SD band
                fig_erp.add_trace(go.Scatter(
                    x=np.concatenate([t_erp, t_erp[::-1]]),
                    y=np.concatenate([
                        grand_avg[ch_i] + std_dev[ch_i],
                        (grand_avg[ch_i] - std_dev[ch_i])[::-1],
                    ]),
                    fill="toself",
                    fillcolor=f"rgba(255,255,255,0.04)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Event onset line at t=0
            fig_erp.add_vline(
                x=0, line_dash="dash", line_color=color_e, line_width=2,
                annotation_text=f"{icon_e} Event onset",
                annotation_font_color=color_e,
            )

            fig_erp.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Time relative to event (s)",
                yaxis_title="Amplitude (µV)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=20, t=40, b=60),
            )
            st.plotly_chart(fig_erp, use_container_width=True, key=f"erp_{etype}")
    else:
        st.info("Not enough events (≥2 of the same type) within the recording range for ERP analysis.")

    st.markdown("---")

    # ── 4. Pre / Post Event Anomaly Comparison ─────────────────────────
    st.subheader("4. Pre / Post Event Anomaly Comparison")
    st.markdown(
        '<p style="color: #8b949e; font-size: 0.85rem;">Mann-Whitney U test '
        'comparing anomaly score distributions before vs. after video events, '
        'grouped by event type. Cohen\'s d quantifies effect size. Positive '
        'change indicates anomaly scores increase after the behavioural event, '
        'suggesting the event elicited a neural anomaly response.</p>',
        unsafe_allow_html=True,
    )

    event_stats_df = compute_event_anomaly_stats(
        _video_events, all_scores, epoch_dur,
        window_s=coincidence_window * 2,
        time_offset_s=_video_offset,
    )

    if not event_stats_df.empty:
        # Pivot: bar chart of Change % by event type and detector
        etypes_in_stats = event_stats_df["Event Type"].unique()
        for et in etypes_in_stats:
            subset = event_stats_df[event_stats_df["Event Type"] == et]
            st.markdown(f"**{et}** (N = {subset.iloc[0]['N Events']})")

            fig_pp = go.Figure()
            fig_pp.add_trace(go.Bar(
                name="Pre-event", x=subset["Detector"], y=subset["Pre Mean"],
                marker_color="#58a6ff", opacity=0.8,
            ))
            fig_pp.add_trace(go.Bar(
                name="Post-event", x=subset["Detector"], y=subset["Post Mean"],
                marker_color="#ff7b72", opacity=0.8,
            ))

            for _, row in subset.iterrows():
                pv = row["p-value"]
                if isinstance(pv, str):
                    continue
                y_pos = max(row["Pre Mean"], row["Post Mean"]) * 1.08
                star = ""
                if pv < 0.001:
                    star = "***"
                elif pv < 0.01:
                    star = "**"
                elif pv < 0.05:
                    star = "*"
                if star:
                    fig_pp.add_annotation(
                        x=row["Detector"], y=y_pos, text=star,
                        showarrow=False, font=dict(color="#ffa657", size=14),
                    )

            fig_pp.update_layout(
                template="plotly_dark", barmode="group", height=350,
                yaxis_title="Mean anomaly score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=20, t=30, b=80),
            )
            st.plotly_chart(fig_pp, use_container_width=True, key=f"pp_{et}")

        # Full stats table
        st.dataframe(event_stats_df, use_container_width=True, hide_index=True)

        # Interpretation
        sig_rows = event_stats_df[
            event_stats_df["p-value"].apply(
                lambda x: isinstance(x, (int, float)) and x < 0.05
            )
        ]
        if len(sig_rows) > 0:
            st.success(
                f"**{len(sig_rows)}** event-type × detector combinations show "
                "statistically significant (p < 0.05) pre/post anomaly score changes. "
                "This supports the hypothesis that these behavioural events are "
                "associated with measurable neural responses."
            )
        else:
            st.info(
                "No statistically significant pre/post differences found (p < 0.05)."
            )
    else:
        st.info(
            "Insufficient data for pre/post analysis. Need ≥2 events of the same type "
            "with enough surrounding epochs."
        )

# ── Tab: Feature Importance ───────────────────────────────────────────────
with tab_importance:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        Feature importances from the <strong>Isolation Forest</strong> model,
        showing which extracted features contribute most to anomaly scoring.
        </p>
        """,
        unsafe_allow_html=True,
    )
    features = st.session_state.get("anomaly_features")
    if features is not None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(features.values)
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X)

        # Average absolute split importances across trees
        importances = np.zeros(X.shape[1])
        for tree in clf.estimators_:
            fi = tree.feature_importances_
            importances += fi
        importances /= len(clf.estimators_)

        # Top 20
        top_n = min(20, len(importances))
        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_names = [features.columns[i] for i in top_idx]
        top_values = importances[top_idx]

        fig_fi = go.Figure(go.Bar(
            x=top_values[::-1],
            y=top_names[::-1],
            orientation="h",
            marker_color="#58a6ff",
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            height=max(400, top_n * 28),
            xaxis_title="Feature Importance",
            margin=dict(l=200, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("No feature data available.")

# ── Summary ────────────────────────────────────────────────────────────────
st.header("📋 Detection Summary")

ensemble_sc = all_scores.get("Ensemble", np.array([]))
if len(ensemble_sc) > 0:
    thresh_val = np.percentile(ensemble_sc, threshold)
    n_flagged = int(np.sum(ensemble_sc > thresh_val))

    # Metrics row — extend if trigger is available
    if _trigger_time is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Epochs", n_epochs)
        col2.metric("Flagged Epochs", n_flagged)
        col3.metric("Anomaly Rate", f"{n_flagged / n_epochs * 100:.1f}%")
        col4.metric("Detectors Used", len(all_scores) - 1)
        col5.metric(
            f"{_trigger_icon} Trigger",
            f"{_trigger_time:.1f}s",
            delta=_trigger_group,
            delta_color="off",
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Epochs", n_epochs)
        col2.metric("Flagged Epochs", n_flagged)
        col3.metric("Anomaly Rate", f"{n_flagged / n_epochs * 100:.1f}%")
        col4.metric("Detectors Used", len(all_scores) - 1)

    # Per-detector comparison table
    st.subheader("Per-Detector Breakdown")

    # Pre-compute coincidence counts if trigger loaded
    _near_trigger_counts = {}
    if _trigger_time is not None:
        near_mask = (time_axis >= _trigger_time - 10) & (time_axis <= _trigger_time + 10)
        for name, sc in all_scores.items():
            if sc is None or len(sc) == 0 or name == "Ensemble":
                continue
            tv = np.percentile(sc, threshold)
            _near_trigger_counts[name] = int(np.sum((sc > tv) & near_mask))

    rows = []
    for name, sc in all_scores.items():
        if sc is None or len(sc) == 0 or name == "Ensemble":
            continue
        tv = np.percentile(sc, threshold)
        nf = int(np.sum(sc > tv))
        category = "EEG-Specific" if name in eeg_scores else "General ML"
        row_data = {
            "Detector": name,
            "Category": category,
            "Flagged Epochs": nf,
            "Flagged %": f"{nf / n_epochs * 100:.1f}%",
            "Mean Score": f"{np.mean(sc):.4f}",
            "Max Score": f"{np.max(sc):.4f}",
        }
        if _trigger_time is not None:
            row_data["Near Trigger (±10s)"] = _near_trigger_counts.get(name, 0)
        rows.append(row_data)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

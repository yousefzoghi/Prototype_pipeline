"""
Raw EEG Viewer — Interactive multi-channel time-series visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Raw EEG Viewer", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
from utils.signal_processing import DEFAULT_MAPPING
render_sidebar()
render_sidebar_footer()

from analysis.video_analysis import (
    add_video_event_overlays, EVENT_COLORS, EVENT_ICONS,
)

st.title("Raw EEG Viewer")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([3, 1])

with col_ctrl1:
    selected_channels = st.multiselect(
        "Channels to display",
        options=rec.ch_names,
        default=rec.ch_names[:8],
        format_func=lambda x: DEFAULT_MAPPING.get(x, x),
    )

with col_ctrl2:
    offset_scale = st.slider(
        "Channel offset (µV)",
        min_value=0,
        max_value=5000,
        value=500,
        step=50,
    )
    show_measurements = st.checkbox("Show amplitude ticks", value=True)

# Time range (Range Slider)
total_duration = rec.metadata.get("duration_s", len(rec.data) / rec.sfreq)
time_range = st.slider(
    "Time range (seconds)",
    min_value=0.0,
    max_value=float(total_duration),
    value=(0.0, min(10.0, float(total_duration))),
    step=0.1,
)

if not selected_channels:
    st.info("Select at least one channel to display.")
    st.stop()

# ── Build plot ─────────────────────────────────────────────────────────────
t_start, t_end = time_range
start_sample = int(t_start * rec.sfreq)
end_sample = int(t_end * rec.sfreq)
end_sample = min(end_sample, len(rec.data))

# Initialize 'tag' column if missing
if "tag" not in rec.data.columns:
    rec.data["tag"] = "none"

time_axis = np.arange(start_sample, end_sample) / rec.sfreq

fig = go.Figure()

for i, ch in enumerate(selected_channels):
    ch_display_name = DEFAULT_MAPPING.get(ch, ch)
    y = rec.data[ch].values[start_sample:end_sample]
    y_offset = y + i * offset_scale  # Stack channels vertically

    fig.add_trace(go.Scattergl(
        x=time_axis,
        y=y_offset,
        mode="lines",
        name=ch_display_name,
        line=dict(width=0.8),
        hovertemplate=f"{ch_display_name}<br>Time: %{{x:.3f}} s<br>Amplitude: %{{customdata:.1f}} µV",
        customdata=y,
    ))

# ── Add Annotation Overlays ──────────────────────────────────────────────
rec_name = st.session_state.get("active_rec_name", "default")
if "annotations_dict" not in st.session_state:
    st.session_state["annotations_dict"] = {}
if rec_name not in st.session_state["annotations_dict"]:
    st.session_state["annotations_dict"][rec_name] = []

annotations = st.session_state["annotations_dict"][rec_name]
t_start_vis = float(time_axis[0]) if len(time_axis) > 0 else 0.0
t_end_vis = float(time_axis[-1]) if len(time_axis) > 0 else 0.0

for ann in annotations:
    # Check if annotation overlaps with visible window
    if ann["start"] <= t_end_vis and ann["end"] >= t_start_vis:
        fig.add_vrect(
            x0=max(ann["start"], t_start_vis),
            x1=min(ann["end"], t_end_vis),
            fillcolor="yellow",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text=ann["label"],
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="yellow")
        )

fig.update_layout(
    template="plotly_dark",
    height=max(400, len(selected_channels) * 60),
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (µV, offset)",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=40, b=60),
    yaxis=dict(
        title=dict(text="Amplitude (µV, offset)", font=dict(size=14)),
        tickvals=[i * offset_scale for i in range(len(selected_channels))],
        ticktext=[DEFAULT_MAPPING.get(ch, ch) for ch in selected_channels],
        showline=True,
        linewidth=1,
        linecolor="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
    ),
    dragmode="zoom",
)

# ── Video event overlays ──────────────────────────────────────────────────
_video_events = st.session_state.get("video_events", [])
_video_offset = float(st.session_state.get("video_time_offset_s", 0.0))
if _video_events:
    t_start = float(time_axis[0]) if len(time_axis) > 0 else 0.0
    t_end = float(time_axis[-1]) if len(time_axis) > 0 else 0.0
    # Filter by EEG-aligned timestamp (video_time + offset)
    visible_events = [
        ev for ev in _video_events
        if t_start <= (ev.timestamp_s + _video_offset) <= t_end
    ]
    if visible_events:
        add_video_event_overlays(
            fig, visible_events,
            time_range=(t_start, t_end),
            show_labels=True,
            max_labels=10,
            time_offset_s=_video_offset,
        )
        fig.update_layout(margin=dict(l=60, r=20, t=110, b=60))

st.plotly_chart(fig, use_container_width=True, key="raw_viewer")

# Video event legend (below plot)
if _video_events:
    offset_note = f" &nbsp;|&nbsp; ⏱️ offset: <strong>{_video_offset:+.1f}s</strong>" if _video_offset != 0.0 else ""
    legend_html = (
        '<div style="background: #161b22; border: 1px solid #30363d; '
        'border-radius: 8px; padding: 8px 14px; margin-bottom: 1rem;">'
        '<span style="color: #8b949e; font-size: 0.8rem; margin-right: 10px;">'
        f'<strong>Video Events:</strong>{offset_note}&nbsp;&nbsp;</span>'
    )
    for etype, color in EVENT_COLORS.items():
        icon = EVENT_ICONS.get(etype, "📌")
        legend_html += (
            f'<span style="color: {color}; font-size: 0.8rem; margin-right: 10px;">'
            f'{icon} {etype}</span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

# ── Annotation tools ───────────────────────────────────────────────────────
with st.expander("Annotation Tools", expanded=True):
    st.markdown("Mark segments of interest for downstream analysis. Tags are saved directly to the data.")
    
    # Ensure annotations_dict is initialized
    rec_name = st.session_state.get("active_rec_name", "default")
    if "annotations_dict" not in st.session_state:
        st.session_state["annotations_dict"] = {}
    if rec_name not in st.session_state["annotations_dict"]:
        st.session_state["annotations_dict"][rec_name] = []

    # Initialize 'tag' column if missing
    if "tag" not in rec.data.columns:
        rec.data["tag"] = "none"
        # Apply existing annotations to the dataframe if they exist in dictionary
        for ann in st.session_state["annotations_dict"][rec_name]:
            s_idx = int(ann["start"] * rec.sfreq)
            e_idx = int(ann["end"] * rec.sfreq)
            rec.data.iloc[s_idx:e_idx, rec.data.columns.get_loc("tag")] = ann["label"]

    ann_col1, ann_col2, ann_col3 = st.columns(3)
    ann_start = ann_col1.number_input("Start (s)", min_value=0.0, max_value=total_duration, value=t_start, step=0.5)
    ann_end = ann_col2.number_input("End (s)", min_value=0.0, max_value=total_duration, value=min(t_start + 2.0, total_duration), step=0.5)
    ann_label = ann_col3.text_input("Label", value="event")

    if st.button("Apply Annotation & Update Data", use_container_width=True):
        if ann_start >= ann_end:
            st.error("Start time must be before end time.")
        else:
            # Update Session State List
            st.session_state["annotations_dict"][rec_name].append({
                "start": ann_start, "end": ann_end, "label": ann_label
            })
            
            # Update the actual dataframe
            start_idx = int(ann_start * rec.sfreq)
            end_idx = int(ann_end * rec.sfreq)
            # Clip indices
            start_idx = max(0, min(start_idx, len(rec.data)-1))
            end_idx = max(0, min(end_idx, len(rec.data)))
            
            rec.data.iloc[start_idx:end_idx, rec.data.columns.get_loc("tag")] = ann_label
            
            st.success(f"Tagged samples from {ann_start:.1f}s to {ann_end:.1f}s as '{ann_label}'")
            st.rerun()

    current_anns = st.session_state["annotations_dict"][rec_name]
    if current_anns:
        st.markdown("### Current Annotations")
        st.dataframe(current_anns, use_container_width=True)
        if st.button("Clear All Annotations"):
            st.session_state["annotations_dict"][rec_name] = []
            rec.data["tag"] = "none"
            st.rerun()

    # Download button for modified data
    csv = rec.data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Tagged Data (CSV)",
        data=csv,
        file_name=f"tagged_{st.session_state.get('active_rec_name', 'eeg_data.csv')}",
        mime='text/csv',
    )

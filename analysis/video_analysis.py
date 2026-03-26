"""
video_analysis.py — VR screen-recording analysis via Vision-Language Models
(Gemini) with EEG-anomaly temporal alignment, event-locked ERP computation,
and statistical coincidence testing for multi-modal VR evacuation research.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoEvent:
    """A single behavioural event extracted from VR screen recording."""
    timestamp_s: float
    event_type: str          # head_movement | gaze_shift | sudden_action | interaction | environmental
    description: str
    severity: str = "minor"  # minor | moderate | major
    end_timestamp_s: float | None = None
    objects: list[str] = field(default_factory=list)


# Consistent colour & icon encoding used across all pages
EVENT_COLORS: dict[str, str] = {
    "head_movement":  "#f0883e",
    "gaze_shift":     "#58a6ff",
    "sudden_action":  "#ff7b72",
    "interaction":    "#3fb950",
    "environmental":  "#d2a8ff",
}

EVENT_ICONS: dict[str, str] = {
    "head_movement":  "🔄",
    "gaze_shift":     "👁️",
    "sudden_action":  "⚡",
    "interaction":    "🤚",
    "environmental":  "🌐",
}

EVENT_DASH_STYLES: dict[str, str] = {
    "head_movement":  "dash",
    "gaze_shift":     "dot",
    "sudden_action":  "solid",
    "interaction":    "dashdot",
    "environmental":  "longdash",
}

SEVERITY_SIZE: dict[str, int] = {
    "minor":    8,
    "moderate": 12,
    "major":    16,
}


# ---------------------------------------------------------------------------
# Gemini VLM integration
# ---------------------------------------------------------------------------

def build_gemini_prompt(custom_context: str = "", video_duration_s: float | None = None) -> str:
    """Construct an engineered prompt for Gemini video analysis.

    The prompt establishes VR evacuation research context, requests structured
    JSON output conforming to our VideoEvent schema, and includes a few-shot
    example for reliable parsing.
    """
    duration_note = ""
    if video_duration_s is not None:
        duration_note = (
            f"\n\n**CRITICAL: This video is {video_duration_s:.1f} seconds long. "
            f"All timestamps MUST be between 0 and {video_duration_s:.1f}. "
            f"Do NOT generate any timestamp beyond {video_duration_s:.1f} seconds. "
            f"Use the video player timeline to determine accurate timestamps.**"
        )

    prompt = f"""You are an expert research assistant analysing VR (Virtual Reality) screen recordings from an evacuation behaviour study. Participants wear an EEG headset while navigating a virtual environment. We need precise behavioural event extraction to correlate with neural signals.{duration_note}

**Your task**: Analyse this VR screen recording and identify ALL observable behavioural events. For each event, provide:

1. **timestamp_s** — Start time in seconds from the beginning of the video (float, e.g. 3.5). Use the video playback timeline to determine timestamps accurately.
2. **end_timestamp_s** — End time in seconds if the event has duration (float or null)
3. **event_type** — One of: `head_movement`, `gaze_shift`, `sudden_action`, `interaction`, `environmental`
4. **description** — Concise description of what happened (1-2 sentences)
5. **severity** — `minor` (routine), `moderate` (notable), or `major` (sudden/dramatic change)
6. **objects** — List of objects/elements the participant interacts with or observes

**Event type definitions:**
- `head_movement`: Any rotation, pan, tilt, or nod of the participant's viewpoint
- `gaze_shift`: Sustained change in visual focus direction (>0.5s fixation on new target)
- `sudden_action`: Rapid, unexpected movement — quick head snap, startle response, abrupt direction change
- `interaction`: Participant engages with an object (picks up, pushes, opens door, etc.)
- `environmental`: Change in the virtual environment itself (alarm sounds, lights change, smoke appears, door opens automatically, fire/emergency cue)

**Important guidelines:**
- Timestamps must be in seconds from video start (float precision), based on the actual video playback position
- ALL timestamps must fall within the actual video duration — do NOT extrapolate beyond the video
- Capture ALL head movements, not just dramatic ones — subtle scans matter for EEG correlation
- Mark severity=major for any event that could trigger a neural startle/orienting response
- For head movements, describe direction (left, right, up, down, clockwise rotation)
- Group very rapid consecutive micro-movements into one event with a time range
- Note any environmental changes that could trigger the P300 or N200 ERP components

**Output format**: Return ONLY a JSON array. No markdown formatting, no code fences, no explanation before or after. Just the raw JSON array.

**Example output:**
[
  {{"timestamp_s": 0.0, "end_timestamp_s": 1.2, "event_type": "head_movement", "description": "Participant pans head slowly to the right, scanning the room from left wall to right wall.", "severity": "minor", "objects": ["wall", "window"]}},
  {{"timestamp_s": 1.5, "end_timestamp_s": null, "event_type": "gaze_shift", "description": "Gaze fixates on exit sign above the door.", "severity": "moderate", "objects": ["exit_sign", "door"]}},
  {{"timestamp_s": 3.2, "end_timestamp_s": 3.4, "event_type": "sudden_action", "description": "Rapid head snap to the left in response to a sound cue.", "severity": "major", "objects": []}},
  {{"timestamp_s": 3.2, "end_timestamp_s": null, "event_type": "environmental", "description": "Fire alarm begins sounding in the virtual environment.", "severity": "major", "objects": ["alarm"]}}
]"""

    if custom_context.strip():
        prompt += f"\n\n**Additional research context:**\n{custom_context.strip()}"

    return prompt


def _get_video_duration(video_bytes: bytes) -> float | None:
    """Try to determine video duration in seconds. Returns None on failure."""
    import tempfile, subprocess, json as _json
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", tmp_path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            info = _json.loads(result.stdout)
            return float(info["format"]["duration"])
    except Exception:
        pass
    finally:
        try:
            import os
            os.unlink(tmp_path)
        except Exception:
            pass
    return None


def analyze_video_with_gemini(
    video_bytes: bytes,
    api_key: str,
    model: str = "gemini-flash-latest",
    custom_prompt: str = "",
    mime_type: str = "video/mp4",
) -> dict[str, str]:
    """Send video to Gemini and return structured analysis.

    Returns dict with keys:
      - ``events_text``: Raw JSON text of events
      - ``scene_summary``: Narrative summary of the recording
      - ``video_duration_s``: Detected video duration (or None)
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Try to detect video duration for timestamp validation
    video_duration_s = _get_video_duration(video_bytes)

    prompt = build_gemini_prompt(custom_prompt, video_duration_s=video_duration_s)

    # First call: extract events
    event_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type=mime_type, data=video_bytes),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    event_response = client.models.generate_content(
        model=model,
        contents=event_contents,
        config=config,
    )
    events_text = event_response.text

    # Second call: scene narrative summary
    summary_prompt = """Based on the VR screen recording you just analysed, provide a concise narrative summary (3-5 paragraphs) describing:

1. **Scene Overview**: What virtual environment the participant is in, key objects and layout
2. **Behavioural Narrative**: What the participant does chronologically — their exploration pattern, attention focus, and decision-making behaviour
3. **Notable Moments**: Any sudden reactions, hesitations, or changes in behaviour that may correlate with elevated neural activity (EEG anomalies)
4. **Evacuation Relevance**: If applicable, describe any evacuation-related behaviour — wayfinding, response to alarms/cues, route selection

Write as a research observation report suitable for inclusion in a scientific paper."""

    summary_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type=mime_type, data=video_bytes),
                types.Part.from_text(text=summary_prompt),
            ],
        ),
    ]

    summary_response = client.models.generate_content(
        model=model,
        contents=summary_contents,
        config=config,
    )

    return {
        "events_text": events_text,
        "scene_summary": summary_response.text,
        "video_duration_s": video_duration_s,
    }


def parse_gemini_response(response_text: str) -> list[VideoEvent]:
    """Parse Gemini's JSON response into a sorted list of VideoEvents.

    Handles markdown code fences, extra text before/after JSON, and falls
    back to regex-based timestamp extraction if JSON parsing fails.
    """
    events: list[VideoEvent] = []

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Try to find JSON array in text
    json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ts = item.get("timestamp_s")
                    if ts is None:
                        continue
                    try:
                        ts = float(ts)
                    except (ValueError, TypeError):
                        continue

                    et = item.get("end_timestamp_s")
                    if et is not None:
                        try:
                            et = float(et)
                        except (ValueError, TypeError):
                            et = None

                    event_type = str(item.get("event_type", "head_movement")).lower().strip()
                    valid_types = set(EVENT_COLORS.keys())
                    if event_type not in valid_types:
                        # Best-effort mapping
                        if "head" in event_type or "movement" in event_type:
                            event_type = "head_movement"
                        elif "gaze" in event_type or "look" in event_type:
                            event_type = "gaze_shift"
                        elif "sudden" in event_type or "startle" in event_type:
                            event_type = "sudden_action"
                        elif "interact" in event_type:
                            event_type = "interaction"
                        else:
                            event_type = "environmental"

                    severity = str(item.get("severity", "minor")).lower().strip()
                    if severity not in ("minor", "moderate", "major"):
                        severity = "minor"

                    objects = item.get("objects", [])
                    if not isinstance(objects, list):
                        objects = [str(objects)] if objects else []
                    objects = [str(o) for o in objects]

                    events.append(VideoEvent(
                        timestamp_s=ts,
                        end_timestamp_s=et,
                        event_type=event_type,
                        description=str(item.get("description", "")),
                        severity=severity,
                        objects=objects,
                    ))
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction of timestamps
    if not events:
        ts_pattern = re.compile(
            r"(\d{1,2}:\d{2}(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d{1,2}:\d{2}(?:\.\d+)?)"
            r"|(?:(?:at|@)\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec)"
        )
        for m in ts_pattern.finditer(response_text):
            if m.group(1):  # MM:SS range
                start_s = _mmss_to_seconds(m.group(1))
                end_s = _mmss_to_seconds(m.group(2))
            elif m.group(3):  # plain seconds
                start_s = float(m.group(3))
                end_s = None
            else:
                continue

            # Grab surrounding text as description (up to 200 chars)
            ctx_start = max(0, m.start() - 100)
            ctx_end = min(len(response_text), m.end() + 100)
            desc = response_text[ctx_start:ctx_end].strip()
            desc = re.sub(r"\s+", " ", desc)

            events.append(VideoEvent(
                timestamp_s=start_s,
                end_timestamp_s=end_s,
                event_type="head_movement",
                description=desc[:200],
                severity="moderate",
            ))

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp_s)
    return events


def clamp_events_to_duration(
    events: list[VideoEvent],
    duration_s: float,
) -> list[VideoEvent]:
    """Remove or clamp events whose timestamps exceed the video duration.

    Events starting beyond the video length are dropped.
    End timestamps are clamped to duration_s.
    """
    valid = []
    for ev in events:
        if ev.timestamp_s > duration_s:
            continue  # hallucinated timestamp beyond video
        if ev.end_timestamp_s is not None and ev.end_timestamp_s > duration_s:
            ev.end_timestamp_s = duration_s
        valid.append(ev)
    return valid


def _mmss_to_seconds(val: str) -> float:
    """Convert MM:SS.S to seconds."""
    parts = val.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(val)


# ---------------------------------------------------------------------------
# Event ↔ Anomaly statistical analysis
# ---------------------------------------------------------------------------

def compute_event_anomaly_coincidence(
    events: list[VideoEvent],
    scores: np.ndarray,
    epoch_dur: float,
    window_s: float = 2.0,
    n_permutations: int = 1000,
    rng_seed: int = 42,
    threshold_pct: float = 95.0,
    time_offset_s: float = 0.0,
) -> dict:
    """Test whether EEG anomalies cluster around video events.

    For each event, checks if any anomaly epoch falls within ±window_s.
    Computes permutation p-value comparing observed coincidence rate
    against chance.

    Parameters
    ----------
    time_offset_s : float
        Seconds to add to every video timestamp to convert from video
        timeline to EEG timeline (EEG_time = video_time + time_offset_s).

    Returns dict with:
      - ``per_event``: list of dicts, one per event, with coincidence info
      - ``overall_rate``: fraction of events coinciding with an anomaly
      - ``expected_rate``: expected rate under null hypothesis
      - ``p_value``: permutation-based p-value
      - ``by_type``: dict of event_type → coincidence rate
    """
    n_epochs = len(scores)
    epoch_starts = np.arange(n_epochs) * epoch_dur
    thresh_val = np.percentile(scores, threshold_pct)
    is_anomaly = scores > thresh_val

    rng = np.random.default_rng(rng_seed)

    per_event = []
    for ev in events:
        eeg_ts = ev.timestamp_s + time_offset_s  # convert to EEG time
        near_mask = (epoch_starts >= eeg_ts - window_s) & (
            epoch_starts <= eeg_ts + window_s
        )
        coincides = bool(np.any(is_anomaly & near_mask))
        nearest_anomaly_idx = None
        nearest_anomaly_dist = None
        if np.any(is_anomaly):
            anomaly_times = epoch_starts[is_anomaly]
            dists = np.abs(anomaly_times - eeg_ts)
            nearest_idx = np.argmin(dists)
            nearest_anomaly_dist = float(dists[nearest_idx])
            nearest_anomaly_idx = int(np.where(is_anomaly)[0][nearest_idx])

        per_event.append({
            "timestamp_s": ev.timestamp_s,
            "eeg_time_s": eeg_ts,
            "event_type": ev.event_type,
            "severity": ev.severity,
            "description": ev.description[:80],
            "coincides": coincides,
            "nearest_anomaly_epoch": nearest_anomaly_idx,
            "nearest_anomaly_dist_s": nearest_anomaly_dist,
        })

    observed_count = sum(1 for pe in per_event if pe["coincides"])
    overall_rate = observed_count / max(len(events), 1)

    # Permutation test: shuffle anomaly labels, recount coincidences
    perm_counts = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_labels = rng.permutation(is_anomaly)
        count = 0
        for ev in events:
            near_mask = (epoch_starts >= ev.timestamp_s - window_s) & (
                epoch_starts <= ev.timestamp_s + window_s
            )
            if np.any(perm_labels & near_mask):
                count += 1
        perm_counts[p] = count

    expected_rate = float(np.mean(perm_counts)) / max(len(events), 1)
    p_value = float(np.mean(perm_counts >= observed_count))

    # By event type
    by_type: dict[str, dict] = {}
    for etype in set(ev.event_type for ev in events):
        type_events = [pe for pe in per_event if pe["event_type"] == etype]
        n_type = len(type_events)
        n_coinc = sum(1 for pe in type_events if pe["coincides"])
        by_type[etype] = {
            "n_events": n_type,
            "n_coincident": n_coinc,
            "rate": n_coinc / max(n_type, 1),
        }

    return {
        "per_event": per_event,
        "overall_rate": overall_rate,
        "observed_count": observed_count,
        "expected_rate": expected_rate,
        "p_value": p_value,
        "by_type": by_type,
    }


def compute_event_locked_erp(
    events: list[VideoEvent],
    raw,
    pre_s: float = 1.0,
    post_s: float = 2.0,
    event_types: list[str] | None = None,
    time_offset_s: float = 0.0,
) -> dict:
    """Compute event-locked grand-average EEG (ERP-like) around video events.

    For each event type with sufficient occurrences (≥2), extracts EEG segments
    [event_time - pre_s, event_time + post_s] and averages across events.

    Parameters
    ----------
    events : list[VideoEvent]
    raw : mne.io.RawArray
    pre_s : pre-event window in seconds
    post_s : post-event window in seconds
    event_types : optional list of event types to include (default: all)
    time_offset_s : float
        Seconds to add to every video timestamp to align with EEG timeline.

    Returns
    -------
    dict with keys per event_type:
      ``time_axis``, ``grand_avg`` (n_channels, n_times),
      ``individual`` (n_events, n_channels, n_times),
      ``ch_names``, ``n_events``, ``std`` (n_channels, n_times)
    """
    sfreq = raw.info["sfreq"]
    n_pre = int(pre_s * sfreq)
    n_post = int(post_s * sfreq)
    total_samples = n_pre + n_post
    data_full = raw.get_data()  # (n_channels, n_times)
    ch_names = raw.ch_names
    max_sample = data_full.shape[1]

    if event_types is None:
        event_types = list(set(ev.event_type for ev in events))

    result: dict = {}

    for etype in event_types:
        type_events = [ev for ev in events if ev.event_type == etype]
        if len(type_events) < 2:
            continue

        segments = []
        for ev in type_events:
            eeg_ts = ev.timestamp_s + time_offset_s
            center_sample = int(eeg_ts * sfreq)
            start = center_sample - n_pre
            end = center_sample + n_post
            if start < 0 or end > max_sample:
                continue
            seg = data_full[:, start:end]  # (n_channels, n_times)
            segments.append(seg)

        if len(segments) < 2:
            continue

        stacked = np.stack(segments, axis=0)  # (n_events, n_channels, n_times)

        time_axis = np.linspace(-pre_s, post_s, total_samples)

        result[etype] = {
            "time_axis": time_axis,
            "grand_avg": np.mean(stacked, axis=0),      # (n_channels, n_times)
            "std": np.std(stacked, axis=0),              # (n_channels, n_times)
            "individual": stacked,                        # (n_events, n_channels, n_times)
            "ch_names": ch_names,
            "n_events": len(segments),
        }

    return result


def compute_event_anomaly_stats(
    events: list[VideoEvent],
    score_dict: dict[str, np.ndarray],
    epoch_dur: float,
    window_s: float = 5.0,
    time_offset_s: float = 0.0,
) -> pd.DataFrame:
    """Compare anomaly scores before vs after each event type.

    Adapts the pre_post_trigger_test pattern for multiple video events,
    grouped by event_type.

    Parameters
    ----------
    time_offset_s : float
        Seconds to add to every video timestamp to align with EEG timeline.

    Returns DataFrame with: Event Type, N Events, Detector, Pre Mean,
    Post Mean, Change %, Mann-Whitney U, p-value, Cohen d
    """
    n_epochs = len(list(score_dict.values())[0])
    epoch_starts = np.arange(n_epochs) * epoch_dur

    rows = []
    event_types = sorted(set(ev.event_type for ev in events))

    for etype in event_types:
        type_events = [ev for ev in events if ev.event_type == etype]
        if len(type_events) < 2:
            continue

        for det_name, sc in score_dict.items():
            if sc is None or len(sc) == 0 or det_name == "Ensemble":
                continue

            # Collect pre and post scores across all events of this type
            pre_vals_all = []
            post_vals_all = []
            for ev in type_events:
                eeg_ts = ev.timestamp_s + time_offset_s
                pre_mask = (epoch_starts >= eeg_ts - window_s) & (
                    epoch_starts < eeg_ts
                )
                post_mask = (epoch_starts >= eeg_ts) & (
                    epoch_starts < eeg_ts + window_s
                )
                pre_vals_all.extend(sc[pre_mask].tolist())
                post_vals_all.extend(sc[post_mask].tolist())

            pre_arr = np.array(pre_vals_all)
            post_arr = np.array(post_vals_all)

            if len(pre_arr) < 3 or len(post_arr) < 3:
                continue

            pre_mean = float(np.mean(pre_arr))
            post_mean = float(np.mean(post_arr))
            change_pct = ((post_mean - pre_mean) / (pre_mean + 1e-12)) * 100

            try:
                stat, pval = sp_stats.mannwhitneyu(
                    pre_arr, post_arr, alternative="two-sided"
                )
            except ValueError:
                stat, pval = np.nan, np.nan

            pooled_std = np.sqrt(
                ((len(pre_arr) - 1) * np.var(pre_arr, ddof=1)
                 + (len(post_arr) - 1) * np.var(post_arr, ddof=1))
                / (len(pre_arr) + len(post_arr) - 2 + 1e-12)
            )
            cohens_d = (post_mean - pre_mean) / (pooled_std + 1e-12)

            icon = EVENT_ICONS.get(etype, "📌")
            rows.append({
                "Event Type": f"{icon} {etype}",
                "N Events": len(type_events),
                "Detector": det_name,
                "Pre Mean": round(pre_mean, 5),
                "Post Mean": round(post_mean, 5),
                "Change %": round(change_pct, 1),
                "Mann-Whitney U": round(float(stat), 2) if not np.isnan(stat) else "N/A",
                "p-value": round(float(pval), 4) if not np.isnan(pval) else "N/A",
                "Cohen d": round(float(cohens_d), 3),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilities for Plotly overlays (used by multiple pages)
# ---------------------------------------------------------------------------

def add_video_event_overlays(
    fig,
    events: list[VideoEvent],
    time_range: tuple[float, float] | None = None,
    row: int | None = None,
    col: int = 1,
    show_labels: bool = True,
    max_labels: int = 15,
    time_offset_s: float = 0.0,
):
    """Add video event vertical lines and annotations to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    events : list of VideoEvent
    time_offset_s : float
        Seconds to add to every video timestamp when plotting on the EEG
        timeline (EEG_time = video_time + time_offset_s).
    time_range : optional (start, end) to filter events to visible window
    row, col : subplot indices (None for single-plot figures)
    show_labels : whether to add text annotations
    max_labels : limit annotations to avoid clutter
    """
    visible = events
    if time_range is not None:
        visible = [
            e for e in events
            if time_range[0] <= (e.timestamp_s + time_offset_s) <= time_range[1]
        ]

    subplot_kwargs = {}
    if row is not None:
        subplot_kwargs = {"row": row, "col": col}

    label_count = 0
    for ev in visible:
        color = EVENT_COLORS.get(ev.event_type, "#8b949e")
        dash = EVENT_DASH_STYLES.get(ev.event_type, "dash")
        plot_ts = ev.timestamp_s + time_offset_s  # EEG-aligned timestamp

        # Vertical line
        fig.add_vline(
            x=plot_ts,
            line_dash=dash,
            line_color=color,
            line_width=1.2,
            opacity=0.7,
            **subplot_kwargs,
        )

        # Shaded region for events with duration
        if ev.end_timestamp_s is not None:
            fig.add_vrect(
                x0=plot_ts,
                x1=ev.end_timestamp_s + time_offset_s,
                fillcolor=color,
                opacity=0.06,
                line_width=0,
                **subplot_kwargs,
            )

        # Annotation label
        if show_labels and label_count < max_labels:
            icon = EVENT_ICONS.get(ev.event_type, "📌")
            sev_marker = "❗" if ev.severity == "major" else ""
            short_desc = ev.description[:28] + ("…" if len(ev.description) > 28 else "")
            # Stagger horizontally across 5 slots to separate overlapping labels
            ax_offset = (label_count % 5 - 2) * 40
            fig.add_annotation(
                x=plot_ts,
                y=1.10,
                yref="paper",
                text=f"{icon}{sev_marker} {short_desc}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=1.5,
                arrowcolor=color,
                ax=ax_offset,
                ay=-55,
                font=dict(color=color, size=12),
                align="center",
                bgcolor="rgba(13,17,23,0.75)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                **subplot_kwargs,
            )
            label_count += 1


def build_event_strip_trace(
    events: list[VideoEvent],
    y_position: float = 0,
    time_offset_s: float = 0.0,
) -> dict:
    """Create a Plotly Scattergl trace for a colour-coded event strip.

    Parameters
    ----------
    time_offset_s : float
        Seconds added to each timestamp to align with EEG timeline.

    Returns kwargs suitable for ``fig.add_trace(go.Scattergl(**kwargs))``.
    """
    if not events:
        return {}

    timestamps = [ev.timestamp_s + time_offset_s for ev in events]
    colors = [EVENT_COLORS.get(ev.event_type, "#8b949e") for ev in events]
    sizes = [SEVERITY_SIZE.get(ev.severity, 8) for ev in events]
    icons = [EVENT_ICONS.get(ev.event_type, "📌") for ev in events]
    texts = [
        f"{icons[i]} {ev.event_type} ({ev.severity})<br>{ev.description[:60]}"
        for i, ev in enumerate(events)
    ]

    return dict(
        x=timestamps,
        y=[y_position] * len(events),
        mode="markers",
        marker=dict(
            color=colors,
            size=sizes,
            symbol="diamond",
            line=dict(color="white", width=0.5),
        ),
        text=texts,
        hoverinfo="text",
        name="Video Events",
        showlegend=False,
    )


def events_to_dataframe(events: list[VideoEvent], time_offset_s: float = 0.0) -> pd.DataFrame:
    """Convert VideoEvent list to a display-ready DataFrame.

    Parameters
    ----------
    time_offset_s : float
        When non-zero, adds an "EEG Time (s)" column showing the
        EEG-aligned timestamp alongside the original video time.
    """
    rows = []
    for ev in events:
        icon = EVENT_ICONS.get(ev.event_type, "📌")
        row = {
            "Video Time (s)": f"{ev.timestamp_s:.1f}",
            "End (s)": f"{ev.end_timestamp_s:.1f}" if ev.end_timestamp_s else "—",
            "Type": f"{icon} {ev.event_type}",
            "Severity": ev.severity,
            "Description": ev.description,
            "Objects": ", ".join(ev.objects) if ev.objects else "—",
        }
        if time_offset_s != 0.0:
            row["EEG Time (s)"] = f"{ev.timestamp_s + time_offset_s:.1f}"
            # Insert EEG Time right after Video Time
            ordered = {"Video Time (s)": row["Video Time (s)"],
                       "EEG Time (s)": row["EEG Time (s)"]}
            ordered.update({k: v for k, v in row.items()
                            if k not in ("Video Time (s)", "EEG Time (s)")})
            row = ordered
        rows.append(row)
    return pd.DataFrame(rows)


def get_mime_type(filename: str) -> str:
    """Infer MIME type from video file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "mp4": "video/mp4",
        "avi": "video/x-msvideo",
        "mov": "video/quicktime",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
    }.get(ext, "video/mp4")

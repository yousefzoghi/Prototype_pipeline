"""
Tests for analysis.video_analysis — VR screen recording analysis module.

Run from the project root:
    python -m pytest tests/ -v
or:
    python tests/test_video_analysis.py
"""

import sys
import os

# Ensure project root is on the path so `analysis.*` imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from analysis.video_analysis import (
    VideoEvent,
    parse_gemini_response,
    clamp_events_to_duration,
    compute_event_anomaly_coincidence,
    compute_event_anomaly_stats,
    events_to_dataframe,
    get_mime_type,
    build_gemini_prompt,
    build_event_strip_trace,
    EVENT_COLORS,
    EVENT_ICONS,
)


# ---------------------------------------------------------------------------
# parse_gemini_response
# ---------------------------------------------------------------------------

class TestParseGeminiResponse:
    """Test JSON extraction and fallback parsing from Gemini output."""

    def test_clean_json(self):
        response = '[{"timestamp_s": 1.0, "event_type": "head_movement", "description": "Pan right"}]'
        events = parse_gemini_response(response)
        assert len(events) == 1
        assert events[0].timestamp_s == 1.0
        assert events[0].event_type == "head_movement"

    def test_markdown_code_fence(self):
        response = '```json\n[{"timestamp_s": 0.5, "event_type": "gaze_shift", "description": "Look left"}]\n```'
        events = parse_gemini_response(response)
        assert len(events) == 1
        assert events[0].event_type == "gaze_shift"

    def test_multiple_events_sorted(self):
        response = """[
          {"timestamp_s": 5.0, "event_type": "sudden_action", "description": "B"},
          {"timestamp_s": 1.0, "event_type": "head_movement", "description": "A"},
          {"timestamp_s": 3.0, "event_type": "interaction", "description": "C"}
        ]"""
        events = parse_gemini_response(response)
        assert len(events) == 3
        assert events[0].timestamp_s == 1.0
        assert events[1].timestamp_s == 3.0
        assert events[2].timestamp_s == 5.0

    def test_end_timestamp(self):
        response = '[{"timestamp_s": 2.0, "end_timestamp_s": 3.5, "event_type": "head_movement", "description": "Pan"}]'
        events = parse_gemini_response(response)
        assert events[0].end_timestamp_s == 3.5

    def test_null_end_timestamp(self):
        response = '[{"timestamp_s": 2.0, "end_timestamp_s": null, "event_type": "head_movement", "description": "Pan"}]'
        events = parse_gemini_response(response)
        assert events[0].end_timestamp_s is None

    def test_severity_levels(self):
        response = """[
          {"timestamp_s": 0, "event_type": "head_movement", "description": "a", "severity": "minor"},
          {"timestamp_s": 1, "event_type": "head_movement", "description": "b", "severity": "moderate"},
          {"timestamp_s": 2, "event_type": "head_movement", "description": "c", "severity": "major"}
        ]"""
        events = parse_gemini_response(response)
        assert [e.severity for e in events] == ["minor", "moderate", "major"]

    def test_invalid_severity_defaults_to_minor(self):
        response = '[{"timestamp_s": 0, "event_type": "head_movement", "description": "x", "severity": "extreme"}]'
        events = parse_gemini_response(response)
        assert events[0].severity == "minor"

    def test_unknown_event_type_mapping(self):
        """Unknown types should be mapped to nearest valid type."""
        response = '[{"timestamp_s": 0, "event_type": "looking_around", "description": "x"}]'
        events = parse_gemini_response(response)
        assert events[0].event_type in EVENT_COLORS  # must be a valid type

    def test_objects_field(self):
        response = '[{"timestamp_s": 0, "event_type": "interaction", "description": "x", "objects": ["door", "handle"]}]'
        events = parse_gemini_response(response)
        assert events[0].objects == ["door", "handle"]

    def test_objects_empty(self):
        response = '[{"timestamp_s": 0, "event_type": "interaction", "description": "x", "objects": []}]'
        events = parse_gemini_response(response)
        assert events[0].objects == []

    def test_extra_text_around_json(self):
        response = "Here are the events:\n\n```json\n[{\"timestamp_s\": 1.0, \"event_type\": \"head_movement\", \"description\": \"Pan\"}]\n```\n\nHope this helps!"
        events = parse_gemini_response(response)
        assert len(events) == 1

    def test_empty_json_array(self):
        response = "[]"
        events = parse_gemini_response(response)
        assert len(events) == 0

    def test_missing_timestamp_skipped(self):
        response = '[{"event_type": "head_movement", "description": "no timestamp"}]'
        events = parse_gemini_response(response)
        assert len(events) == 0

    def test_non_dict_items_skipped(self):
        response = '[1, "string", {"timestamp_s": 1.0, "event_type": "head_movement", "description": "ok"}]'
        events = parse_gemini_response(response)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# compute_event_anomaly_coincidence
# ---------------------------------------------------------------------------

class TestEventAnomalyCoincidence:
    """Test the permutation-based coincidence analysis."""

    def _make_events(self, timestamps):
        return [
            VideoEvent(timestamp_s=t, event_type="head_movement", description=f"event at {t}")
            for t in timestamps
        ]

    def test_perfect_coincidence(self):
        """When anomalies are exactly at event times, coincidence should be high."""
        scores = np.zeros(50)
        scores[10] = 10.0  # anomaly at epoch 10 → time 20s (epoch_dur=2)
        events = self._make_events([20.0])
        result = compute_event_anomaly_coincidence(
            events, scores, epoch_dur=2.0, window_s=3.0, threshold_pct=90.0,
        )
        assert result["observed_count"] == 1
        assert result["overall_rate"] == 1.0

    def test_no_coincidence(self):
        """When anomalies are far from events, coincidence should be zero."""
        scores = np.zeros(50)
        scores[0] = 10.0  # anomaly at t=0
        events = self._make_events([90.0])  # event far away
        result = compute_event_anomaly_coincidence(
            events, scores, epoch_dur=2.0, window_s=2.0, threshold_pct=90.0,
        )
        assert result["observed_count"] == 0

    def test_by_type_breakdown(self):
        """Results should include per-type breakdown."""
        scores = np.ones(50)
        scores[5] = 10.0
        events = [
            VideoEvent(timestamp_s=10.0, event_type="head_movement", description="a"),
            VideoEvent(timestamp_s=50.0, event_type="gaze_shift", description="b"),
        ]
        result = compute_event_anomaly_coincidence(
            events, scores, epoch_dur=2.0, window_s=2.0, threshold_pct=90.0,
        )
        assert "head_movement" in result["by_type"]
        assert "gaze_shift" in result["by_type"]


# ---------------------------------------------------------------------------
# compute_event_anomaly_stats
# ---------------------------------------------------------------------------

class TestEventAnomalyStats:
    """Test pre/post anomaly score comparisons."""

    def test_returns_dataframe(self):
        events = [
            VideoEvent(timestamp_s=20.0, event_type="head_movement", description="a"),
            VideoEvent(timestamp_s=40.0, event_type="head_movement", description="b"),
        ]
        scores = np.random.default_rng(42).random(50)
        score_dict = {"Z-Score Threshold": scores}
        df = compute_event_anomaly_stats(events, score_dict, epoch_dur=2.0, window_s=5.0)
        assert not df.empty
        assert "Event Type" in df.columns
        assert "p-value" in df.columns

    def test_single_event_type_skipped(self):
        """Types with fewer than 2 events should be skipped."""
        events = [
            VideoEvent(timestamp_s=20.0, event_type="sudden_action", description="once"),
        ]
        scores = np.random.default_rng(42).random(50)
        score_dict = {"Z-Score": scores}
        df = compute_event_anomaly_stats(events, score_dict, epoch_dur=2.0)
        assert df.empty


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_events_to_dataframe(self):
        events = [
            VideoEvent(timestamp_s=1.0, event_type="head_movement", description="Pan", severity="minor", objects=["wall"]),
            VideoEvent(timestamp_s=5.0, event_type="sudden_action", description="Snap", severity="major"),
        ]
        df = events_to_dataframe(events)
        assert len(df) == 2
        assert "Video Time (s)" in df.columns
        assert "Type" in df.columns
        assert "Severity" in df.columns

    def test_get_mime_type(self):
        assert get_mime_type("video.mp4") == "video/mp4"
        assert get_mime_type("recording.avi") == "video/x-msvideo"
        assert get_mime_type("clip.mov") == "video/quicktime"
        assert get_mime_type("test.webm") == "video/webm"
        assert get_mime_type("unknown.xyz") == "video/mp4"  # default

    def test_build_gemini_prompt(self):
        prompt = build_gemini_prompt()
        assert "timestamp_s" in prompt
        assert "head_movement" in prompt
        assert "JSON" in prompt

    def test_build_gemini_prompt_with_context(self):
        prompt = build_gemini_prompt("This is an office VR scenario.")
        assert "office VR scenario" in prompt

    def test_build_gemini_prompt_with_duration(self):
        prompt = build_gemini_prompt(video_duration_s=243.0)
        assert "243.0" in prompt
        assert "CRITICAL" in prompt

    def test_clamp_events_drops_beyond_duration(self):
        events = [
            VideoEvent(timestamp_s=10.0, event_type="head_movement", description="ok"),
            VideoEvent(timestamp_s=300.0, event_type="gaze_shift", description="too far"),
            VideoEvent(timestamp_s=200.0, end_timestamp_s=260.0, event_type="interaction", description="end clamped"),
        ]
        result = clamp_events_to_duration(events, 243.0)
        assert len(result) == 2
        assert result[0].timestamp_s == 10.0
        assert result[1].timestamp_s == 200.0
        assert result[1].end_timestamp_s == 243.0  # clamped

    def test_build_event_strip_trace_empty(self):
        result = build_event_strip_trace([])
        assert result == {}

    def test_build_event_strip_trace(self):
        events = [
            VideoEvent(timestamp_s=1.0, event_type="head_movement", description="a"),
            VideoEvent(timestamp_s=3.0, event_type="sudden_action", description="b"),
        ]
        result = build_event_strip_trace(events)
        assert "x" in result
        assert len(result["x"]) == 2
        assert result["marker"]["color"][0] == EVENT_COLORS["head_movement"]
        assert result["marker"]["color"][1] == EVENT_COLORS["sudden_action"]

    def test_event_colors_and_icons_match(self):
        """All event types should have both a color and an icon."""
        for etype in EVENT_COLORS:
            assert etype in EVENT_ICONS, f"Missing icon for {etype}"
        for etype in EVENT_ICONS:
            assert etype in EVENT_COLORS, f"Missing color for {etype}"


# ---------------------------------------------------------------------------
# Run with `python tests/test_video_analysis.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

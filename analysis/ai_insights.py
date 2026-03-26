"""
ai_insights.py — Clustering, NLP summary, and XAI overlays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_epochs(
    features: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """UMAP + HDBSCAN clustering on epoch features.

    Returns: (embedding_2d, embedding_2d_col2, cluster_labels)
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        raise ImportError(
            "UMAP and HDBSCAN are required for clustering. "
            "Install with: pip install umap-learn hdbscan"
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)

    return embedding[:, 0], embedding[:, 1], labels.tolist()


# ---------------------------------------------------------------------------
# NLP Summary
# ---------------------------------------------------------------------------

def generate_summary(
    rec_metadata: dict,
    ch_names: list[str],
    band_powers: pd.DataFrame | None = None,
    anomaly_scores: np.ndarray | None = None,
    survey_responses: dict | None = None,
    trigger_info: dict | None = None,
    trigger_stats: dict | None = None,
    video_events: list | None = None,
) -> str:
    """Generate a natural-language summary of the EEG recording analysis.

    This is template-driven (no external LLM needed).

    Parameters
    ----------
    trigger_info : dict, optional
        Keys: participant, group, trigger_time_s, explanations
    trigger_stats : dict, optional
        Keys: pre_post_df (DataFrame), coincidence_df (DataFrame),
        band_shift (dict with bands/pre_powers/post_powers/p_values)
    """
    lines = []

    # Recording overview
    fmt = rec_metadata.get("format", "unknown").upper()
    dur = rec_metadata.get("duration_s", 0)
    sfreq = rec_metadata.get("sfreq", 0)
    n_ch = len(ch_names)
    lines.append(
        f"**Recording Overview:** This {fmt}-format recording contains "
        f"{n_ch} EEG channels sampled at {sfreq:.0f} Hz, "
        f"totaling {dur:.1f} seconds of continuous data."
    )

    # Band power analysis
    if band_powers is not None and not band_powers.empty:
        mean_bp = band_powers.mean(axis=0)
        dominant_band = mean_bp.idxmax()
        lines.append(
            f"\n**Spectral Profile:** The dominant frequency band across channels "
            f"is **{dominant_band}** (mean power: {mean_bp[dominant_band]:.2e} V²/Hz)."
        )

        # Per-channel highlights
        for ch in ch_names[:4]:  # First 4 channels
            if ch in band_powers.index:
                ch_dominant = band_powers.loc[ch].idxmax()
                ratio = band_powers.loc[ch, ch_dominant] / band_powers.loc[ch].mean()
                if ratio > 2.0:
                    lines.append(
                        f"- *{ch}* shows elevated **{ch_dominant}** power "
                        f"({ratio:.1f}× mean), potentially indicating localized activity."
                    )

    # Anomaly summary
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        n_anom = np.sum(anomaly_scores > np.percentile(anomaly_scores, 95))
        pct = n_anom / len(anomaly_scores) * 100
        lines.append(
            f"\n**Anomaly Detection:** {n_anom} epochs ({pct:.1f}%) were flagged "
            f"as potentially anomalous (top 5% of anomaly scores)."
        )

        if pct > 10:
            lines.append(
                "- A high anomaly rate may indicate widespread artifacts or "
                "a non-stationary recording. Consider re-examining preprocessing parameters."
            )

    # Survey context
    if survey_responses:
        lines.append("\n**Participant Context:**")
        stress_keys = ["I feel stressed", "I feel anxious", "I feel nervous", "I feel tense"]
        calm_keys = ["I feel calm", "I feel relaxed"]

        stress_scores = [v for k, v in survey_responses.items()
                         if any(sk.lower() in k.lower() for sk in stress_keys) and not np.isnan(v)]
        calm_scores = [v for k, v in survey_responses.items()
                       if any(ck.lower() in k.lower() for ck in calm_keys) and not np.isnan(v)]

        if stress_scores:
            avg_stress = np.mean(stress_scores)
            level = "low" if avg_stress <= -1 else ("moderate" if avg_stress <= 0 else "elevated")
            lines.append(f"- Self-reported stress level: **{level}** (avg score: {avg_stress:.1f}/2)")

        if calm_scores:
            avg_calm = np.mean(calm_scores)
            level = "low" if avg_calm <= -1 else ("moderate" if avg_calm <= 0 else "high")
            lines.append(f"- Self-reported calmness: **{level}** (avg score: {avg_calm:.1f}/2)")

    # Trigger analysis
    if trigger_info is not None:
        t_s = trigger_info["trigger_time_s"]
        grp = trigger_info["group"]
        icon = "🔊" if grp == "Auditory" else "👁️"
        t_min = int(t_s // 60)
        t_sec = t_s % 60
        lines.append(
            f"\n**Trigger Event:** {icon} A **{grp}** trigger was administered at "
            f"**{t_min}:{t_sec:04.1f}** ({t_s:.1f}s) for participant "
            f"**{trigger_info['participant']}**."
        )

        if trigger_stats:
            # Pre/post comparison
            pre_post_df = trigger_stats.get("pre_post_df")
            if pre_post_df is not None and not pre_post_df.empty:
                sig = pre_post_df[
                    pre_post_df["p-value"].apply(
                        lambda x: isinstance(x, (int, float)) and x < 0.05
                    )
                ]
                total = len(pre_post_df)
                lines.append(
                    f"\n**Trigger–Anomaly Relationship:** {len(sig)}/{total} "
                    f"detectors show statistically significant (p < 0.05) "
                    f"changes in anomaly scores after the trigger."
                )
                for _, r in sig.iterrows():
                    direction = "increased" if r["Change %"] > 0 else "decreased"
                    lines.append(
                        f"- {r['Detector']}: scores {direction} by "
                        f"{abs(r['Change %']):.1f}% (p = {r['p-value']:.4f}, "
                        f"Cohen's d = {r['Cohen d']:.2f})"
                    )

            # Coincidence test
            coin_df = trigger_stats.get("coincidence_df")
            if coin_df is not None and not coin_df.empty:
                sig_coins = coin_df[coin_df["p-value (perm)"] < 0.05]
                if not sig_coins.empty:
                    lines.append(
                        f"\n**Temporal Coincidence:** {len(sig_coins)} "
                        f"detector–window combinations show anomalies clustering "
                        f"near the trigger beyond chance levels (permutation p < 0.05)."
                    )

            # Band shift
            band_shift = trigger_stats.get("band_shift")
            if band_shift is not None:
                bands = band_shift.get("bands", [])
                pre_p = band_shift.get("pre_powers", [])
                post_p = band_shift.get("post_powers", [])
                p_vals = band_shift.get("p_values", [])
                sig_bands = [
                    (b, pre, post, p)
                    for b, pre, post, p in zip(bands, pre_p, post_p, p_vals)
                    if p < 0.05
                ]
                if sig_bands:
                    lines.append("\n**Spectral Shift at Trigger:**")
                    for b, pre, post, p in sig_bands:
                        change = ((post - pre) / (pre + 1e-12)) * 100
                        direction = "increase" if change > 0 else "decrease"
                        short_name = b.split(" (")[0]
                        lines.append(
                            f"- **{short_name}**: {abs(change):.1f}% {direction} "
                            f"post-trigger (p = {p:.4f})"
                        )
                    # VR-specific interpretation
                    band_names_lower = [b.lower() for b, _, _, _ in sig_bands]
                    notes = []
                    for b, pre, post, p in sig_bands:
                        bl = b.lower()
                        change_pct = ((post - pre) / (pre + 1e-12)) * 100
                        if "alpha" in bl and change_pct < 0:
                            notes.append(
                                "Alpha suppression suggests **increased alertness/arousal** "
                                "consistent with the VR evacuation stimulus."
                            )
                        if "beta" in bl and change_pct > 0:
                            notes.append(
                                "Beta increase indicates **elevated cognitive load** "
                                "or active processing of the emergency cue."
                            )
                        if "theta" in bl and change_pct > 0:
                            notes.append(
                                "Theta elevation may reflect **increased "
                                "anxiety/stress response** to the trigger."
                            )
                    for n in notes:
                        lines.append(f"- {n}")

        if trigger_info.get("explanations"):
            lines.append("\n*Trigger notes:*")
            for e in trigger_info["explanations"]:
                lines.append(f"- {e}")

    # VR behavioural events from screen recording
    if video_events:
        type_counts: dict[str, int] = {}
        for ev in video_events:
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
        summary_parts = ", ".join(
            f"{count} {etype}" for etype, count in sorted(type_counts.items())
        )
        lines.append(
            f"\n**VR Behavioural Events ({len(video_events)} total):** "
            f"Screen-recording analysis identified: {summary_parts}."
        )
        major_events = [ev for ev in video_events if ev.severity == "major"]
        if major_events:
            sample = "; ".join(
                f"{ev.event_type} @ {ev.timestamp_s:.1f}s"
                for ev in major_events[:6]
            )
            lines.append(
                f"- **{len(major_events)} major events** detected "
                f"(most likely to correlate with EEG responses): {sample}."
            )
        sudden = [ev for ev in video_events if ev.event_type == "sudden_action"]
        if sudden:
            lines.append(
                f"- {len(sudden)} sudden movement/action events — prime candidates "
                "for EEG orienting responses (N200, P300 components)."
            )
        environmental = [ev for ev in video_events if ev.event_type == "environmental"]
        if environmental:
            lines.append(
                f"- {len(environmental)} environmental changes (alarms, stimuli) — "
                "expect P300 and arousal-related alpha suppression near these timestamps."
            )

    return "\n".join(lines)


def generate_llm_summary(
    provider: str,
    model_name: str,
    api_key: str | None,
    rec_metadata: dict,
    ch_names: list[str],
    band_powers: pd.DataFrame | None = None,
    anomaly_scores: np.ndarray | None = None,
    survey_responses: dict | None = None,
    trigger_info: dict | None = None,
    trigger_stats: dict | None = None,
    video_events: list | None = None,
) -> str:
    """Generate an AI summary using an LLM (Ollama or OpenRouter)."""

    # 1. Construct context string
    context = []
    fmt = rec_metadata.get("format", "unknown").upper()
    dur = rec_metadata.get("duration_s", 0)
    sfreq = rec_metadata.get("sfreq", 0)
    context.append(f"Recording: {fmt} format, {len(ch_names)} channels, {dur:.1f}s duration, {sfreq:.0f} Hz.")

    if band_powers is not None and not band_powers.empty:
        mean_bp = band_powers.mean(axis=0)
        dom = mean_bp.idxmax()
        context.append(f"Spectral Analysis: Dominant band across channels is {dom}.")
        # Top channel deviations
        for ch in ch_names[:4]:
            if ch in band_powers.index:
                ch_dom = band_powers.loc[ch].idxmax()
                val = band_powers.loc[ch, ch_dom]
                context.append(f"- Channel {ch}: Dominant {ch_dom} ({val:.2e} V²/Hz).")

    if anomaly_scores is not None and len(anomaly_scores) > 0:
        n_anom = np.sum(anomaly_scores > np.percentile(anomaly_scores, 95))
        rate = n_anom / len(anomaly_scores) * 100
        context.append(f"Anomaly Detection: {rate:.1f}% of epochs flagged as anomalous (ensemble consensus).")

    if survey_responses:
        context.append("Participant Survey Responses:")
        for k, v in survey_responses.items():
            context.append(f"- {k}: {v}")

    # Trigger context for LLM
    if trigger_info is not None:
        t_s = trigger_info["trigger_time_s"]
        grp = trigger_info["group"]
        context.append(
            f"\nVR Experiment Trigger: {grp} stimulus administered at {t_s:.1f}s "
            f"for {trigger_info['participant']}."
        )
        if trigger_info.get("explanations"):
            for e in trigger_info["explanations"]:
                context.append(f"  Note: {e}")

    if trigger_stats:
        pre_post_df = trigger_stats.get("pre_post_df")
        if pre_post_df is not None and not pre_post_df.empty:
            context.append("\nTrigger–Anomaly Pre/Post Comparison:")
            for _, r in pre_post_df.iterrows():
                pv = r['p-value']
                pv_str = f"{pv:.4f}" if isinstance(pv, (int, float)) else str(pv)
                context.append(
                    f"- {r['Detector']}: pre={r['Pre Mean']:.5f}, "
                    f"post={r['Post Mean']:.5f}, change={r['Change %']:.1f}%, "
                    f"p={pv_str}, Cohen_d={r['Cohen d']:.3f}"
                )

        coin_df = trigger_stats.get("coincidence_df")
        if coin_df is not None and not coin_df.empty:
            context.append("\nTrigger–Anomaly Temporal Coincidence (permutation test):")
            for _, r in coin_df.iterrows():
                context.append(
                    f"- {r['Detector']} ±{r['Window (±s)']:.0f}s: "
                    f"observed={r['Observed']}, expected={r['Expected']:.2f}, "
                    f"fold={r['Fold Enrichment']:.2f}, p={r['p-value (perm)']:.4f}"
                )

        band_shift = trigger_stats.get("band_shift")
        if band_shift is not None:
            context.append("\nSpectral Band Power Shift at Trigger:")
            for b, pre, post, p in zip(
                band_shift.get("bands", []),
                band_shift.get("pre_powers", []),
                band_shift.get("post_powers", []),
                band_shift.get("p_values", []),
            ):
                change = ((post - pre) / (pre + 1e-12)) * 100
                context.append(
                    f"- {b}: pre={pre:.6f}, post={post:.6f}, "
                    f"change={change:.1f}%, p={p:.4f}"
                )

    trigger_clause = ""
    if trigger_info is not None:
        trigger_clause = (
            " This is a VR evacuation experiment with a trigger event "
            f"({trigger_info['group']} stimulus). Analyze the trigger–anomaly "
            "relationship: are the detected anomalies related to the trigger? "
            "Discuss pre/post spectral shifts in the context of neuroscience "
            "(Alpha suppression = arousal, Beta increase = cognitive load, "
            "Theta changes = stress). "
        )

    video_clause = ""
    if video_events:
        type_counts: dict[str, int] = {}
        for ev in video_events:
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
        context.append(
            f"\nVR Screen Recording Behavioural Events ({len(video_events)} total): "
            + ", ".join(f"{c} {t}" for t, c in sorted(type_counts.items()))
        )
        major_ev = [ev for ev in video_events if ev.severity == "major"]
        if major_ev:
            context.append(f"Major events ({len(major_ev)}):")
            for ev in major_ev[:10]:
                context.append(
                    f"  - {ev.event_type} @ t={ev.timestamp_s:.1f}s: "
                    f"{ev.description[:70]}"
                )
        video_clause = (
            " The session also includes VR screen-recording analysis: "
            f"{len(video_events)} behavioural events were detected "
            f"({', '.join(f'{c} {t}' for t, c in sorted(type_counts.items()))}). "
            "Discuss whether the detected EEG anomalies or spectral changes "
            "temporally coincide with the major VR events (head movements, "
            "sudden actions, environmental stimuli). "
        )

    prompt = (
        "You are an expert neuroscientist and data analyst. "
        "Analyze the following EEG session summary data and provide a concise, "
        "professional report. Highlight the spectral composition, potential artifacts "
        "(suggested by anomalies), and any psychological context from the survey. "
        + trigger_clause
        + video_clause
        + "Do not hallucinate data not present here.\n\n"
        "DATA:\n" + "\n".join(context)
    )

    # 2. Call LLM
    try:
        if provider == "Ollama":
            try:
                import ollama
            except ImportError:
                import requests
                # Fallback to raw API if lib missing
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code != 200:
                    raise ConnectionError(f"Ollama error: {resp.text}")
                return resp.json().get("response", "")

            # Use lib
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]

        elif provider == "OpenRouter":
            if not api_key:
                return "Error: OpenRouter API Key required."
            try:
                from openai import OpenAI
            except ImportError:
                return "Error: `openai` library required. Install with `pip install openai`."

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content

    except Exception as e:
        return f"Error generating summary with {provider}: {str(e)}"

    return "Error: Unknown provider."



# ---------------------------------------------------------------------------
# XAI (optional)
# ---------------------------------------------------------------------------

def compute_shap_values(
    features: pd.DataFrame,
    model,
) -> np.ndarray | None:
    """Compute SHAP values for the given model.

    Returns None if shap is not available.
    """
    try:
        import shap
    except ImportError:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        return shap_values.values
    except Exception:
        # Fallback to KernelExplainer for unsupported models
        try:
            explainer = shap.KernelExplainer(model.decision_function, X[:100])
            return explainer.shap_values(X)
        except Exception:
            return None

# NeuroEvac

A research-grade Streamlit dashboard for multi-modal EEG analysis, purpose-built for studying neurophysiological responses in VR evacuation scenarios. NeuroEvac covers the full analysis pipeline — from raw signal ingestion and preprocessing to spectral analysis, topographic mapping, ML/DL anomaly detection, VR screen recording analysis via Vision-Language Models (Gemini), UMAP epoch clustering, and AI-generated neurophysiological reports via Ollama or OpenRouter.

## Research Context

This dashboard was developed in the context of a study on **human neurophysiological responses during VR evacuation scenarios**. EEG recordings were collected from participants wearing EEG headsets while navigating virtual emergency evacuation environments. The study examines cognitive load, stress arousal, and attentional dynamics under time pressure. The analysis pipeline supports the full workflow from raw signal ingestion to anomaly-flagged epoch review, multi-modal video–EEG behavioural correlation, and AI-generated neurophysiological summaries — enabling reproducible analysis of high-motion, high-noise EEG data collected in ecologically valid VR settings.

## Overview

NeuroEvac ingests raw EEG recordings (OpenBCI CSV/TXT, BrainFlow CSV) alongside optional participant survey files (XLSX) and VR screen recordings (MP4, AVI, MOV, WebM). All signal processing runs on-device; no data leaves the local environment by default. Optional AI features include LLM-based reporting (Ollama / OpenRouter) and Vision-Language Model video analysis (Google Gemini) for extracting behavioural events from VR recordings.

## Features

| Page | Description |
|---|---|
| **Raw EEG Viewer** | Multi-channel time-series visualization with channel selector and time-window control |
| **Preprocessing** | Bandpass/notch filtering, re-referencing, bad-channel interpolation, ICA-based artifact removal |
| **Spectral Analysis** | Power Spectral Density (Welch), topographic scalp maps, frequency band power heatmaps |
| **Topography** | Spatial cortical activity maps across standard EEG frequency bands |
| **Anomaly Detection** | EEG-specific detectors (amplitude Z-score, spectral ratio, kurtosis/entropy) and general ML detectors (Isolation Forest, One-Class SVM, LOF, Autoencoder). Interactive heatmap, epoch gallery, feature importance, VR trigger analysis, and **video–EEG multi-modal correlation** |
| **AI Insights** | UMAP + HDBSCAN unsupervised epoch clustering; AI-generated analysis reports via Template, Ollama, or OpenRouter |
| **Survey Data** | Participant survey ingestion, psychometric scoring, radar and bar chart visualizations |

### VR Screen Recording Analysis (New)

The Anomaly Detection page now integrates **Vision-Language Model (VLM) analysis** of VR screen recordings using Google Gemini. This enables:

- **Automated behavioural event extraction** — head movements, gaze shifts, sudden actions, object interactions, and environmental changes are detected with precise timestamps
- **Multi-modal timeline** — unified visualization of EEG anomaly scores alongside behavioural events
- **Temporal coincidence testing** — permutation-based statistical test (n=1000) determining if EEG anomalies cluster around behavioural events more than expected by chance
- **Event-locked ERP analysis** — grand-average EEG response time-locked to each category of behavioural event, with ±1 SD confidence bands
- **Pre/post event comparison** — Mann-Whitney U tests with Cohen's d effect sizes comparing anomaly scores before vs. after behavioural events
- **Cross-page annotations** — Video events are overlaid on both Anomaly Detection plots and the Raw EEG Viewer for visual inspection

## Project Structure

```
eeg_dashboard/
├── Dashboard.py              # Entry point — run this with streamlit
├── requirements.txt
├── assets/
│   ├── style.css             # Global theme styles
│   └── eeg.png               # Logo
├── pages/
│   ├── 1_Raw_EEG_Viewer.py   # Multi-channel viewer + video event overlays
│   ├── 2_Preprocessing.py
│   ├── 3_Spectral_Analysis.py
│   ├── 4_Topography.py
│   ├── 5_Anomaly_Detection.py # Anomaly detection + video–EEG analysis
│   ├── 6_AI_Insights.py
│   └── 7_Survey_Data.py
├── analysis/
│   ├── ai_insights.py        # Clustering, LLM integration (Ollama / OpenRouter)
│   ├── anomaly.py            # Anomaly detection algorithms
│   ├── report.py             # HTML report builder
│   └── video_analysis.py     # VR video analysis via Gemini VLM
├── tests/
│   └── test_video_analysis.py # Unit tests for video analysis module
└── utils/
    ├── data_loader.py        # EEG and survey file parsing
    ├── sidebar.py            # Shared sidebar CSS, logo, footer
    └── signal_processing.py  # MNE wrappers, feature extraction, band power
```

## Requirements

- Python 3.10+
- See `requirements.txt` for all Python dependencies

```
pip install -r requirements.txt
```

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/pozapas/neuroevac.git
cd eeg-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run Dashboard.py
```

Then open `http://localhost:8501` in your browser.

### Supported File Formats

| Type | Format | Notes |
|---|---|---|
| EEG | OpenBCI raw CSV / TXT | 1-indexed headers, µV data |
| EEG | BrainFlow CSV | Auto-detected column layout |
| Survey | XLSX | Participant ID in column A, Likert-scale responses |
| Video | MP4, AVI, MOV, WebM, MKV | VR screen recording for behavioural event extraction |

## AI Insights — LLM Configuration

The **AI Insights** page supports three summary modes:

1. **Template (Offline)** — Rule-based summary, no API key required.
2. **Local (Ollama)** — Sends a structured prompt to a locally running Ollama instance. Requires `ollama serve` and a downloaded model (e.g., `ollama pull llama3`).
3. **Cloud (OpenRouter)** — Sends the prompt to any model available on [openrouter.ai](https://openrouter.ai) using your personal API key.

No EEG data is transmitted externally unless you explicitly choose a cloud provider.

## VR Video Analysis — Gemini Configuration

The **Anomaly Detection** page includes a VR Screen Recording Analysis section:

1. Expand the **🎥 VR Screen Recording Analysis** panel.
2. Enter your **Google AI Studio API key** (get one at [aistudio.google.com](https://aistudio.google.com)).
3. Upload a video file or provide a local file path.
4. Select a Gemini model (`gemini-flash-latest` for full analysis, `gemini-flash-lite-latest` for faster/lighter processing).
5. Optionally customise the prompt with experiment-specific context.
6. Click **Analyse Video** — events will be extracted and overlaid on anomaly plots.
7. The new **🎥 Video-Anomaly Analysis** tab provides statistical correlation analyses.

Video data is sent to Google's Gemini API for processing. No video data is stored remotely.


## Dependencies

Core scientific stack:
- [MNE-Python](https://mne.tools) — EEG preprocessing and analysis
- [Plotly](https://plotly.com/python/) — Interactive visualizations
- [scikit-learn](https://scikit-learn.org) — Machine learning detectors
- [UMAP-learn](https://umap-learn.readthedocs.io) + [HDBSCAN](https://hdbscan.readthedocs.io) — Epoch clustering
- [SciPy](https://scipy.org) — Spectral estimation
- [google-genai](https://pypi.org/project/google-genai/) — Gemini Vision-Language Model for VR video analysis

## License

MIT License. See `LICENSE` for details.

## Author

**Amir Rafe**  
Texas State University  
[amir.rafe@txstate.edu](mailto:amir.rafe@txstate.edu)  
[pozapas.github.io](https://pozapas.github.io)

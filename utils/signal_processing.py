"""
signal_processing.py — Filtering, epoching, and feature extraction utilities.
"""

from __future__ import annotations

from typing import Optional

import mne
import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_bandpass(
    raw: mne.io.RawArray,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
) -> mne.io.RawArray:
    """Apply FIR bandpass filter (in-place copy)."""
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq, h_freq, method="fir", verbose=False)
    return raw_filtered


def apply_notch(
    raw: mne.io.RawArray,
    freqs: float | list[float] = 60.0,
) -> mne.io.RawArray:
    """Apply notch filter at specified frequency/frequencies (in-place copy).
    Uses IIR method for sharper attenuation and includes harmonics.
    """
    raw_filtered = raw.copy()
    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    
    # Generate harmonics up to Nyquist to ensure thorough filtering
    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2.0
    all_freqs = []
    for f in freqs:
        h = f
        while h < nyquist:
            all_freqs.append(h)
            h += f
    
    if not all_freqs:
        return raw_filtered

    # IIR filters are generally more effective for notch filtering power line noise
    raw_filtered.notch_filter(
        all_freqs, 
        method="iir", 
        verbose=False
    )
    return raw_filtered


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------

def make_fixed_epochs(
    raw: mne.io.RawArray,
    duration: float = 2.0,
    overlap: float = 0.0,
) -> mne.Epochs:
    """Create fixed-length epochs from continuous data."""
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


# ---------------------------------------------------------------------------
# PSD / Band Power
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "Delta (1-4 Hz)": (1, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (30-100 Hz)": (30, 100),
}


def compute_psd(
    raw: mne.io.RawArray,
    fmin: float = 0.5,
    fmax: float = 100.0,
    n_fft: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD using Welch's method. Returns (psds, freqs)."""
    spectrum = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    psds = spectrum.get_data()   # (n_channels, n_freqs)
    freqs = spectrum.freqs
    return psds, freqs


def compute_band_powers(
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute absolute band power for each channel.

    Returns DataFrame with shape (n_channels, n_bands).
    """
    if bands is None:
        bands = FREQ_BANDS

    results = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        results[band_name] = np.mean(psds[:, idx], axis=1)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Feature Extraction (per-epoch)
# ---------------------------------------------------------------------------

def hjorth_parameters(epoch: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, complexity for a 1-D signal."""
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)

    activity = np.var(epoch)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0.0
    complexity = (
        (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility)
        if mobility > 0 and np.var(diff1) > 0
        else 0.0
    )
    return activity, mobility, complexity


def line_length(epoch: np.ndarray) -> float:
    """Sum of absolute successive differences."""
    return float(np.sum(np.abs(np.diff(epoch))))


def extract_epoch_features(
    epochs: mne.Epochs,
    bands: Optional[dict] = None,
) -> pd.DataFrame:
    """Extract feature vectors for each epoch.

    Features per channel: 5 band powers + 3 Hjorth + line length = 9.
    Total features = n_channels × 9.
    """
    if bands is None:
        bands = FREQ_BANDS

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.info["ch_names"]

    all_features = []

    for ep_idx in range(data.shape[0]):
        feat_row = {}
        for ch_idx, ch_name in enumerate(ch_names):
            sig = data[ep_idx, ch_idx, :]

            # Band powers via Welch
            freqs, pxx = sp_signal.welch(sig, fs=sfreq, nperseg=min(len(sig), 256))
            for band_name, (fmin, fmax) in bands.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                feat_row[f"{ch_name}_{band_name}"] = np.mean(pxx[idx]) if np.any(idx) else 0.0

            # Hjorth
            act, mob, comp = hjorth_parameters(sig)
            feat_row[f"{ch_name}_hjorth_activity"] = act
            feat_row[f"{ch_name}_hjorth_mobility"] = mob
            feat_row[f"{ch_name}_hjorth_complexity"] = comp

            # Line length
            feat_row[f"{ch_name}_line_length"] = line_length(sig)

        all_features.append(feat_row)

    return pd.DataFrame(all_features)


# ---------------------------------------------------------------------------
# EEG-specific anomaly helpers
# ---------------------------------------------------------------------------

def get_safe_bands(sfreq: float, bands: dict | None = None) -> dict:
    """Return frequency bands capped at Nyquist to prevent errors."""
    if bands is None:
        bands = FREQ_BANDS
    nyquist = sfreq / 2.0 - 0.5
    return {name: (fmin, min(fmax, nyquist)) for name, (fmin, fmax) in bands.items()}


def compute_epoch_kurtosis(epochs: mne.Epochs) -> np.ndarray:
    """Compute kurtosis per channel per epoch.

    Returns array of shape (n_epochs, n_channels).
    High kurtosis indicates peaked distributions (muscle artifacts).
    """
    from scipy.stats import kurtosis
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    return kurtosis(data, axis=2, fisher=True)


def compute_sample_entropy(
    epochs: mne.Epochs,
    m: int = 2,
    r_factor: float = 0.2,
) -> np.ndarray:
    """Compute approximate sample entropy per channel per epoch.

    Returns array of shape (n_epochs, n_channels).
    Very low entropy → flat-line; very high → noise / non-physiological.
    Uses a fast vectorised approximation.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs_val, n_ch, n_times = data.shape
    result = np.zeros((n_epochs_val, n_ch))

    for ep in range(n_epochs_val):
        for ch in range(n_ch):
            x = data[ep, ch, :]
            std = np.std(x)
            if std < 1e-12:
                result[ep, ch] = 0.0
                continue
            r = r_factor * std
            # Count template matches for length m and m+1
            n = len(x)
            count_m, count_m1 = 0, 0
            # subsample for speed if signal is long
            step = max(1, n // 200)
            indices = range(0, n - m, step)
            for i in indices:
                template_m = x[i : i + m]
                template_m1 = x[i : i + m + 1] if i + m + 1 <= n else None
                for j in range(i + 1, min(i + 50 * step, n - m), step):
                    if np.max(np.abs(x[j : j + m] - template_m)) < r:
                        count_m += 1
                        if template_m1 is not None and j + m + 1 <= n:
                            if np.max(np.abs(x[j : j + m + 1] - template_m1)) < r:
                                count_m1 += 1
            if count_m > 0 and count_m1 > 0:
                result[ep, ch] = -np.log(count_m1 / count_m)
            else:
                result[ep, ch] = 0.0

    return result


def compute_spectral_ratios(epochs: mne.Epochs) -> pd.DataFrame:
    """Compute theta/beta and alpha/beta ratios per epoch (averaged over channels).

    Returns DataFrame with columns ['theta_beta', 'alpha_beta'] and n_epochs rows.
    Abnormal ratios indicate artifacts (muscle → low ratios, drowsiness → high theta).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_ep = data.shape[0]

    safe_bands = get_safe_bands(sfreq)
    theta_range = safe_bands.get("Theta (4-8 Hz)", (4, 8))
    alpha_range = safe_bands.get("Alpha (8-13 Hz)", (8, 13))
    beta_range = safe_bands.get("Beta (13-30 Hz)", (13, 30))

    theta_beta = np.zeros(n_ep)
    alpha_beta = np.zeros(n_ep)

    for i in range(n_ep):
        # Average PSD across channels for this epoch
        sig = data[i]  # (n_channels, n_times)
        all_theta, all_alpha, all_beta = [], [], []
        for ch in range(sig.shape[0]):
            freqs, pxx = sp_signal.welch(sig[ch], fs=sfreq, nperseg=min(sig.shape[1], 256))
            all_theta.append(np.mean(pxx[(freqs >= theta_range[0]) & (freqs <= theta_range[1])]))
            all_alpha.append(np.mean(pxx[(freqs >= alpha_range[0]) & (freqs <= alpha_range[1])]))
            all_beta.append(np.mean(pxx[(freqs >= beta_range[0]) & (freqs <= beta_range[1])]))

        mean_theta = np.mean(all_theta)
        mean_alpha = np.mean(all_alpha)
        mean_beta = np.mean(all_beta)

        theta_beta[i] = mean_theta / mean_beta if mean_beta > 1e-20 else 0.0
        alpha_beta[i] = mean_alpha / mean_beta if mean_beta > 1e-20 else 0.0

    return pd.DataFrame({"theta_beta": theta_beta, "alpha_beta": alpha_beta})


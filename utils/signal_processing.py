"""
signal_processing.py — Filtering, epoching, and feature extraction utilities.
"""

from __future__ import annotations

from typing import Optional

import mne
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import pywt  # Import pywt at the top level for WPT

# ---------------------------------------------------------------------------
# Montage & Channel Mapping
# ---------------------------------------------------------------------------

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

def apply_standard_montage(raw: mne.io.BaseRaw, mapping: dict[str, str] = None) -> mne.io.BaseRaw:
    """Rename channels and apply standard 10-20 montage.
    
    Returns a copy of the raw object with channels renamed and montage set.
    """
    raw_new = raw.copy()
    if mapping:
        rename_dict = {old: new for old, new in mapping.items() if old in raw_new.ch_names}
        raw_new.rename_channels(rename_dict)
    
    montage = mne.channels.make_standard_montage("standard_1020")
    # Keep only channels that exist in the montage for ICA/Topography
    valid_chs = [ch for ch in raw_new.ch_names if ch in montage.ch_names]
    if valid_chs:
        raw_new.pick(valid_chs)
        raw_new.set_montage(montage, on_missing="ignore")
    return raw_new

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_custom_filter(
    raw: mne.io.RawArray,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    method: str = "fir",
    order: int = 4,
    iir_type: str = "butter"
) -> mne.io.RawArray:
    """Apply a custom filter (FIR or IIR) to the data."""
    raw_filtered = raw.copy()
    
    if method == "fir":
        # For FIR, 'order' can be related to filter_length, 
        # but MNE's default 'auto' is usually best. 
        # We'll use order to influence filter_length if provided.
        filter_length = f"{order}s" if order > 0 else "auto"
        raw_filtered.filter(
            l_freq, h_freq, 
            method="fir", 
            phase="zero", 
            fir_window="hamming", 
            fir_design="firwin",
            verbose=False
        )
    else:
        # IIR filters: butter, bessel, cheby1, etc.
        iir_params = {
            "order": order,
            "ftype": iir_type,
            "output": "sos"
        }
        # Chebyshev requires extra params, we'll use defaults for simplicity or add them if needed
        if iir_type == "cheby1":
            iir_params["rp"] = 0.5
        
        raw_filtered.filter(
            l_freq, h_freq, 
            method="iir", 
            iir_params=iir_params, 
            verbose=False
        )
        
    return raw_filtered


def apply_notch(
    raw: mne.io.RawArray,
    freqs: float | list[float] = 60.0
) -> mne.io.RawArray:
    """Apply a sharp IIR notch filter at specified frequency/frequencies."""
    raw_filtered = raw.copy()
    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    
    if not freqs:
        return raw_filtered

    raw_filtered.notch_filter(
        freqs, 
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


def extract_3d_features(
    epochs: mne.Epochs,
    bands: Optional[dict] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract 3D feature set: (epochs, channels, features).
    
    Features: Delta, Theta, Alpha, Beta, Gamma, RMS.
    """
    if bands is None:
        bands = FREQ_BANDS

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_channels, n_times = data.shape
    
    feature_names = list(bands.keys()) + ["RMS"]
    n_features = len(feature_names)
    
    features_3d = np.zeros((n_epochs, n_channels, n_features))
    
    for ep_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Scale signal to microvolts for standard EEG feature units
            sig = data[ep_idx, ch_idx, :] * 1e6 
            
            # Band powers via Welch (converted to dB: 10*log10(P))
            freqs, pxx = sp_signal.welch(sig, fs=sfreq, nperseg=min(len(sig), 256))
            for f_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                if np.any(idx):
                    p_mean = np.mean(pxx[idx])
                    # Result in dB relative to 1 µV²/Hz
                    features_3d[ep_idx, ch_idx, f_idx] = 10 * np.log10(p_mean + 1e-20)
                else:
                    features_3d[ep_idx, ch_idx, f_idx] = -200.0
            
            # RMS in microvolts
            features_3d[ep_idx, ch_idx, -1] = np.sqrt(np.mean(sig**2))
            
    return features_3d, feature_names


def get_epoch_tags(
    epochs: mne.Epochs,
    raw: mne.io.RawArray
) -> list[str]:
    """Identify the primary tag for each epoch based on Raw annotations.
    
    If an epoch overlaps with an annotation, it gets that label. 
    If multiple, the one with the most overlap is chosen.
    """
    annotations = raw.annotations
    if not annotations:
        return ["none"] * len(epochs)

    tags = []
    # epoch onsets in seconds
    onsets = epochs.events[:, 0] / raw.info['sfreq']
    duration = epochs.tmax - epochs.tmin

    for start_t in onsets:
        end_t = start_t + duration
        
        # Find annotations that overlap [start_t, end_t]
        epoch_label = "none"
        max_overlap = 0
        
        for ann in annotations:
            ann_start = ann['onset']
            ann_end = ann_start + ann['duration']
            
            # Intersection
            overlap_start = max(start_t, ann_start)
            overlap_end = min(end_t, ann_end)
            
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > max_overlap:
                    max_overlap = overlap
                    epoch_label = ann['description']
        
        tags.append(epoch_label)
    
    return tags


def apply_asr(
    raw: mne.io.RawArray,
    cutoff: float = 20.0,
    window_len: float = 0.5
) -> mne.io.RawArray:
    """Artifact Subspace Reconstruction (ASR) inspired denoising.
    Uses sliding window PCA to identify and project out high-variance components.
    """
    raw_copy = raw.copy()
    data = raw_copy.get_data()
    sfreq = raw.info['sfreq']
    n_ch, n_samples = data.shape
    
    # 1. Standardize data (Z-score per channel)
    mu = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_norm = (data - mu) / (std + 1e-12)
    
    # 2. Window-based processing
    win_samples = int(window_len * sfreq)
    # Ensure window is large enough for PCA
    win_samples = max(win_samples, n_ch * 2)
    
    # Step size (50% overlap)
    step = win_samples // 2
    
    # Final data buffer
    data_out = np.zeros_like(data_norm)
    weights = np.zeros(n_samples)
    
    # Window function (Hanning) to smooth overlaps
    hann = np.hanning(win_samples)
    
    # 3. Sliding window PCA
    for start in range(0, n_samples - win_samples, step):
        end = start + win_samples
        window = data_norm[:, start:end]
        
        # PCA on this window
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_ch)
        # We use components whose variance is significantly higher than 1.0 
        # (since data is normalized, expected variance of PCA components is 1.0 
        # if the data is white noise, but real EEG has structure).
        # We look for components exceeding 'cutoff' times the median variance.
        
        pca.fit(window.T)
        comp_vars = pca.explained_variance_
        
        # Identify "bad" components in this window
        # In standardized data, component variances sum to n_ch.
        # A component with variance >> 1 is likely an artifact.
        bad_idx = comp_vars > cutoff
        
        if np.any(bad_idx):
            components = pca.transform(window.T).T
            components[bad_idx] = 0 # Suppress bad components
            window_clean = pca.inverse_transform(components.T).T
        else:
            window_clean = window
            
        data_out[:, start:end] += window_clean * hann
        weights[start:end] += hann
        
    # Handle regions not covered or at ends
    weights[weights == 0] = 1.0
    data_out /= weights
    
    # 4. Rescale and return
    data_final = data_out * std + mu
    raw_copy._data = data_final
    return raw_copy

def apply_wpt_denoising(
    raw: mne.io.RawArray,
    wavelet: str = "db4",
    level: int = 4,
    threshold: float = 0.02,
) -> mne.io.RawArray:
    """Denoise EEG signal using Wavelet Packet Transform (WPT)."""
    raw_copy = raw.copy()
    data = raw_copy.get_data()
    
    # We apply this channel by channel
    for i in range(data.shape[0]):
        sig = data[i]
        # Decomposition
        wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode='symmetric', maxlevel=level)
        
        # Hard thresholding on nodes
        # In a real scenario, we'd use more sophisticated thresholding (Universal, SURE, etc.)
        # For this prototype, we'll zero out nodes where the energy is below the threshold
        # of the total signal energy to keep it simple but functional.
        total_energy = np.sum(sig**2)
        for node in wp.get_level(level, 'freq'):
            node_energy = np.sum(node.data**2)
            if node_energy < threshold * total_energy:
                node.data.fill(0)
        
        data[i] = wp.reconstruct(update=True)[:len(sig)]
        
    raw_copy._data = data
    return raw_copy


def apply_wat_denoising(
    raw: mne.io.RawArray,
    scale_level: int = 4,
    threshold: float = 0.02,
) -> mne.io.RawArray:
    """Simplified 1D Wave Atom-like denoising using frequency partitioning."""
    raw_copy = raw.copy()
    data = raw_copy.get_data()
    n_samples = data.shape[1]
    
    # Wave atoms often require power-of-2 length or specific padding
    # For this implementation, we'll work on the FFT of the signal
    for i in range(data.shape[0]):
        sig = data[i]
        f_hat = np.fft.fft(sig)
        n = len(f_hat)
        
        # Partition frequency axis into tiles (atoms)
        num_tiles = 2**scale_level
        tile_size = n // num_tiles
        
        if tile_size == 0:
            continue
            
        for j in range(num_tiles):
            start = j * tile_size
            end = (j + 1) * tile_size if j < num_tiles - 1 else n
            
            tile_coeffs = f_hat[start:end]
            # Thresholding in frequency domain per "atom"
            if np.mean(np.abs(tile_coeffs)) < threshold * np.mean(np.abs(f_hat)):
                f_hat[start:end] = 0
                
        data[i] = np.real(np.fft.ifft(f_hat))
        
    raw_copy._data = data
    return raw_copy


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

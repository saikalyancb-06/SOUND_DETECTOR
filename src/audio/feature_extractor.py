import numpy as np
import scipy.signal
import librosa
from src.utils.logger import get_logger

logger = get_logger("FeatureExtractor")

def estimate_formants(y, sr):
    """
    Formant tracking (F1-F4) using LPC root-finding.
    """
    try:
        # Pre-emphasis filter
        y_filt = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # LPC order: 2 + sr/1000
        n_coeff = 2 + int(sr / 1000)
        a = librosa.lpc(y_filt, order=n_coeff)
        
        # LPC polynomial roots
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]
        
        # Frequencies
        angs = np.arctan2(np.imag(roots), np.real(roots))
        frqs = angs * (sr / (2 * np.pi))
        
        # Keep valid formant range
        valid_frqs = sorted([f for f in frqs if 90 < f < (sr / 2) - 90])
        
        formants = [0.0, 0.0, 0.0, 0.0]
        for i in range(min(4, len(valid_frqs))):
            formants[i] = float(valid_frqs[i])
        return formants
    except Exception as ex:
        logger.warning(f"Error estimating formants: {ex}")
        return [0.0, 0.0, 0.0, 0.0]

def estimate_jitter_shimmer_hnr(y, sr, f0):
    """
    Estimates cycle-to-cycle frequency variations (Jitter),
    amplitude variations (Shimmer), and Harmonic-to-Noise Ratio (HNR).
    """
    f0_valid = f0[f0 > 0.0]
    if len(f0_valid) < 2:
        return 0.0, 0.0, 0.0
        
    try:
        # Jitter (Relative Average Perturbation estimate)
        periods = 1.0 / f0_valid
        diff_periods = np.abs(np.diff(periods))
        jitter = float(np.mean(diff_periods) / np.mean(periods)) if np.mean(periods) > 0 else 0.0
        
        # Shimmer (local variation in peak amplitude)
        frame_lengths = (sr / f0_valid).astype(int)
        amplitudes = []
        idx = 0
        for fl in frame_lengths:
            if idx + fl >= len(y):
                break
            amplitudes.append(np.max(np.abs(y[idx:idx+fl])) + 1e-6)
            idx += fl
            
        if len(amplitudes) >= 2:
            diff_amps = np.abs(np.diff(amplitudes))
            shimmer = float(np.mean(diff_amps) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 0.0
        else:
            shimmer = 0.0
            
        # Harmonic-to-Noise Ratio (HNR) via autocorrelation of voiced regions
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        mean_period = int(sr / np.mean(f0_valid))
        search_min = int(0.8 * mean_period)
        search_max = int(1.2 * mean_period)
        
        peak_val = 1e-6
        if search_max < len(autocorr) and search_min < search_max:
            peak_val = np.max(autocorr[search_min:search_max])
            
        noise_floor = np.abs(autocorr[0] - peak_val)
        # Avoid dividing by zero and ensure positive argument for log10
        val_for_log = max(1e-6, peak_val / (noise_floor + 1e-6))
        hnr = float(10 * np.log10(val_for_log))
        return jitter, shimmer, hnr
    except Exception as ex:
        logger.warning(f"Error estimating jitter/shimmer/hnr: {ex}")
        return 0.0, 0.0, 0.0

def extract_features(y, sr):
    """
    Computes a comprehensive set of acoustic features from input waveform segment.
    """
    features = {}
    
    # 1. MFCC (1-40) + Deltas + Delta-Deltas
    n_mfcc = 40
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Mean of each MFCC coefficient
    for i in range(1, n_mfcc + 1):
        features[f"mfcc_{i}"] = float(np.mean(mfccs[i-1]))
        
    # Delta and Delta-Delta MFCCs
    try:
        delta_mfcc = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        for i in range(1, 14): # Keep first 13 delta / delta-deltas to avoid feature explosion
            features[f"delta_mfcc_{i}"] = float(np.mean(delta_mfcc[i-1]))
            features[f"delta2_mfcc_{i}"] = float(np.mean(delta2_mfcc[i-1]))
    except Exception:
        # Fallback to zero if signal too short for delta width
        for i in range(1, 14):
            features[f"delta_mfcc_{i}"] = 0.0
            features[f"delta2_mfcc_{i}"] = 0.0
            
    # 2. Pitch (F0)
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=512)
        f0_clean = f0[f0 > 0.0]
        mean_f0 = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        std_f0 = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
        pitch_range = float(np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0.0
    except Exception:
        f0 = np.zeros_like(y)
        mean_f0 = 0.0
        std_f0 = 0.0
        pitch_range = 0.0
        
    features["mean_f0"] = mean_f0
    features["std_f0"] = std_f0
    features["pitch_range"] = pitch_range
    
    # 3. Formants (F1-F4)
    formants = estimate_formants(y, sr)
    for idx, f_val in enumerate(formants, 1):
        features[f"formant_F{idx}"] = f_val
        
    # 4. Chroma Features (12 dimensions, mean over time)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(1, 13):
        features[f"chroma_{i}"] = float(np.mean(chroma[i-1]))
        
    # 5. Spectral Contrast (7 bands, mean over time)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(1, 8):
        features[f"spectral_contrast_{i}"] = float(np.mean(contrast[i-1]))
        
    # 6. Spectral Roll-off, Spectral Centroid, ZCR, RMS Energy
    features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features["rms_energy"] = float(np.mean(librosa.feature.rms(y=y)))
    
    # 7. Jitter, Shimmer, HNR
    jitter, shimmer, hnr = estimate_jitter_shimmer_hnr(y, sr, f0)
    features["jitter"] = jitter
    features["shimmer"] = shimmer
    features["hnr"] = hnr
    
    # 8. Speaking Rate, Pause Duration, Silence Ratio (within segment)
    intervals = librosa.effects.split(y, top_db=25)
    active_samples = sum([end - start for start, end in intervals])
    total_samples = len(y)
    
    pause_samples = total_samples - active_samples
    features["pause_duration"] = float(pause_samples / float(sr))
    features["silence_ratio"] = float(pause_samples / float(total_samples)) if total_samples > 0 else 0.0
    
    # Estimate speaking rate: count energy-based syllable nuclei/onsets
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration = float(total_samples) / sr
    features["speaking_rate"] = float(len(onsets) / duration) if duration > 0 else 0.0
    
    # 9. Speaker Embeddings (ECAPA-TDNN)
    try:
        from src.diarization.speaker_embeddings import extract_speaker_embedding
        emb = extract_speaker_embedding(y, sr)
        for i, val in enumerate(emb):
            features[f"embedding_{i}"] = float(val)
    except Exception as ex:
        logger.warning(f"Error extracting speaker embedding: {ex}")
        for i in range(192):
            features[f"embedding_{i}"] = 0.0

    # Sanitize dictionary values to eliminate NaNs or Infs
    sanitized_features = {}
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            sanitized_features[k] = 0.0
        else:
            sanitized_features[k] = v
            
    return sanitized_features

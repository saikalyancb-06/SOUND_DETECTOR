# ==========================================
# FILE: src/audio_processing.py
# ==========================================
import librosa
import numpy as np

def extract_acoustic_parameters(audio_path):
    """
    Computes exactly 21 explicit linguistic and acoustic features across
    tonal, spectral, and dynamic properties.
    """
    # Load with native sampling configurations; limit duration to 10s for CPU efficiency
    y, sr = librosa.load(audio_path, sr=None, duration=10.0)
    
    # 1-2. Fundamental Frequency Tracking (F0/Pitch Analytics)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_cleaned = f0[~np.isnan(f0)]
    mean_f0 = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0.0
    std_f0 = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0.0
    pitch_range = (np.max(f0_cleaned) - np.min(f0_cleaned)) if len(f0_cleaned) > 0 else 0.0
    voiced_ratio = float(np.mean(~np.isnan(f0))) if len(f0) > 0 else 0.0

    # 3-5. Spectral Domain Footprints
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # 6. Signal Energy Dynamics (RMS Proxy)
    rms = librosa.feature.rms(y=y)[0]
    
    # 7-19. Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)

    # Compile explicit baseline payload mapping dictionary
    feature_vector = {
        "mean_f0": float(mean_f0),
        "std_f0": float(std_f0),
        "pitch_range": float(pitch_range),
        "voiced_ratio": float(voiced_ratio),
        "mean_spectral_centroid": float(np.mean(spectral_centroids)),
        "mean_spectral_rolloff": float(np.mean(spectral_rolloff)),
        "mean_zcr": float(np.mean(zero_crossing_rate)),
        "rms_energy": float(np.mean(rms)),
    }
    
    # Map raw structural MFCC elements cleanly into vector
    for i, mfcc_val in enumerate(mean_mfccs, 1):
        feature_vector[f"mfcc_{i}"] = float(mfcc_val)
        
    return feature_vector  # Matrix containing precisely 21 scalar features
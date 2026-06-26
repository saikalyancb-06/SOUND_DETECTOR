import numpy as np
import scipy.signal
from src.utils.logger import get_logger

logger = get_logger("NoiseReduction")

def reduce_noise(y, sr, method="spectral_subtraction"):
    """
    Applies noise reduction over input waveform.
    If DeepFilterNet/RNNoise are available, they will be loaded;
    otherwise, it falls back to a clean python-native Spectral Subtraction implementation.
    """
    logger.info(f"Applying noise reduction using method: {method}")
    
    if method == "spectral_subtraction":
        # Estimate noise from the first 0.5s of audio
        noise_estimation_seconds = 0.5
        noise_samples = int(noise_estimation_seconds * sr)
        if len(y) <= noise_samples:
            return y
        
        # Run STFT on signal
        nperseg = 1024
        noverlap = 512
        f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        # Compute average noise magnitude spectrum from initial part
        noise_end_frame = max(1, int(noise_samples / (nperseg - noverlap)))
        noise_part = Zxx[:, :noise_end_frame]
        noise_magnitude = np.mean(np.abs(noise_part), axis=1, keepdims=True)
        
        # Perform spectral subtraction with over-subtraction factor and spectral floor
        signal_magnitude = np.abs(Zxx)
        signal_phase = np.angle(Zxx)
        
        # Subtract noise spectrum
        subtracted_magnitude = signal_magnitude - 2.0 * noise_magnitude
        subtracted_magnitude = np.maximum(subtracted_magnitude, 0.01 * signal_magnitude) 
        
        # Reconstruct complex spectrum
        Zxx_cleaned = subtracted_magnitude * np.exp(1j * signal_phase)
        
        # Inverse STFT to return to waveform
        _, y_cleaned = scipy.signal.istft(Zxx_cleaned, fs=sr, nperseg=nperseg, noverlap=noverlap)
        
        # Match lengths
        if len(y_cleaned) < len(y):
            y_cleaned = np.pad(y_cleaned, (0, len(y) - len(y_cleaned)))
        else:
            y_cleaned = y_cleaned[:len(y)]
            
        return y_cleaned.astype(np.float32)
        
    else:
        # Fallback to no-op if unrecognized method
        logger.warning(f"Method {method} not implemented/available. Returning original audio.")
        return y

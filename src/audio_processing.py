# ==========================================
# FILE: src/audio_processing.py
# ==========================================
import librosa
try:
    from src.audio.feature_extractor import extract_features
except ModuleNotFoundError:
    from audio.feature_extractor import extract_features

def extract_acoustic_parameters(audio_path):
    """
    Legacy wrapper function to preserve compatibility.
    Loads audio and extracts acoustic parameters.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    # Return features dict
    return extract_features(y, sr)
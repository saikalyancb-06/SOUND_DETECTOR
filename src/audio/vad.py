import numpy as np
import torch
import librosa
from src.utils.logger import get_logger

logger = get_logger("VAD")

_silero_model = None
_silero_utils = None

def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        try:
            logger.info("Attempting to load Silero VAD from torch hub...")
            # Set trust_repo=True to allow loading the hub model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            _silero_model = model
            _silero_utils = utils
            logger.info("Successfully loaded Silero VAD model.")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD model from hub ({e}). Using librosa VAD fallback.")
            _silero_model = False
    return _silero_model, _silero_utils

def get_speech_intervals(y, sr, threshold=0.5):
    """
    Finds active speech intervals (start and end sample indices).
    Falls back to librosa.effects.split if Silero VAD is unavailable.
    """
    model, utils = load_silero_vad()
    
    if model:
        try:
            # Silero expects 16kHz audio
            if sr != 16000:
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                y_16k = y
                
            # Convert to torch tensor
            wav = torch.from_numpy(y_16k)
            get_speech_timestamps, _, _, _, _ = utils
            
            # Get timestamps dictionary
            logger.info("Running Silero VAD forward pass...")
            speech_timestamps = get_speech_timestamps(wav, model, threshold=threshold, sampling_rate=16000)
            
            intervals = []
            for ts in speech_timestamps:
                # Convert 16k sample index back to original sampling rate sample index
                start_sample = int(ts['start'] * (sr / 16000.0))
                end_sample = int(ts['end'] * (sr / 16000.0))
                intervals.append([start_sample, end_sample])
                
            if len(intervals) > 0:
                logger.info(f"Silero VAD identified {len(intervals)} speech segments.")
                return np.array(intervals)
        except Exception as ex:
            logger.error(f"Error executing Silero VAD: {ex}. Falling back to librosa VAD.")
            
    # Librosa energy-based VAD fallback
    logger.info("Executing energy-based VAD split fallback...")
    intervals = librosa.effects.split(y, top_db=25)
    logger.info(f"Fallback VAD identified {len(intervals)} speech segments.")
    return intervals

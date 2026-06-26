import numpy as np
import torch
import librosa
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("SpeakerEmbeddings")

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            logger.info("Loading SpeechBrain ECAPA-TDNN model for speaker embeddings...")
            
            # Monkeypatch torch.amp for PyTorch 2.1+ compatibility with SpeechBrain
            import torch.amp
            import torch.cuda.amp
            
            def custom_fwd_patched(*args, **kwargs):
                if len(args) == 1 and callable(args[0]):
                    return torch.cuda.amp.custom_fwd(args[0])
                return lambda f: torch.cuda.amp.custom_fwd(f)
                
            def custom_bwd_patched(*args, **kwargs):
                if len(args) == 1 and callable(args[0]):
                    return torch.cuda.amp.custom_bwd(args[0])
                return lambda f: torch.cuda.amp.custom_bwd(f)
                
            torch.amp.custom_fwd = custom_fwd_patched
            torch.amp.custom_bwd = custom_bwd_patched
                
            from speechbrain.inference.speaker import EncoderClassifier
            # Download and cache from SpeechBrain VoxCeleb model
            _embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": config.device}
            )
            logger.info("SpeechBrain ECAPA-TDNN model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SpeechBrain model ({e}). Using MFCC-based embedding fallback.")
            _embedding_model = False
    return _embedding_model

def extract_speaker_embedding(y, sr):
    """
    Extracts a 192-dimensional ECAPA-TDNN speaker embedding.
    Falls back to a 40-dimensional MFCC-based embedding vector if SpeechBrain is unavailable.
    """
    model = get_embedding_model()
    
    if model:
        try:
            # Resample to 16kHz if needed (SpeechBrain expects 16k mono)
            if sr != 16000:
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                y_16k = y
                
            # Convert to PyTorch tensor [batch=1, time]
            wav = torch.from_numpy(y_16k).unsqueeze(0).to(config.device)
            
            with torch.no_grad():
                embeddings = model.encode_batch(wav)
                # Reshape to 1D vector and return as numpy array
                embedding_np = embeddings.squeeze().cpu().numpy()
                return embedding_np
        except Exception as ex:
            logger.error(f"SpeechBrain embedding extraction failed: {ex}. Falling back to MFCC.")
            
    # MFCC-based fallback embedding (40-dimensional mean-MFCC vector)
    logger.info("Computing MFCC-based speaker embedding representation...")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    embedding_np = np.mean(mfccs, axis=1)
    # Normalize L2
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm
    return embedding_np

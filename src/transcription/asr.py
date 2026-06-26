import whisper
import torch
import librosa
import numpy as np
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("ASR")

_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        model_name = config.asr_model
        logger.info(f"Loading Whisper ASR model: {model_name} on device: {config.device}...")
        try:
            _asr_model = whisper.load_model(model_name, device=config.device)
            logger.info("ASR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ASR model {model_name}: {e}. Falling back to default 'small' model.")
            _asr_model = whisper.load_model("small", device="cpu")
    return _asr_model

def transcribe_segment(segment_audio, sr):
    """
    Transcribes a single segment of audio in a single pass.
    Detects language automatically, and returns the transcribed text.
    If the detected language is Indic/Kannada or config language is set to 'kn',
    it ensures correct Kannada output.
    """
    model = get_asr_model()
    
    # Pre-process audio (Whisper expects 16kHz float32)
    if sr != 16000:
        segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=16000)
        
    try:
        # Run Whisper ASR with greedy decoding for high speed and loop prevention
        # Let Whisper detect the language automatically unless forced in config
        options = dict(
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        # If language is set to 'auto', restrict detection to english and kannada
        if config.language == "auto":
            # Detect language probability using first 30 seconds of the segment
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(segment_audio), n_mels=model.dims.n_mels)
            mel = mel.to(model.device)
            _, probs = model.detect_language(mel)
            
            # Restrict choices to Kannada (kn) and English (en)
            en_prob = probs.get("en", 0.0)
            kn_prob = probs.get("kn", 0.0)
            
            # Force the language to the higher probability choice
            chosen_lang = "kn" if kn_prob >= en_prob else "en"
            options["language"] = chosen_lang
            logger.info(f"Auto-restricted language detection: 'kn' ({kn_prob:.4f}) vs 'en' ({en_prob:.4f}). Selected: '{chosen_lang}'")
        elif config.language and config.language != "auto":
            options["language"] = config.language
            
        res = model.transcribe(segment_audio, fp16=torch.cuda.is_available(), **options)
        text = res.get("text", "").strip()
        detected_lang = res.get("language", "unknown")
        
        logger.info(f"ASR segment transcribed. Detected Lang: {detected_lang}, Text Length: {len(text)}")
        return text, detected_lang
        
    except Exception as ex:
        logger.error(f"ASR transcription failed: {ex}")
        return "", "error"


def transcribe_full_audio_with_words(audio_path):
    """
    Transcribes the full audio file using Whisper with word-level timestamps enabled.
    Returns a list of word dicts: [{"word": str, "start": float, "end": float}, ...]
    and the detected language.
    """
    model = get_asr_model()
    
    # Load audio at 16kHz
    y, sr = librosa.load(audio_path, sr=16000)
    
    logger.info(f"Running full-audio Whisper transcription with word timestamps on {audio_path} ({len(y)/sr:.1f}s)...")
    
    try:
        # Detect language first
        options = dict(
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            word_timestamps=True,
        )
        
        if config.language == "auto":
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(y), n_mels=model.dims.n_mels)
            mel = mel.to(model.device)
            _, probs = model.detect_language(mel)
            en_prob = probs.get("en", 0.0)
            kn_prob = probs.get("kn", 0.0)
            chosen_lang = "kn" if kn_prob >= en_prob else "en"
            options["language"] = chosen_lang
            logger.info(f"Full-audio language detection: 'kn' ({kn_prob:.4f}) vs 'en' ({en_prob:.4f}). Selected: '{chosen_lang}'")
        elif config.language and config.language != "auto":
            options["language"] = config.language
        
        res = model.transcribe(y, fp16=torch.cuda.is_available(), **options)
        detected_lang = res.get("language", "unknown")
        
        # Extract word-level timestamps from all segments
        words = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if word_text:
                    words.append({
                        "word": word_text,
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })
        
        logger.info(f"Full-audio transcription complete. {len(words)} words extracted with timestamps. Lang: {detected_lang}")
        return words, detected_lang
        
    except Exception as ex:
        logger.error(f"Full-audio transcription with word timestamps failed: {ex}")
        return [], "error"

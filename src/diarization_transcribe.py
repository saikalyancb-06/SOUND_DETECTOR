# ==========================================
# FILE: src/diarization_transcribe.py
# ==========================================
import whisper
import torch
import librosa
import re

try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None

_cached_whisper_tiny = None
_cached_whisper_small = None
_cached_diarization = None
_diarization_loaded = False

def devanagari_to_kannada(text):
    converted = []
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            converted.append(chr(cp + 0x0380))
        else:
            converted.append(char)
    return "".join(converted)

def sanitize_script(text):
    # Retain only Kannada characters (\u0C80-\u0CFF), Latin characters (a-zA-Z), digits, spaces, and standard formatting punctuation.
    cleaned = re.sub(r'[^\u0C80-\u0CFFa-zA-Z0-9\s\.,!\?\-\'\"\(\)/\[\]]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

class SpeechProcessor:
    def __init__(self, auth_token=None):
        global _cached_whisper_tiny, _cached_whisper_small, _cached_diarization, _diarization_loaded
        
        # Load and cache both Whisper models
        if _cached_whisper_tiny is None:
            _cached_whisper_tiny = whisper.load_model("tiny")
        self.whisper_tiny = _cached_whisper_tiny

        if _cached_whisper_small is None:
            _cached_whisper_small = whisper.load_model("small")
        self.whisper_small = _cached_whisper_small
        
        # Load and cache Diarization pipeline
        if not _diarization_loaded:
            if Pipeline is not None and auth_token:
                try:
                    _cached_diarization = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
                    )
                except Exception:
                    _cached_diarization = None
            _diarization_loaded = True
        self.diarization_pipeline = _cached_diarization

    def process_audio(self, audio_path):
        """
        Extracts timelines and runs continuous transcription over target slices.
        """
        if self.diarization_pipeline is not None and torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))

        # Load waveform once and reuse; limit to 10s for fast CPU transcription and demonstration
        waveform, sample_rate = librosa.load(audio_path, sr=16000, duration=10.0)

        # Fallback path when diarization backend is unavailable.
        if self.diarization_pipeline is None:
            options = dict(language="kn", beam_size=5, no_speech_threshold=0.6, logprob_threshold=-1.0)
            
            # Pass 1: Tiny model for English/Latin transliteration
            res_tiny = self.whisper_tiny.transcribe(waveform, fp16=torch.cuda.is_available(), **options)
            latin_text = res_tiny["text"].strip()
            
            # Pass 2: Small model for Devanagari mapped to Kannada script
            res_small = self.whisper_small.transcribe(waveform, fp16=torch.cuda.is_available(), **options)
            kannada_text = devanagari_to_kannada(res_small["text"].strip())
            
            combined_text = f"[ಕನ್ನಡ] {kannada_text} / [English] {latin_text}"
            duration = len(waveform) / float(sample_rate)
            return [{
                "speaker": "Speaker_1",
                "start": 0.0,
                "end": round(duration, 2),
                "text": sanitize_script(combined_text)
            }]
            
        diarization = self.diarization_pipeline(audio_path)
        transcription_timeline = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # STRICT GUARDRAIL: Strip explicit names. Map strictly to generic token.
            anonymized_label = f"Speaker_{speaker.split('_')[-1]}"
            start_sample = max(0, int(segment.start * sample_rate))
            end_sample = min(len(waveform), int(segment.end * sample_rate))
            segment_audio = waveform[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue
            
            # Target segment transcription bounded strictly by diarization timestamps
            options = dict(language="kn", beam_size=5, no_speech_threshold=0.6, logprob_threshold=-1.0)
            
            # Pass 1: Tiny model for English/Latin
            res_tiny = self.whisper_tiny.transcribe(segment_audio, fp16=torch.cuda.is_available(), **options)
            latin_text = res_tiny["text"].strip()
            
            # Pass 2: Small model for Devanagari mapped to Kannada
            res_small = self.whisper_small.transcribe(segment_audio, fp16=torch.cuda.is_available(), **options)
            kannada_text = devanagari_to_kannada(res_small["text"].strip())
            
            combined_text = f"[ಕನ್ನಡ] {kannada_text} / [English] {latin_text}"
            
            transcription_timeline.append({
                "speaker": anonymized_label,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": sanitize_script(combined_text)
            })
            
        return transcription_timeline
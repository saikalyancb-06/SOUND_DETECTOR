# ==========================================
# FILE: src/diarization_transcribe.py
# ==========================================
import whisper
import torch
import librosa

try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None

_cached_whisper = None
_cached_diarization = None
_diarization_loaded = False

class SpeechProcessor:
    def __init__(self, auth_token=None):
        global _cached_whisper, _cached_diarization, _diarization_loaded
        
        # Load and cache Whisper model
        if _cached_whisper is None:
            _cached_whisper = whisper.load_model("tiny")
        self.whisper_model = _cached_whisper
        
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
            options = dict(language="kn", beam_size=1, no_speech_threshold=0.6, logprob_threshold=-1.0)
            result = self.whisper_model.transcribe(
                waveform,
                fp16=torch.cuda.is_available(),
                **options
            )
            duration = len(waveform) / float(sample_rate)
            return [{
                "speaker": "Speaker_1",
                "start": 0.0,
                "end": round(duration, 2),
                "text": result["text"].strip()
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
            options = dict(language="kn", beam_size=1, no_speech_threshold=0.6, logprob_threshold=-1.0)
            result = self.whisper_model.transcribe(
                segment_audio,
                fp16=torch.cuda.is_available(),
                **options
            )
            
            transcription_timeline.append({
                "speaker": anonymized_label,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": result["text"].strip()
            })
            
        return transcription_timeline
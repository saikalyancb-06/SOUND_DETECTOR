import yaml
import os
import torch

class Config:
    def __init__(self, config_path="config.yaml"):
        # Load from config.yaml
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}
        else:
            self.data = {}

        self.asr_model = self.data.get("ASR_MODEL", "small")
        self.diarization_model = self.data.get("DIARIZATION_MODEL", "ecapa-tdnn")
        self.classifier = self.data.get("CLASSIFIER", "Voting")
        device_conf = self.data.get("DEVICE", "auto")
        
        if device_conf == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_conf

        self.language = self.data.get("LANGUAGE", "kn")
        self.noise_reduction = self.data.get("NOISE_REDUCTION", "spectral_subtraction")
        self.vad_threshold = self.data.get("VAD_THRESHOLD", 0.5)
        self.min_turn_duration = self.data.get("MIN_TURN_DURATION", 0.8)
        self.merge_turn_gap = self.data.get("MERGE_TURN_GAP", 0.8)
        self.pitch_male_threshold = self.data.get("PITCH_MALE_THRESHOLD", 165.0)

# Global configuration instance
config = Config()

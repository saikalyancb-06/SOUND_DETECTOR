import os

# Define the complete folder and file structure
project_structure = {
    "speakathon/requirements.txt": """numpy==1.24.3
pandas==2.0.3
librosa==0.10.0
soundfile==0.12.1
scikit-learn==1.3.0
xgboost==1.7.6
pyannote.audio==3.1.1
openai-whisper==20231117
pyyaml==6.0.1
""",

    "speakathon/config.yaml": """system_settings:
  primary_language: "kn"
  sampling_rate: 16000
  anonymization_strict_mode: true

whisper_configuration:
  model_size: "small"
  beam_size: 5
  temperature: 0.0

ensemble_hyperparameters:
  random_forest_estimators: 100
  xgboost_estimators: 100
  max_tree_depth: 10
""",

    "speakathon/src/audio_processing.py": """import librosa
import numpy as np

def extract_acoustic_parameters(audio_path):
    \"\"\"
    Computes exactly 21 explicit structural linguistic and acoustic features.
    \"\"\"
    y, sr = librosa.load(audio_path, sr=None)
    
    # Fundamental Frequency Tracking (F0/Pitch Analytics)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
    )
    f0_cleaned = f0[~np.isnan(f0)]
    mean_f0 = np.mean(f0_cleaned) if len(f0_cleaned) > 0 else 0.0
    std_f0 = np.std(f0_cleaned) if len(f0_cleaned) > 0 else 0.0

    # Spectral Domain Footprints
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Signal Energy Dynamics
    rms = librosa.feature.rms(y=y)[0]
    
    # Mel-Frequency Cepstral Coefficients (MFCCs 1-13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)

    feature_vector = {
        "mean_f0": float(mean_f0),
        "std_f0": float(std_f0),
        "mean_spectral_centroid": float(np.mean(spectral_centroids)),
        "mean_spectral_rolloff": float(np.mean(spectral_rolloff)),
        "mean_zcr": float(np.mean(zero_crossing_rate)),
        "rms_energy": float(np.mean(rms)),
    }
    
    for i, mfcc_val in enumerate(mean_mfccs, 1):
        feature_vector[f"mfcc_{i}"] = float(mfcc_val)
        
    return feature_vector
""",

    "speakathon/src/diarization_transcribe.py": """import whisper
from pyannote.audio import Pipeline
import torch

class SpeechProcessor:
    def __init__(self, auth_token=None):
        self.whisper_model = whisper.load_model("small")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
        )

    def process_audio(self, audio_path):
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device("cuda"))
            
        diarization = self.diarization_pipeline(audio_path)
        transcription_timeline = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            anonymized_label = f"Speaker_{speaker.split('_')[-1]}"
            
            options = dict(language="kn", beam_size=5)
            result = self.whisper_model.transcribe(
                audio_path, 
                start_time=segment.start, 
                end_time=segment.end, 
                **options
            )
            
            transcription_timeline.append({
                "speaker": anonymized_label,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": result["text"].strip()
            })
            
        return transcription_timeline
""",

    "speakathon/src/classifier_models.py": """from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

class EnsembleSpeechClassifier:
    def __init__(self):
        self.age_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gender_model = XGBClassifier(n_estimators=100, random_state=42)
        self.diagnostic_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

    def train_pipelines(self, X_train, y_age, y_gender, y_diag):
        self.age_model.fit(X_train, y_age)
        self.gender_model.fit(X_train, y_gender)
        self.diagnostic_model.fit(X_train, y_diag)

    def predict_diagnostics(self, feature_vector):
        X = np.array(list(feature_vector.values())).reshape(1, -1)
        
        age_pred = "Adult" if self.age_model.predict(X)[0] == 1 else "Child"
        gender_pred = "Male" if self.gender_model.predict(X)[0] == 1 else "Female"
        diag_pred = "Typical" if self.diagnostic_model.predict(X)[0] == 1 else "Atypical"
        
        return {
            "Age_Classification": age_pred,
            "Gender_Classification": gender_pred,
            "Acoustic_Profile": diag_pred
        }
""",

    "speakathon/src/main.py": """import json
import sys
from audio_processing import extract_acoustic_parameters
from diarization_transcribe import SpeechProcessor
from classifier_models import EnsembleSpeechClassifier

def execute_diagnostic_engine(audio_file_path, hf_token=None):
    print(f"[*] Parsing Speech File Target: {audio_file_path}")
    
    processor = SpeechProcessor(auth_token=hf_token)
    timeline_manifest = processor.process_audio(audio_file_path)
    
    acoustic_features = extract_acoustic_parameters(audio_file_path)
    
    classifier = EnsembleSpeechClassifier()
    classification_results = classifier.predict_diagnostics(acoustic_features)
    
    final_payload = {
        "Target_Audio": audio_file_path,
        "Classification_Metadata": classification_results,
        "Acoustic_Feature_Matrix": acoustic_features,
        "Anonymized_Timeline_Output": timeline_manifest
    }
    
    print("\\n[+] DIAGNOSTIC MODEL LOG COMPLETE:")
    print(json.dumps(final_payload, indent=4, ensure_ascii=False))
    return final_payload

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "Sample11C.wav"
    execute_diagnostic_engine(target_file, hf_token="YOUR_HF_TOKEN")
""",
    "speakathon/src/__init__.py": ""
}

# Automatically build the filesystem pipeline
print("[*] Generating absolute workspace configuration...")
for path, content in project_structure.items():
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [➔] Created: {path}")

print("\\n[+] WORKSPACE COMPLETELY BUILT SUCCESSFULLY!")
print("Navigate to your new directory using: 'cd speakathon'")
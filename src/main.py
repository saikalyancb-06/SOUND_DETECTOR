# ==========================================
# FILE: src/main.py
# ==========================================
import json
import sys
import os
import static_ffmpeg
static_ffmpeg.add_paths()

# Reconfigure stdout/stderr to UTF-8 for Windows console support of Kannada text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from audio_processing import extract_acoustic_parameters
from diarization_transcribe import SpeechProcessor
from classifier_models import EnsembleSpeechClassifier

def execute_diagnostic_engine(audio_file_path, hf_token=None, model_dir="models"):
    """
    Coordinates global processing, pipeline steps, and payload returns.
    """
    print(f"[*] Parsing Speech File Target: {audio_file_path}")
    
    # 1. Segments, Timestamps, and Anonymized Text Layout
    processor = SpeechProcessor(auth_token=hf_token)
    timeline_manifest = processor.process_audio(audio_file_path)
    
    # 2. Extract Acoustic Feature Matrix Arrays
    acoustic_features = extract_acoustic_parameters(audio_file_path)
    
    # 3. Model Pipeline Execution
    classifier = EnsembleSpeechClassifier()
    if os.path.isdir(model_dir):
        try:
            classifier.load_pipelines(model_dir)
        except Exception:
            # Fall back to rule-based mode when artifacts are absent or incompatible.
            pass

    classification_results = classifier.predict_diagnostics(acoustic_features)
    
    # Assemble Clean Deliverable 
    final_payload = {
        "Target_Audio": audio_file_path,
        "Classification_Metadata": classification_results,
        "Acoustic_Feature_Matrix": acoustic_features,
        "Anonymized_Timeline_Output": timeline_manifest
    }
    
    print("\n=======================================================")
    print("            DIAGNOSTIC REPORT SUMMARY                  ")
    print("=======================================================")
    print(f" Target Audio:     {audio_file_path}")
    print(f" Age Group:        {classification_results.get('Age_Classification')}")
    print(f" Gender:           {classification_results.get('Gender_Classification')}")
    print(f" Clinical Profile: {classification_results.get('Acoustic_Profile')}")
    print(f" Prediction Mode:  {classification_results.get('Prediction_Mode')}")
    print("=======================================================")

    print("\n[+] DIAGNOSTIC MODEL LOG COMPLETE:")
    print(json.dumps(final_payload, indent=4, ensure_ascii=False))
    return final_payload

if __name__ == "__main__":
    # Ensure a local file parameter is provided safely
    target_file = sys.argv[1] if len(sys.argv) > 1 else "Datasets/Sample11C.wav"
    hf_token = os.getenv("HF_TOKEN")
    execute_diagnostic_engine(target_file, hf_token=hf_token)
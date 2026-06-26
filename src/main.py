# ==========================================
# FILE: src/main.py
# ==========================================
import json
import sys
import os
import numpy as np
import librosa

# Reconfigure stdout/stderr to UTF-8 for Windows console support of Kannada text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Append src path to python path to resolve legacy imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_processing import extract_acoustic_parameters # Legacy import preservation
except ModuleNotFoundError:
    from src.audio_processing import extract_acoustic_parameters

from src.audio.feature_extractor import extract_features
from src.diarization_transcribe import SpeechProcessor
from src.classifier_models import EnsembleSpeechClassifier
from src.diarization.speaker_diarizer import diarize_audio
from src.utils.logger import get_logger

logger = get_logger("Main")

def format_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    ms = int(round((seconds - int(seconds)) * 100))
    if ms >= 100:
        ms = 99
    return f"{m:02d}:{s:02d}.{ms:02d}"

def execute_diagnostic_engine(audio_file_path, hf_token=None, model_dir="models", debug=False):
    """
    Main pipeline entry point. 
    1. Runs VAD, noise reduction, and speaker clustering to generate timelines.
    2. Groups waveforms per speaker, extracts and aggregates features.
    3. Runs ensemble classifiers (gender, age, clinical profile) per speaker.
    4. Generates a comprehensive structured report.
    """
    logger.info(f"[*] Starting Speech Diagnostic Engine for: {audio_file_path}")
    
    # 1. Timeline & Speaker Segmentation
    processor = SpeechProcessor(auth_token=hf_token)
    result = processor.process_audio(audio_file_path)
    
    # Handle new 3-tuple return: (transcript, merged_diarization, diagnostics)
    if isinstance(result, tuple) and len(result) == 3:
        timeline_manifest, merged_diarization, diarization_diagnostics = result
    else:
        # Fallback for legacy return format
        timeline_manifest = result
        merged_diarization = []
        diarization_diagnostics = getattr(processor, "diarization_diagnostics", {})
    
    # Reload clean audio to extract exact waveforms
    y, sr = librosa.load(audio_file_path, sr=16000)
    
    # Group segment waveforms and transcripts by unique speaker ID
    # Use merged_diarization for feature extraction (longer segments = better features)
    speaker_data = {}
    
    # First, populate waveforms from the merged diarization segments
    for seg in merged_diarization:
        if isinstance(seg, tuple):
            start_sec, end_sec, speaker_id, seg_audio, confidence = seg
        else:
            continue
        
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = {
                "waveforms": [],
                "transcripts": [],
                "speaking_time": 0.0
            }
        speaker_data[speaker_id]["waveforms"].append(seg_audio)
        speaker_data[speaker_id]["speaking_time"] += (end_sec - start_sec)
    
    # Then, populate transcripts from the word-level aligned timeline
    for segment in timeline_manifest:
        speaker_id = segment["speaker"]
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = {
                "waveforms": [],
                "transcripts": [],
                "speaking_time": 0.0
            }
        speaker_data[speaker_id]["transcripts"].append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })


    # Load Ensemble Classifiers
    classifier = EnsembleSpeechClassifier()
    if os.path.isdir(model_dir):
        try:
            classifier.load_pipelines(model_dir)
        except Exception:
            pass # Fall back to rule-based logic if models aren't trained yet
            
    # 2. Extract, Aggregate Features and Predict per Speaker
    speaker_features = {}
    speaker_analysis = {}
    from concurrent.futures import ThreadPoolExecutor
    
    for speaker_id, data in speaker_data.items():
        logger.info(f"Extracting segment features for speaker: {speaker_id}...")
        segment_features = []
        
        valid_wavs = []
        for wav in data["waveforms"]:
            if len(wav) >= 160:
                max_samples = int(15.0 * sr)
                valid_wavs.append(wav[:max_samples])
                
        if valid_wavs:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(extract_features, wav, sr) for wav in valid_wavs]
                segment_features = [f.result() for f in futures]
        
        if not segment_features:
            dummy_wav = np.zeros(16000, dtype=np.float32)
            segment_features.append(extract_features(dummy_wav, sr))
            
        # Store mean features for validation check
        keys = segment_features[0].keys()
        mean_features = {}
        for k in keys:
            mean_features[k] = float(np.mean([f[k] for f in segment_features]))
        speaker_features[speaker_id] = mean_features
        
        # Log complete feature vector during debug mode
        if debug:
            logger.info(f"=== DEBUG: Complete Feature Vector for {speaker_id} ===")
            for k, v in sorted(mean_features.items()):
                print(f"  [DEBUG] {speaker_id} - {k}: {v}", flush=True)
            logger.info("====================================================")
            
        # Ignore silence and low-confidence segments, perform majority voting/probability averaging
        valid_predictions = []
        active_segment_features = []
        for feat in segment_features:
            # Silence detection thresholds
            if feat.get("rms_energy", 0.0) < 0.005 or feat.get("silence_ratio", 0.0) > 0.8:
                continue
            active_segment_features.append(feat)
            valid_predictions.append(classifier.predict_diagnostics(feat))
            
        if not valid_predictions:
            active_segment_features = segment_features
            for feat in segment_features:
                valid_predictions.append(classifier.predict_diagnostics(feat))
                
        gender_male_probs = []
        age_adult_probs = []
        diag_typical_probs = []
        
        for pred in valid_predictions:
            g_conf = pred.get("Gender_Confidence", 1.0)
            a_conf = pred.get("Age_Confidence", 1.0)
            d_conf = pred.get("Acoustic_Confidence", 1.0)
            
            g_label = pred.get("Gender_Classification", "Unknown")
            if g_label == "Male":
                gender_male_probs.append(g_conf)
            elif g_label == "Female":
                gender_male_probs.append(1.0 - g_conf)
            else:
                gender_male_probs.append(0.50)
                
            a_label = pred.get("Age_Classification", "Unknown")
            if a_label == "Adult":
                age_adult_probs.append(a_conf)
            elif a_label == "Child":
                age_adult_probs.append(1.0 - a_conf)
            else:
                age_adult_probs.append(0.50)
                
            d_label = pred.get("Acoustic_Profile", "Unknown")
            if d_label == "Typical":
                diag_typical_probs.append(d_conf)
            elif d_label == "Atypical":
                diag_typical_probs.append(1.0 - d_conf)
            else:
                diag_typical_probs.append(0.50)
                
        avg_male_prob = float(np.mean(gender_male_probs)) if gender_male_probs else 0.50
        avg_adult_prob = float(np.mean(age_adult_probs)) if age_adult_probs else 0.50
        avg_typical_prob = float(np.mean(diag_typical_probs)) if diag_typical_probs else 0.50
        
        if avg_male_prob >= 0.50:
            final_gender = "Male"
            final_gender_conf = avg_male_prob
        else:
            final_gender = "Female"
            final_gender_conf = 1.0 - avg_male_prob
            
        if avg_adult_prob >= 0.50:
            final_age = "Adult"
            final_age_conf = avg_adult_prob
        else:
            final_age = "Child"
            final_age_conf = 1.0 - avg_adult_prob
            
        if avg_typical_prob >= 0.50:
            final_profile = "Typical"
            final_profile_conf = avg_typical_prob
        else:
            final_profile = "Atypical"
            final_profile_conf = 1.0 - avg_typical_prob
            
        mean_features = {}
        for k in keys:
            mean_features[k] = float(np.mean([f[k] for f in active_segment_features]))
            
        rationales = [pred.get("Explainability_Rationale", "") for pred in valid_predictions]
        unique_rationales = list(set([r for r in rationales if r]))
        representative_rationale = " ".join(unique_rationales[:2]) if unique_rationales else "No features identified."
        
        speaker_analysis[speaker_id] = {
            "gender": final_gender,
            "gender_confidence": round(final_gender_conf, 4),
            "age": final_age,
            "age_confidence": round(final_age_conf, 4),
            "profile": final_profile,
            "profile_confidence": round(final_profile_conf, 4),
            "speaking_time": round(data["speaking_time"], 2),
            "features": mean_features,
            "transcripts": data["transcripts"],
            "rationale": representative_rationale
        }

    # Verify that every speaker's feature vector is different before inference
    all_spks = list(speaker_features.keys())
    for i in range(len(all_spks)):
        for j in range(i + 1, len(all_spks)):
            spk1, spk2 = all_spks[i], all_spks[j]
            feat1, feat2 = speaker_features[spk1], speaker_features[spk2]
            is_identical = True
            for k in feat1.keys():
                if abs(feat1[k] - feat2[k]) > 1e-7:
                    is_identical = False
                    break
            if is_identical:
                logger.warning(f"[!] Critical Warning: Speaker {spk1} and Speaker {spk2} have identical feature vectors!")
            else:
                logger.info(f"[+] Verified: Feature vectors for {spk1} and {spk2} are distinct.")

    # 3. Print Structured Speaker Analysis Report
    print("\n=======================================================")
    print("               SPEAKER ANALYSIS REPORT                 ")
    print("=======================================================")
    
    # Chronological Transcript Section printed first
    print("\n-------------------------------------------------------")
    print("               CHRONOLOGICAL TRANSCRIPT                ")
    print("-------------------------------------------------------")
    for idx, turn in enumerate(timeline_manifest):
        if idx > 0:
            print("\n-------------------------------------\n")
        time_range = f"{format_time(turn['start'])} – {format_time(turn['end'])}"
        print(f"{time_range}\n")
        print(f"{turn['speaker']}\n")
        print(f"{turn['text']}")
    print("-------------------------------------------------------\n")
    
    for speaker_id, analysis in sorted(speaker_analysis.items()):
        print(f"\n{speaker_id}")
        print(f"  Gender:        {analysis['gender']} (Confidence: {analysis['gender_confidence']*100:.1f}%)")
        print(f"  Age Group:     {analysis['age']} (Confidence: {analysis['age_confidence']*100:.1f}%)")
        print(f"  Clinical Rec:  {analysis['profile']} (Confidence: {analysis['profile_confidence']*100:.1f}%)")
        print(f"  Speaking Time: {analysis['speaking_time']:.1f} seconds")
        print(f"  Acoustic F0:   {analysis['features'].get('mean_f0', 0.0):.1f} Hz (RMS Energy: {analysis['features'].get('rms_energy', 0.0):.4f})")
        print(f"  Explainability Rationale: {analysis['rationale']}")
    print("=======================================================\n")

    # Save output metadata payload matching original format
    final_payload = {
        "Target_Audio": audio_file_path,
        "Speaker_Diagnostics": {sp: {
            "Age": val["age"], "Age_Confidence": val["age_confidence"],
            "Gender": val["gender"], "Gender_Confidence": val["gender_confidence"],
            "Clinical_Profile": val["profile"], "Rationale": val["rationale"]
        } for sp, val in speaker_analysis.items()},
        "Anonymized_Timeline_Output": timeline_manifest,
        "Diarization_Diagnostics": diarization_diagnostics
    }
    
    # Print Speaker Similarity Matrix
    if diarization_diagnostics and "centroids" in diarization_diagnostics:
        centroids_dict = diarization_diagnostics["centroids"]
        sorted_labels = sorted([int(k) for k in centroids_dict.keys()])
        if len(sorted_labels) > 0:
            print("\n=======================================================")
            print("         SPEAKER CENTROID SIMILARITY MATRIX            ")
            print("=======================================================")
            header = "          " + " ".join([f"Spk {l+1:<3}" for l in sorted_labels])
            print(header)
            for l1 in sorted_labels:
                c1 = np.array(centroids_dict[str(l1)])
                row_str = f"Speaker {l1+1:<2}"
                for l2 in sorted_labels:
                    c2 = np.array(centroids_dict[str(l2)])
                    sim = float(np.dot(c1, c2))
                    row_str += f" {sim:.3f}  "
                print(row_str)
            print("=======================================================\n")
            
    # Generate Reports and Run Error Analysis
    try:
        from src.reports.report_generator import generate_json_report, generate_html_report, generate_pdf_report
        from src.evaluation.error_analysis import generate_error_log
        
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        os.makedirs("reports", exist_ok=True)
        
        json_report_path = f"reports/speaker_analysis_{base_name}.json"
        html_report_path = f"reports/speaker_analysis_{base_name}.html"
        pdf_report_path = f"reports/speaker_analysis_{base_name}.pdf"
        
        generate_json_report(final_payload, json_report_path)
        generate_html_report(final_payload, speaker_analysis, html_report_path)
        generate_pdf_report(final_payload, speaker_analysis, pdf_report_path)
        
        # Match with ground truth in metadata.csv if available
        ground_truth = None
        try:
            import pandas as pd
            metadata_csv = "Datasets/metadata.csv"
            if os.path.exists(metadata_csv):
                df_meta = pd.read_csv(metadata_csv)
                matching_rows = df_meta[df_meta["filename"].apply(lambda x: os.path.basename(str(x)) == os.path.basename(audio_file_path))]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    ground_truth = {
                        "gender": str(row.get("gender", "")),
                        "age": str(row.get("age_group", "")),
                        "clinical_profile": str(row.get("clinical_profile", ""))
                    }
        except Exception as ge:
            logger.warning(f"Could not load ground truth metadata: {ge}")
            
        for sp_id, val in speaker_analysis.items():
            generate_error_log(
                audio_file_path=audio_file_path,
                speaker_id=sp_id,
                predictions=val,
                features=val["features"],
                ground_truth=ground_truth,
                threshold=0.65,
                log_dir="reports"
            )
            
    except Exception as re:
        logger.error(f"Failed to generate evaluation reports/logs: {re}")
    
    return final_payload

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speech Diagnostic Clinical Pipeline")
    parser.add_argument("audio_path", nargs="?", default="Datasets/Sample11C.wav", help="Path to input WAV file")
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostics logging")
    args = parser.parse_args()
    
    execute_diagnostic_engine(args.audio_path, debug=args.debug)
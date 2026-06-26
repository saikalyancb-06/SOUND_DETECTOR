# ==========================================
# FILE: src/evaluation/error_analysis.py
# ==========================================
import os
import json
from datetime import datetime

def identify_low_confidence(predictions, threshold=0.85):
    """
    Identifies if any prediction confidence is below the threshold.
    Returns a list of reasons/triggers if any are found.
    """
    triggers = []
    
    gender_conf = predictions.get("gender_confidence", 1.0)
    age_conf = predictions.get("age_confidence", 1.0)
    profile_conf = predictions.get("profile_confidence", 1.0)
    
    if gender_conf < threshold:
        triggers.append(f"Gender confidence ({gender_conf:.2f}) is below threshold ({threshold})")
    if age_conf < threshold:
        triggers.append(f"Age confidence ({age_conf:.2f}) is below threshold ({threshold})")
    if profile_conf < threshold:
        triggers.append(f"Clinical profile confidence ({profile_conf:.2f}) is below threshold ({threshold})")
        
    return triggers

def check_acoustic_outliers(features):
    """
    Checks if acoustic parameters reside outside standard physiological bounds.
    """
    outliers = []
    f0 = features.get("mean_f0", 0.0)
    zcr = features.get("mean_zcr", 0.0)
    rms = features.get("rms_energy", 0.0)
    
    # Check if F0 is extremely high or low
    if f0 > 0.0:
        if f0 < 60.0 or f0 > 500.0:
            outliers.append(f"Physiological outlier F0 ({f0:.1f} Hz) is outside standard bounds (60-500 Hz).")
    else:
        outliers.append("Unvoiced/silent signal detected (F0 is zero).")
        
    # Check energy
    if rms < 0.005:
        outliers.append(f"Low signal-to-noise ratio / low energy (RMS: {rms:.5f}).")
        
    # Check ZCR
    if zcr > 0.3:
        outliers.append(f"High Zero Crossing Rate ({zcr:.4f}), indicating excessive noise or fricatives.")
        
    return outliers

def generate_error_log(audio_file_path, speaker_id, predictions, features, ground_truth=None, threshold=0.85, log_dir="reports"):
    """
    Generates and saves a structured JSON log entry for low-confidence or incorrect predictions.
    """
    triggers = identify_low_confidence(predictions, threshold)
    outliers = check_acoustic_outliers(features)
    
    # Check if predictions do not match ground truth (if ground truth is supplied)
    is_incorrect = False
    mismatches = []
    if ground_truth:
        gt_gender = ground_truth.get("gender")
        gt_age = ground_truth.get("age")
        gt_profile = ground_truth.get("clinical_profile")
        
        pred_gender = predictions.get("gender")
        pred_age = predictions.get("age")
        pred_profile = predictions.get("profile")
        
        if gt_gender and pred_gender and gt_gender.lower() != pred_gender.lower():
            is_incorrect = True
            mismatches.append(f"Gender mismatch: Pred={pred_gender}, GT={gt_gender}")
        if gt_age and pred_age and gt_age.lower() != pred_age.lower():
            is_incorrect = True
            mismatches.append(f"Age mismatch: Pred={pred_age}, GT={gt_age}")
        if gt_profile and pred_profile and gt_profile.lower() != pred_profile.lower():
            is_incorrect = True
            mismatches.append(f"Profile mismatch: Pred={pred_profile}, GT={gt_profile}")
            
    # We log if it is low confidence OR incorrect
    if not triggers and not is_incorrect:
        return None # No issue to log
        
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "file_path": audio_file_path,
        "speaker_id": speaker_id,
        "predictions": {
            "gender": predictions.get("gender"),
            "gender_confidence": predictions.get("gender_confidence"),
            "age": predictions.get("age"),
            "age_confidence": predictions.get("age_confidence"),
            "profile": predictions.get("profile"),
            "profile_confidence": predictions.get("profile_confidence")
        },
        "ground_truth": ground_truth,
        "low_confidence_triggers": triggers,
        "acoustic_outliers": outliers,
        "mismatches": mismatches,
        "rationale": predictions.get("rationale", "")
    }
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "error_analysis_logs.json")
    
    existing_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                existing_logs = json.load(f)
        except Exception:
            existing_logs = []
            
    existing_logs.append(log_entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=2, ensure_ascii=False)
        
    return log_entry

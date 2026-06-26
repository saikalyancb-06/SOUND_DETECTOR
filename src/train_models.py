import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import static_ffmpeg
static_ffmpeg.add_paths()
import sys
import librosa

# Reconfigure stdout/stderr to UTF-8 for Windows console support of Kannada text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.audio.feature_extractor import extract_features
from src.classifier_models import EnsembleSpeechClassifier

def build_training_table(metadata_csv, dataset_dir):
    rows = []
    skipped_files = []
    
    logger_print = lambda x: print(f"[*] {x}", flush=True)

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_csv}. Training requires a valid metadata CSV file.")

    metadata = pd.read_csv(metadata_csv)
    # Support columns: filename,gender,age_group,speaker_id
    required = {"filename", "gender", "age_group"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    total = len(metadata)
    logger_print(f"Loading metadata from {metadata_csv}. Training on {total} entries.")

    for idx, row in metadata.iterrows():
        filename = row["filename"]
        audio_path = os.path.join(dataset_dir, filename) if not os.path.isabs(filename) else filename
        
        if not os.path.exists(audio_path):
            skipped_files.append(audio_path)
            continue

        logger_print(f"[{idx+1}/{total}] Extracting features from {os.path.basename(audio_path)}...")
        try:
            y, sr = librosa.load(audio_path, sr=16000, duration=15.0)
            feature_vector = extract_features(y, sr)
        except Exception as e:
            logger_print(f"Error processing {filename}: {e}")
            skipped_files.append(audio_path)
            continue
            
        gender_label = 1 if str(row["gender"]).strip().lower() == "male" else 0
        age_label = 1 if str(row["age_group"]).strip().lower() == "adult" else 0
        
        # Clinical Profile logic
        diag_val = 1
        if "clinical_profile" in row:
            diag_val = 0 if str(row["clinical_profile"]).strip().lower() == "atypical" else 1
        elif "diagnostic_label" in row:
            diag_val = int(row["diagnostic_label"])
            
        rows.append({
            "file_path": audio_path,
            **feature_vector,
            "age_label": age_label,
            "gender_label": gender_label,
            "diagnostic_label": diag_val,
        })
        
        # Synthesize Male data if sample is Female to balance classes (Feature-Level Augmentation)
        if gender_label == 0:
            male_feat = feature_vector.copy()
            # Pitch shift down by 10 semitones (factor = 2 ** (-10 / 12) = 0.5612)
            pitch_factor = 2 ** (-10 / 12)
            for k in ["mean_f0", "std_f0", "pitch_range", "formant_F1", "formant_F2", "formant_F3", "formant_F4"]:
                if k in male_feat:
                    male_feat[k] *= pitch_factor
            rows.append({
                "file_path": f"{audio_path}_synthetic_male",
                **male_feat,
                "age_label": age_label,
                "gender_label": 1, # Male
                "diagnostic_label": diag_val,
            })
            
            # Synthesize Male pitch shift down by 4 semitones too to balance classes further
            male_feat_pd = male_feat.copy()
            pitch_factor_4 = 2 ** (-4 / 12)
            for k in ["mean_f0", "std_f0", "pitch_range", "formant_F1", "formant_F2", "formant_F3", "formant_F4"]:
                if k in male_feat_pd:
                    male_feat_pd[k] *= pitch_factor_4
            rows.append({
                "file_path": f"{audio_path}_synthetic_male_pd",
                **male_feat_pd,
                "age_label": age_label,
                "gender_label": 1, # Male
                "diagnostic_label": diag_val,
            })
        
        # Feature-level Augmentations (Pitch down by 4 semitones and Time stretch slow rate=0.9)
        # 1. Pitch shift down by 4 semitones (factor = 2 ** (-4/12) = 0.7937)
        feat_pitch_down = feature_vector.copy()
        pitch_factor_4 = 2 ** (-4 / 12)
        for k in ["mean_f0", "std_f0", "pitch_range", "formant_F1", "formant_F2", "formant_F3", "formant_F4"]:
            if k in feat_pitch_down:
                feat_pitch_down[k] *= pitch_factor_4
        rows.append({
            "file_path": f"{audio_path}_pitch_down",
            **feat_pitch_down,
            "age_label": age_label,
            "gender_label": gender_label,
            "diagnostic_label": diag_val,
        })
        
        # 2. Time stretch (slower speed rate = 0.9)
        feat_stretch_slow = feature_vector.copy()
        if "speaking_rate" in feat_stretch_slow:
            feat_stretch_slow["speaking_rate"] *= 0.9
        if "pause_duration" in feat_stretch_slow:
            feat_stretch_slow["pause_duration"] /= 0.9
        rows.append({
            "file_path": f"{audio_path}_stretch_slow",
            **feat_stretch_slow,
            "age_label": age_label,
            "gender_label": gender_label,
            "diagnostic_label": diag_val,
        })

    if not rows:
        raise RuntimeError("No training samples successfully processed.")

    if skipped_files:
        logger_print(f"Skipped {len(skipped_files)} missing/unreadable audio files.")

    return pd.DataFrame(rows)

def train_and_save(df, model_dir, test_size=0.2, random_state=42):
    feature_cols = [
        c for c in df.columns
        if c not in {"file_path", "age_label", "gender_label", "diagnostic_label"}
    ]
    # Ensure keys are sorted for consistent column mapping
    feature_cols = sorted(feature_cols)

    X = df[feature_cols].values
    y_age = df["age_label"].values
    y_gender = df["gender_label"].values
    y_diag = df["diagnostic_label"].values

    idx = np.arange(len(df))
    # Stratify by age label only if both classes are present in y_age
    stratify_labels = y_age if len(np.unique(y_age)) > 1 else None
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify_labels)

    X_train, X_test = X[train_idx], X[test_idx]
    y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]
    y_gender_train, y_gender_test = y_gender[train_idx], y_gender[test_idx]
    y_diag_train, y_diag_test = y_diag[train_idx], y_diag[test_idx]

    classifier = EnsembleSpeechClassifier()
    classifier.train_pipelines(X_train, y_age_train, y_gender_train, y_diag_train)
    classifier.save_pipelines(model_dir)

    # Evaluate predictions using the unified predict_diagnostics interface (handles fallbacks)
    age_preds = []
    gender_preds = []
    diag_preds = []
    
    for row_idx in range(len(X_test)):
        # Construct feature dictionary from columns
        feat_dict = {feat_name: X_test[row_idx, col_idx] for col_idx, feat_name in enumerate(feature_cols)}
        preds = classifier.predict_diagnostics(feat_dict)
        
        # Convert text predictions back to integer labels (Adult=1, Child=0; Male=1, Female=0; Typical=1, Atypical=0)
        age_label_val = 1 if preds["Age_Classification"] == "Adult" else 0
        gender_label_val = 1 if preds["Gender_Classification"] == "Male" else 0
        diag_label_val = 1 if preds["Acoustic_Profile"] == "Typical" else 0
        
        age_preds.append(age_label_val)
        gender_preds.append(gender_label_val)
        diag_preds.append(diag_label_val)
        
    age_pred = np.array(age_preds)
    gender_pred = np.array(gender_preds)
    diag_pred = np.array(diag_preds)

    # Map available unique classes to target labels for classification report
    unique_age = np.unique(np.concatenate([y_age_test, age_pred]))
    age_target_names = ["Child" if c == 0 else "Adult" for c in sorted(unique_age)]
    
    unique_gender = np.unique(np.concatenate([y_gender_test, gender_pred]))
    gender_target_names = ["Female" if c == 0 else "Male" for c in sorted(unique_gender)]
    
    unique_diag = np.unique(np.concatenate([y_diag_test, diag_pred]))
    diag_target_names = ["Atypical" if c == 0 else "Typical" for c in sorted(unique_diag)]

    # Convert numeric predictions to labels for confusion matrix / reports
    metrics = {
        "age_balanced_accuracy": float(balanced_accuracy_score(y_age_test, age_pred)),
        "gender_balanced_accuracy": float(balanced_accuracy_score(y_gender_test, gender_pred)),
        "diag_balanced_accuracy": float(balanced_accuracy_score(y_diag_test, diag_pred)),
        "age_report": classification_report(y_age_test, age_pred, labels=sorted(unique_age), target_names=age_target_names, zero_division=0),
        "gender_report": classification_report(y_gender_test, gender_pred, labels=sorted(unique_gender), target_names=gender_target_names, zero_division=0),
        "diag_report": classification_report(y_diag_test, diag_pred, labels=sorted(unique_diag), target_names=diag_target_names, zero_division=0),
        "n_samples": int(len(df)),
    }

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train ensemble speech diagnostic models using metadata.")
    parser.add_argument("--metadata-csv", default="Datasets/metadata.csv", help="Path to metadata CSV.")
    parser.add_argument("--dataset-dir", default="Datasets", help="Directory containing audio files.")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts.")
    parser.add_argument("--metrics-out", default="models/training_metrics.txt", help="Path to write training metrics.")
    args = parser.parse_args()

    cache_csv = "Datasets/extracted_features.csv"
    if os.path.exists(cache_csv):
        print(f"[+] Found cached features at {cache_csv}. Loading directly...", flush=True)
        df = pd.read_csv(cache_csv)
    else:
        df = build_training_table(args.metadata_csv, args.dataset_dir)
        df.to_csv(cache_csv, index=False)
        print(f"[+] Features cached to {cache_csv}", flush=True)

    metrics = train_and_save(df, args.model_dir)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write(f"Samples (including augmented): {metrics['n_samples']}\n")
        f.write(f"Age Balanced Accuracy: {metrics['age_balanced_accuracy']:.4f}\n")
        f.write(f"Gender Balanced Accuracy: {metrics['gender_balanced_accuracy']:.4f}\n")
        f.write(f"Diagnostic Balanced Accuracy: {metrics['diag_balanced_accuracy']:.4f}\n\n")
        f.write("Age Report\n")
        f.write(metrics["age_report"] + "\n")
        f.write("Gender Report\n")
        f.write(metrics["gender_report"] + "\n")
        f.write("Diagnostic Report\n")
        f.write(metrics["diag_report"] + "\n")

    print("[+] Training complete. Ensemble models saved to:", args.model_dir)
    print("[+] Metrics written to:", args.metrics_out)

if __name__ == "__main__":
    main()

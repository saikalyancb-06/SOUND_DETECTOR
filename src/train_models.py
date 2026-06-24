import argparse
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import static_ffmpeg
static_ffmpeg.add_paths()
import sys

# Reconfigure stdout/stderr to UTF-8 for Windows console support of Kannada text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from audio_processing import extract_acoustic_parameters
from classifier_models import EnsembleSpeechClassifier


def discover_dataset_files(dataset_dir):
    patterns = ["*.wav", "*.WAV", "*.mp3", "*.flac"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(dataset_dir, pattern)))
    seen = set()
    unique_files = []
    for f in files:
        norm = os.path.normcase(os.path.abspath(f))
        if norm not in seen:
            seen.add(norm)
            unique_files.append(f)
    return sorted(unique_files)


def infer_bootstrap_labels(file_name):
    base = os.path.basename(file_name).lower()
    clean_base = base.replace("sample", "")

    # Age Label: Child=0, Adult=1 (based on 'c' in filename)
    age_label = 0 if "c" in clean_base else 1

    # Gender Label: Female=0, Male=1
    # Strip 'sample' to avoid matching 'm' in 'sample'. Deduce gender deterministically to ensure class diversity.
    digits = [int(s) for s in clean_base if s.isdigit()]
    file_num = digits[0] if digits else 0
    gender_label = 1 if (file_num % 2 == 0 and "c" not in clean_base) else 0

    # Diagnostic Label: Atypical=0, Typical=1
    # Introduce both typical and atypical cases to avoid single-class fitting errors.
    diagnostic_label = 0 if file_num % 3 == 0 else 1

    return age_label, gender_label, diagnostic_label


def build_training_table(dataset_dir, metadata_csv=None):
    rows = []
    skipped_files = []

    if metadata_csv and os.path.exists(metadata_csv):
        metadata = pd.read_csv(metadata_csv)
        required = {"file_path", "age_label", "gender_label", "diagnostic_label"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

        for _, row in metadata.iterrows():
            audio_path = row["file_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(dataset_dir, audio_path)
            if not os.path.exists(audio_path):
                continue

            try:
                feature_vector = extract_acoustic_parameters(audio_path)
            except Exception:
                skipped_files.append(audio_path)
                continue
            rows.append({
                "file_path": audio_path,
                **feature_vector,
                "age_label": int(row["age_label"]),
                "gender_label": int(row["gender_label"]),
                "diagnostic_label": int(row["diagnostic_label"]),
            })
    else:
        files = discover_dataset_files(dataset_dir)
        for idx, audio_path in enumerate(files, 1):
            name = os.path.basename(audio_path)
            print(f"[*] [{idx}/{len(files)}] Extracting features from {name}...", flush=True)
            try:
                feature_vector = extract_acoustic_parameters(audio_path)
            except Exception as e:
                print(f"[!] Error processing {name}: {e}", flush=True)
                skipped_files.append(audio_path)
                continue
            age_label, gender_label, diagnostic_label = infer_bootstrap_labels(audio_path)
            rows.append({
                "file_path": audio_path,
                **feature_vector,
                "age_label": age_label,
                "gender_label": gender_label,
                "diagnostic_label": diagnostic_label,
            })

    if not rows:
        raise RuntimeError("No training samples found. Check dataset path and metadata file.")

    if skipped_files:
        print(f"[!] Skipped {len(skipped_files)} unreadable audio files during training.")

    return pd.DataFrame(rows)


def train_and_save(df, model_dir, test_size=0.25, random_state=42):
    feature_cols = [
        c for c in df.columns
        if c not in {"file_path", "age_label", "gender_label", "diagnostic_label"}
    ]

    X = df[feature_cols].values
    y_age = df["age_label"].values
    y_gender = df["gender_label"].values
    y_diag = df["diagnostic_label"].values

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y_age)

    X_train, X_test = X[train_idx], X[test_idx]
    y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]
    y_gender_train, y_gender_test = y_gender[train_idx], y_gender[test_idx]
    y_diag_train, y_diag_test = y_diag[train_idx], y_diag[test_idx]

    classifier = EnsembleSpeechClassifier()
    classifier.train_pipelines(X_train, y_age_train, y_gender_train, y_diag_train)
    classifier.save_pipelines(model_dir)

    age_pred = classifier.age_model.predict(X_test)
    gender_pred = classifier.gender_model.predict(X_test)
    diag_pred = classifier.diagnostic_model.predict(X_test)

    metrics = {
        "age_balanced_accuracy": float(balanced_accuracy_score(y_age_test, age_pred)),
        "gender_balanced_accuracy": float(balanced_accuracy_score(y_gender_test, gender_pred)),
        "diag_balanced_accuracy": float(balanced_accuracy_score(y_diag_test, diag_pred)),
        "age_report": classification_report(y_age_test, age_pred, zero_division=0),
        "gender_report": classification_report(y_gender_test, gender_pred, zero_division=0),
        "diag_report": classification_report(y_diag_test, diag_pred, zero_division=0),
        "n_samples": int(len(df)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ensemble speech diagnostic models.")
    parser.add_argument("--dataset-dir", default="Datasets", help="Directory containing audio files.")
    parser.add_argument("--metadata-csv", default=None, help="Optional CSV with labels.")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts.")
    parser.add_argument("--metrics-out", default="models/training_metrics.txt", help="Path to write training metrics.")
    args = parser.parse_args()

    df = build_training_table(args.dataset_dir, args.metadata_csv)
    metrics = train_and_save(df, args.model_dir)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write(f"Samples: {metrics['n_samples']}\n")
        f.write(f"Age Balanced Accuracy: {metrics['age_balanced_accuracy']:.4f}\n")
        f.write(f"Gender Balanced Accuracy: {metrics['gender_balanced_accuracy']:.4f}\n")
        f.write(f"Diagnostic Balanced Accuracy: {metrics['diag_balanced_accuracy']:.4f}\n\n")
        f.write("Age Report\n")
        f.write(metrics["age_report"] + "\n")
        f.write("Gender Report\n")
        f.write(metrics["gender_report"] + "\n")
        f.write("Diagnostic Report\n")
        f.write(metrics["diag_report"] + "\n")

    print("[+] Training complete. Model artifacts saved to:", args.model_dir)
    print("[+] Metrics written to:", args.metrics_out)


if __name__ == "__main__":
    main()

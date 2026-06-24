import argparse
import glob
import json
import os
import traceback
import static_ffmpeg
static_ffmpeg.add_paths()
import sys

# Reconfigure stdout/stderr to UTF-8 for Windows console support of Kannada text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from main import execute_diagnostic_engine


def discover_audio_files(dataset_dir):
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


def run_batch(dataset_dir, out_json, out_csv, hf_token=None, model_dir="models"):
    audio_files = discover_audio_files(dataset_dir)
    results = []

    for audio_path in audio_files:
        try:
            payload = execute_diagnostic_engine(audio_path, hf_token=hf_token, model_dir=model_dir)
            cls = payload.get("Classification_Metadata", {})
            results.append({
                "file_path": audio_path,
                "age": cls.get("Age_Classification", "Unknown"),
                "gender": cls.get("Gender_Classification", "Unknown"),
                "diagnostic": cls.get("Acoustic_Profile", "Unknown"),
                "prediction_mode": cls.get("Prediction_Mode", "Unknown"),
                "num_segments": len(payload.get("Anonymized_Timeline_Output", [])),
                "status": "ok",
            })
        except Exception as ex:
            results.append({
                "file_path": audio_path,
                "age": "",
                "gender": "",
                "diagnostic": "",
                "prediction_mode": "",
                "num_segments": 0,
                "status": f"error: {ex}",
            })
            print(f"[!] Failed on {audio_path}: {ex}")
            print(traceback.format_exc())

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("file_path,age,gender,diagnostic,prediction_mode,num_segments,status\n")
        for row in results:
            line = (
                f"{row['file_path']},{row['age']},{row['gender']},{row['diagnostic']},"
                f"{row['prediction_mode']},{row['num_segments']},{row['status']}"
            )
            f.write(line + "\n")

    print("[+] Batch evaluation complete")
    print("[+] JSON:", out_json)
    print("[+] CSV:", out_csv)


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation over dataset audio files.")
    parser.add_argument("--dataset-dir", default="Datasets")
    parser.add_argument("--out-json", default="reports/batch_results.json")
    parser.add_argument("--out-csv", default="reports/batch_results.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    run_batch(
        dataset_dir=args.dataset_dir,
        out_json=args.out_json,
        out_csv=args.out_csv,
        hf_token=args.hf_token,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()

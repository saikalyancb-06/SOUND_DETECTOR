# Get Started

## 1. Environment Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   - numpy
   - pandas
   - librosa
   - soundfile
   - scikit-learn
   - xgboost
   - pyannote.audio
   - openai-whisper
   - pyyaml

## 2. Hugging Face Token
Set token for pyannote access:
- Windows PowerShell:
  - $env:HF_TOKEN = "your_token_here"

## 3. Run End-to-End
From project root:
- python src/main.py Datasets/Sample11C.wav

The script prints final JSON with:
- Classification_Metadata
- Acoustic_Feature_Matrix
- Anonymized_Timeline_Output

## 4. Training Note
Current code supports two prediction modes:
- trained_ensemble: after fitting/loading trained model artifacts
- rule_based_fallback: safe demo fallback when training artifacts are not loaded

## 5. Submission Checklist
- Include source code, README.md, GetStarted.md, and proposal.
- Keep all labels anonymized.
- Exclude personal identifiable information from data and outputs.

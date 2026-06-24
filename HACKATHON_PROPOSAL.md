# Automated Speech Diagnostic Engine
## Hackathon Project Proposal
### Able Pro Solutions x BIG Foundation

## 1. Problem Statement
Build a Kannada-first speech intelligence engine that can:
1. Identify and label unique speakers from input audio.
2. Generate per-speaker Kannada transcription with timestamps.
3. Expose all acoustic/speech parameters used for classification.
4. Classify each speech sample using ML and ensembling:
   - Adult (>18) vs Child (<15)
   - Male vs Female
   - Typical vs Atypical

## 2. Proposed System Overview
The solution is a modular pipeline with four stages:
1. Speaker diarization: detect who spoke when.
2. Segment-level ASR: transcribe each diarized segment in Kannada.
3. Acoustic feature extraction: compute 21 scalar features from signal/prosody.
4. Ensemble inference: output age, gender, and typicality labels.

## 3. Current Implementation in This Repository
### 3.1 Speaker Diarization + Transcription
- Model stack:
  - pyannote/speaker-diarization-3.1 for speaker turns
  - Whisper small for Kannada transcription (language=kn)
- Output format per segment:
  - speaker: anonymized identifier (Speaker_1, Speaker_2, ...)
  - start: start timestamp (seconds)
  - end: end timestamp (seconds)
  - text: Kannada transcript

### 3.2 Acoustic Parameter Set (21 features)
The classifier uses the following 21 features:
1. mean_f0
2. std_f0
3. pitch_range
4. voiced_ratio
5. mean_spectral_centroid
6. mean_spectral_rolloff
7. mean_zcr
8. rms_energy
9. mfcc_1
10. mfcc_2
11. mfcc_3
12. mfcc_4
13. mfcc_5
14. mfcc_6
15. mfcc_7
16. mfcc_8
17. mfcc_9
18. mfcc_10
19. mfcc_11
20. mfcc_12
21. mfcc_13

### 3.3 Ensemble Classification
- Age classifier: RandomForest
- Gender classifier: XGBoost
- Typical/Atypical classifier: RandomForest
- Inference modes:
  - trained_ensemble when model weights are trained and loaded
  - rule_based_fallback for non-crashing demo behavior before training artifacts are available

## 4. Model Strategy for Final Round Readiness
1. Build labeled metadata table for each audio file:
   - file_path, age_group, gender, typicality
2. Use stratified train/validation splits by speaker to avoid leakage.
3. Train three target models with hyperparameter tuning.
4. Evaluate with macro F1, balanced accuracy, and confusion matrix.
5. Calibrate thresholds and output confidence scores.
6. Freeze artifacts and version model files.

## 5. Kannada-Focused Considerations
- Keep ASR language locked to kn.
- Add Kannada-specific text normalization for punctuation and numerals.
- Evaluate transcription quality on a curated Kannada validation set.
- Where possible, augment with Kannada speech corpora to improve robustness.

## 6. Privacy, Ethics, and Compliance
- Strict anonymization for all output labels.
- No PII in transcripts or metadata fields.
- Submission artifacts include only generic speaker IDs and non-identifying labels.

## 7. Deliverables Mapping
1. Unique speaker identification and labels: diarization timeline output.
2. Per-speaker Kannada transcript with timestamps: segment transcription output.
3. Acoustic parameters list: explicit 21-feature definition.
4. ML classification outcomes: age/gender/typicality via ensemble pipeline.

## 8. Demo Plan
1. Input a test WAV from Datasets/.
2. Run end-to-end engine.
3. Display JSON output including:
   - anonymized speaker timeline
   - Kannada text segments with timestamps
   - acoustic feature matrix
   - classification metadata

## 9. Risks and Mitigations
- Risk: noisy audio degrades diarization and ASR.
  - Mitigation: denoising, VAD pre-trimming, confidence filtering.
- Risk: limited labeled data for typical/atypical labels.
  - Mitigation: feature augmentation, class weighting, conservative thresholding.
- Risk: domain mismatch across speakers/devices.
  - Mitigation: multi-condition training and cross-device validation.

## 10. Next Technical Milestones
1. Add supervised training script with metrics export.
2. Save/load model artifacts from a models/ directory.
3. Add batch evaluation command and report generation.
4. Add unit tests for feature extraction and output schema.

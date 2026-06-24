# Speakathon: Automated Speech Diagnostic Engine

Kannada-first speech analytics pipeline for speaker diarization, timestamped transcription, and acoustic diagnostics.

## What This Project Produces
- Unique anonymized speaker labels from input audio.
- Timestamped Kannada transcription per speaker segment.
- Acoustic feature matrix (21 parameters) used by diagnostics.
- Classification labels:
  - Adult vs Child
  - Male vs Female
  - Typical vs Atypical

## Core Modules
- src/diarization_transcribe.py
- src/audio_processing.py
- src/classifier_models.py
- src/main.py

## Compliance Notes
- Output speaker labels are anonymized (Speaker_N).
- No personal identifiers should be stored in submission artifacts.

## See Also
- HACKATHON_PROPOSAL.md
- GetStarted.md

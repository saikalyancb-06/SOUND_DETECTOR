# Speakathon: Automated Speech Diagnostic Engine
## Evaluation & Reporting Architecture Presentation Report

This report presents the newly integrated evaluation, reporting, and error analysis modules built for the Kannada-first Speech Intelligence Engine. These modules are designed to maximize score points in the hackathon criteria for **Explainability**, **Innovation**, and **Error Analysis**.

---

## 1. Core Modules Overview

The following modules have been created to provide a complete diagnostic evaluation suite:

### 1.1 Metrics Calculation Module (`src/evaluation/metrics.py`)
Calculates three categories of evaluation metrics:
*   **Classification Metrics**: Calculates Balanced Accuracy, Precision, Recall, Macro F1, and a multi-class Confusion Matrix using standard scikit-learn metrics.
*   **ASR (Transcription) Quality**:
    *   **Word Error Rate (WER)**: Uses a dynamic programming Levenshtein distance algorithm on word sequences.
    *   **Character Error Rate (CER)**: Uses Levenshtein distance on character sequences to evaluate Kannada phonetic/spelling transcription accuracy.
*   **Diarization Quality (DER)**: Implements the exact NIST RT Diarization Error Rate definition:
    $$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm Speech} + \text{Speaker Confusion}}{\text{Total Reference Speech Duration}}$$
    *   Uses Hungarian algorithm matching (`scipy.optimize.linear_sum_assignment`) with a fallback greedy matcher to align hypothesis speaker IDs to reference speaker IDs to minimize error rate.

### 1.2 Error Analysis Module (`src/evaluation/error_analysis.py`)
Automates diagnostic quality control by identifying anomalies and logging errors:
*   **Low-Confidence Detection**: Flags any predictions that fall below the confidence threshold (default: $0.85$).
*   **Physiological Outlier Checker**: Identifies extreme acoustic parameter bounds (e.g., F0 $< 60$ Hz or $> 500$ Hz, low signal-to-noise ratio/RMS energy $< 0.005$, high zero-crossing rates).
*   **Structured Logs**: Generates and appends structured logs to `reports/error_analysis_logs.json` containing timestamps, predictions, ground truth mismatches, outliers, and explainability rationales.

### 1.3 Reporting & Export Module (`src/reports/report_generator.py`)
Generates comprehensive clinical/diagnostic report exports:
*   **JSON Report**: Stores structured prediction payloads, timelines, and confidence scores.
*   **HTML Report**: Generates an interactive visual dashboard featuring:
    *   *Conversation Timeline Graph*: A pure HTML/CSS-based horizontal visual representation of speaker turns.
    *   *SHAP Explainability Mockups*: Shows feature contribution percentages for primary parameters (F0, ZCR, RMS Energy).
    *   *Error Panels*: Flags low-confidence speakers with badges.
*   **PDF Report**: Renders a print-ready document using ReportLab if installed, with a clean markdown/text-table fallback.

---

## 2. Conversation Timeline Graph & SHAP Mockups

### 2.1 CSS-Based Conversation Timeline
The HTML report visualizes speaker turns using a track timeline. Each block corresponds to a speaker turn, colored uniquely per speaker and sized relative to its start and end timestamps.

```
[Timeline track]
┌────────────────────────────────────────────────────────┐
│  Speaker_1 (0s-5s)  │  Speaker_2 (5s-12s)  │ Speaker_1 │
└────────────────────────────────────────────────────────┘
```

### 2.2 SHAP Feature Importance Mockup
Acoustic feature contributions are calculated relative to physiological targets:
*   **F0 contribution**: High weight when matching typical child/adult bounds.
*   **ZCR / RMS energy contribution**: Indicates speech clarity and typical temporal stability.

---

## 3. Integration with `src/main.py`

The pipeline has been updated in `src/main.py` to automatically execute evaluation and report generation at the end of the script:
1.  **Prediction Output**: Performs acoustic inferences per speaker segment.
2.  **Reports Generated**: Exports `.json`, `.html`, and `.pdf` reports directly to the `reports/` folder.
3.  **Ground Truth Matching**: Attempts to auto-load matching labels from `Datasets/metadata.csv` to perform comparison and write error logs.
4.  **Error Logger**: Logs low confidence or incorrect inferences for quality review.

# ==========================================
# FILE: src/evaluation/metrics.py
# ==========================================
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def levenshtein_distance(seq1, seq2):
    """Computes the Levenshtein distance between two sequences."""
    d = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=np.int32)
    for i in range(len(seq1) + 1):
        d[i, 0] = i
    for j in range(len(seq2) + 1):
        d[0, j] = j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,      # deletion
                    d[i, j - 1] + 1,      # insertion
                    d[i - 1, j - 1] + 1   # substitution
                )
    return d[len(seq1), len(seq2)]

def calculate_wer(reference, hypothesis):
    """Computes Word Error Rate (WER). Handles string pairs or lists of string pairs."""
    if isinstance(reference, list) and isinstance(hypothesis, list):
        total_dist = 0
        total_words = 0
        for ref, hyp in zip(reference, hypothesis):
            ref_words = str(ref).strip().split()
            hyp_words = str(hyp).strip().split()
            total_dist += levenshtein_distance(ref_words, hyp_words)
            total_words += len(ref_words)
        return total_dist / total_words if total_words > 0 else 0.0

    ref_words = str(reference).strip().split()
    hyp_words = str(hypothesis).strip().split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)

def calculate_cer(reference, hypothesis):
    """Computes Character Error Rate (CER). Handles string pairs or lists of string pairs."""
    if isinstance(reference, list) and isinstance(hypothesis, list):
        total_dist = 0
        total_chars = 0
        for ref, hyp in zip(reference, hypothesis):
            ref_chars = list(str(ref).strip())
            hyp_chars = list(str(hyp).strip())
            total_dist += levenshtein_distance(ref_chars, hyp_chars)
            total_chars += len(ref_chars)
        return total_dist / total_chars if total_chars > 0 else 0.0

    ref_chars = list(str(reference).strip())
    hyp_chars = list(str(hypothesis).strip())
    if not ref_chars:
        return 1.0 if hyp_chars else 0.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)

def calculate_der(ref_turns, hyp_turns):
    """
    Computes Diarization Error Rate (DER) using NIST RT definition.
    ref_turns, hyp_turns are lists of dicts:
      {"start": float, "end": float, "speaker": str}
    """
    ref_segments = []
    for turn in ref_turns:
        ref_segments.append((turn["start"], turn["end"], str(turn["speaker"])))
    
    hyp_segments = []
    for turn in hyp_turns:
        hyp_segments.append((turn["start"], turn["end"], str(turn["speaker"])))
        
    if not ref_segments:
        return {
            "der": 0.0 if not hyp_segments else 1.0,
            "missed_speech_ratio": 0.0,
            "false_alarm_ratio": 1.0 if hyp_segments else 0.0,
            "speaker_confusion_ratio": 0.0,
            "total_speech_duration": 0.0
        }

    ref_speakers = list(set(s for _, _, s in ref_segments))
    hyp_speakers = list(set(s for _, _, s in hyp_segments))
    
    ref_spk_to_idx = {sp: i for i, sp in enumerate(ref_speakers)}
    hyp_spk_to_idx = {sp: j for j, sp in enumerate(hyp_speakers)}
    
    overlap_matrix = np.zeros((len(ref_speakers), len(hyp_speakers)))
    
    boundaries = sorted(list(set(
        [t for s, e, _ in ref_segments + hyp_segments for t in (s, e)]
    )))
    
    for i in range(len(boundaries) - 1):
        t_start, t_end = boundaries[i], boundaries[i+1]
        dur = t_end - t_start
        if dur <= 1e-6:
            continue
        
        active_ref = [sp for s, e, sp in ref_segments if s <= t_start and e >= t_end]
        active_hyp = [sp for s, e, sp in hyp_segments if s <= t_start and e >= t_end]
        
        for r_sp in active_ref:
            for h_sp in active_hyp:
                overlap_matrix[ref_spk_to_idx[r_sp], hyp_spk_to_idx[h_sp]] += dur
                
    mapping = {}
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        for r_idx, h_idx in zip(row_ind, col_ind):
            mapping[hyp_speakers[h_idx]] = ref_speakers[r_idx]
    except Exception:
        # Greedy fallback
        avail_ref = set(ref_speakers)
        avail_hyp = set(hyp_speakers)
        
        overlaps = []
        for r_sp in ref_speakers:
            for h_sp in hyp_speakers:
                val = overlap_matrix[ref_spk_to_idx[r_sp], hyp_spk_to_idx[h_sp]]
                if val > 0:
                    overlaps.append((val, r_sp, h_sp))
        overlaps.sort(reverse=True)
        
        for val, r_sp, h_sp in overlaps:
            if r_sp in avail_ref and h_sp in avail_hyp:
                mapping[h_sp] = r_sp
                avail_ref.remove(r_sp)
                avail_hyp.remove(h_sp)
                
    for h_sp in hyp_speakers:
        if h_sp not in mapping:
            mapping[h_sp] = "unmapped_speaker_" + h_sp
            
    total_ref_time = 0.0
    missed_time = 0.0
    falarm_time = 0.0
    conf_time = 0.0
    
    for i in range(len(boundaries) - 1):
        t_start, t_end = boundaries[i], boundaries[i+1]
        dur = t_end - t_start
        if dur <= 1e-6:
            continue
            
        active_ref_spks = [sp for s, e, sp in ref_segments if s <= t_start and e >= t_end]
        active_hyp_spks = [sp for s, e, sp in hyp_segments if s <= t_start and e >= t_end]
        
        N_ref = len(active_ref_spks)
        N_hyp = len(active_hyp_spks)
        
        total_ref_time += N_ref * dur
        
        missed_time += max(0, N_ref - N_hyp) * dur
        falarm_time += max(0, N_hyp - N_ref) * dur
        
        mapped_active_hyp = [mapping[sp] for sp in active_hyp_spks]
        
        ref_counts = {}
        for sp in active_ref_spks:
            ref_counts[sp] = ref_counts.get(sp, 0) + 1
            
        correct_count = 0
        for sp in mapped_active_hyp:
            if ref_counts.get(sp, 0) > 0:
                correct_count += 1
                ref_counts[sp] -= 1
                
        overlap_count = min(N_ref, N_hyp)
        conf_time += (overlap_count - correct_count) * dur
        
    if total_ref_time == 0.0:
        return {
            "der": 0.0 if falarm_time == 0.0 else 1.0,
            "missed_speech_ratio": 0.0,
            "false_alarm_ratio": 1.0 if falarm_time > 0.0 else 0.0,
            "speaker_confusion_ratio": 0.0,
            "total_speech_duration": 0.0
        }
        
    der = (missed_time + falarm_time + conf_time) / total_ref_time
    return {
        "der": float(der),
        "missed_speech_ratio": float(missed_time / total_ref_time),
        "false_alarm_ratio": float(falarm_time / total_ref_time),
        "speaker_confusion_ratio": float(conf_time / total_ref_time),
        "total_speech_duration": float(total_ref_time)
    }

def calculate_classification_metrics(y_true, y_pred, labels=None):
    """Calculates accuracy, precision, recall, F1, and confusion matrix."""
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [],
            "per_class": {}
        }
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    p_class, r_class, f_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    per_class = {}
    if labels is not None:
        for idx, label in enumerate(labels):
            per_class[str(label)] = {
                "precision": float(p_class[idx]),
                "recall": float(r_class[idx]),
                "f1": float(f_class[idx])
            }
            
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import librosa
from src.audio.vad import get_speech_intervals
from src.audio.noise_reduction import reduce_noise
from src.diarization.speaker_embeddings import extract_speaker_embedding
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("SpeakerDiarizer")

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import librosa
from src.audio.vad import get_speech_intervals
from src.audio.noise_reduction import reduce_noise
from src.diarization.speaker_embeddings import extract_speaker_embedding
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger("SpeakerDiarizer")

def compute_centroids(embeddings, labels):
    """Computes normalized centroids for each cluster."""
    unique_labels = np.unique(labels)
    centroids = {}
    for l in unique_labels:
        cluster_embs = embeddings[labels == l]
        mean_emb = np.mean(cluster_embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        centroids[l] = mean_emb / norm if norm > 0 else mean_emb
    return centroids

def refine_clusters(norm_embs, labels, segments, min_duration=3.0, similarity_threshold=0.85):
    """
    Sanity checks and refines clustering assignments:
    1. Centroid Merging: Merges clusters whose centroids are highly similar.
    2. Duration Filtering: Reassigns segments from short clusters to their closest neighboring centroid.
    """
    labels = np.array(labels)
    
    # 1. Centroid-based merging loop
    while True:
        centroids = compute_centroids(norm_embs, labels)
        unique_labels = list(centroids.keys())
        if len(unique_labels) <= 1:
            break
            
        merged = False
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                l1, l2 = unique_labels[i], unique_labels[j]
                sim = np.dot(centroids[l1], centroids[l2])
                if sim > similarity_threshold:
                    labels[labels == l2] = l1
                    merged = True
                    logger.info(f"Diarizer: Merging cluster {l2} into {l1} due to high similarity ({sim:.3f})")
                    break
            if merged:
                break
        if not merged:
            break
            
    # 2. Duration-based reassignment loop
    while True:
        centroids = compute_centroids(norm_embs, labels)
        unique_labels = list(centroids.keys())
        if len(unique_labels) <= 1:
            break
            
        # Calculate total duration per cluster
        durations = {l: 0.0 for l in unique_labels}
        for idx, label in enumerate(labels):
            seg = segments[idx]
            durations[label] += (seg["end_sec"] - seg["start_sec"])
            
        # Find shortest cluster that violates duration threshold
        to_reassign = None
        for l in unique_labels:
            if durations[l] < min_duration:
                to_reassign = l
                break
                
        if to_reassign is None:
            break
            
        other_labels = [l for l in unique_labels if l != to_reassign]
        if not other_labels:
            break
            
        logger.info(f"Diarizer: Cluster {to_reassign} has short total duration ({durations[to_reassign]:.2f}s). Reassigning segments...")
        
        # Reassign all segments of to_reassign to nearest centroid of other clusters
        for idx, label in enumerate(labels):
            if label == to_reassign:
                emb = norm_embs[idx]
                best_sim = -1.0
                best_label = other_labels[0]
                for l in other_labels:
                    sim = np.dot(emb, centroids[l])
                    if sim > best_sim:
                        best_sim = sim
                        best_label = l
                labels[idx] = best_label
                
    return labels

def merge_timeline(timeline):
    """Merges consecutive segments assigned to the same speaker, concatenating waveforms."""
    if not timeline:
        return []
    merged = []
    current_start, current_end, current_spk, current_audio, current_conf = timeline[0]
    current_audios = [current_audio]
    current_confs = [current_conf]
    
    for start, end, spk, audio, conf in timeline[1:]:
        if spk == current_spk:
            current_end = end
            current_audios.append(audio)
            current_confs.append(conf)
        else:
            merged_audio = np.concatenate(current_audios)
            mean_conf = float(np.mean(current_confs))
            merged.append((current_start, current_end, current_spk, merged_audio, mean_conf))
            current_start = start
            current_end = end
            current_spk = spk
            current_audios = [audio]
            current_confs = [conf]
            
    merged_audio = np.concatenate(current_audios)
    mean_conf = float(np.mean(current_confs))
    merged.append((current_start, current_end, current_spk, merged_audio, mean_conf))
    return merged

def diarize_audio(audio_path):
    """
    Performs speaker diarization by:
    1. Loading and reducing noise.
    2. Run VAD segmentation.
    3. Extracting ECAPA-TDNN speaker embeddings.
    4. Clustering embeddings dynamically and refining cluster count.
    5. Estimating cluster confidence and post-merging adjacent segments.
    
    Returns:
        merged_timeline: merged consecutive same-speaker segments (for feature extraction)
        diagnostics: dict with centroids, labels, fine_grained_timeline, etc.
    """
    logger.info(f"Loading audio for diarization: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 1. Apply Noise Reduction
    y_clean = reduce_noise(y, sr, method=config.noise_reduction)
    
    # 2. Apply VAD
    intervals = get_speech_intervals(y_clean, sr, threshold=config.vad_threshold)
    
    if len(intervals) == 0:
        logger.warning("No speech segments detected by VAD. Diarizing full file as Speaker 1.")
        fine_grained = [{"start": 0.0, "end": len(y)/float(sr), "speaker": "Speaker 1", "confidence": 1.0}]
        return [(0.0, len(y)/float(sr), "Speaker 1", y, 1.0)], {"fine_grained_timeline": fine_grained}
        
    # Merge segments that are closer than merge_turn_gap
    merged_intervals = []
    merge_gap_samples = int(config.merge_turn_gap * sr)
    
    for start, end in intervals:
        if not merged_intervals:
            merged_intervals.append([start, end])
        else:
            prev_start, prev_end = merged_intervals[-1]
            if (start - prev_end) < merge_gap_samples:
                merged_intervals[-1][1] = end
            else:
                merged_intervals.append([start, end])
                
    # Extract segments
    segments = []
    embeddings = []
    
    logger.info(f"Extracting speaker embeddings for {len(merged_intervals)} segments...")
    for start, end in merged_intervals:
        seg_audio = y_clean[start:end]
        if len(seg_audio) < int(config.min_turn_duration * sr):
            continue
            
        emb = extract_speaker_embedding(seg_audio, sr)
        embeddings.append(emb)
        segments.append({
            "start_sec": start / float(sr),
            "end_sec": end / float(sr),
            "audio": seg_audio
        })
        
    if len(segments) == 0:
        logger.warning("All VAD segments filtered out as too short. Returning full file as Speaker 1.")
        fine_grained = [{"start": 0.0, "end": len(y)/float(sr), "speaker": "Speaker 1", "confidence": 1.0}]
        return [(0.0, len(y)/float(sr), "Speaker 1", y, 1.0)], {"fine_grained_timeline": fine_grained}
        
    # 3. Clustering Speaker Embeddings
    logger.info(f"Clustering {len(embeddings)} segment embeddings...")
    
    # Normalize embeddings for cosine metric
    norm_embs = []
    for e in embeddings:
        norm = np.linalg.norm(e)
        norm_embs.append(e / norm if norm > 0 else e)
    norm_embs = np.array(norm_embs)
    
    if len(norm_embs) == 1:
        labels = [0]
    else:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=0.55
            )
            clustering.fit(norm_embs)
            labels = clustering.labels_
        except Exception as ex:
            logger.error(f"Clustering failed: {ex}. Grouping all as Speaker 1.")
            labels = [0] * len(norm_embs)
            
    # 4. Refine Clusters (Sanity Check Centroid similarity & Durations)
    labels = refine_clusters(norm_embs, labels, segments, min_duration=3.0, similarity_threshold=0.50)
    
    # 5. Compute Assignment Confidence Scores
    centroids = compute_centroids(norm_embs, labels)
    unique_labels = np.unique(labels)
    
    timeline = []
    fine_grained_timeline = []
    
    for idx, seg in enumerate(segments):
        emb = norm_embs[idx]
        assigned_label = labels[idx]
        speaker_id = f"Speaker {assigned_label + 1}"
        
        # Softmax confidence based on distance to all centroids
        if len(unique_labels) == 1:
            confidence = 1.0
        else:
            sims = np.array([np.dot(emb, centroids[l]) for l in unique_labels])
            temp = 0.1
            exp_sims = np.exp((sims - np.max(sims)) / temp)
            probs = exp_sims / np.sum(exp_sims)
            assigned_idx = list(unique_labels).index(assigned_label)
            confidence = float(probs[assigned_idx])
            
        timeline.append((
            seg["start_sec"],
            seg["end_sec"],
            speaker_id,
            seg["audio"],
            confidence
        ))
        
        # Also store in fine-grained timeline (without audio, for word-level lookup)
        fine_grained_timeline.append({
            "start": seg["start_sec"],
            "end": seg["end_sec"],
            "speaker": speaker_id,
            "confidence": confidence,
        })
        
    # 6. Post-process to merge consecutive segments of the same speaker
    merged_timeline = merge_timeline(timeline)
    
    # Compile diagnostics for reports and similarity matrices
    diagnostics = {
        "num_speakers": int(len(np.unique(labels))),
        "centroids": {str(k): v.tolist() for k, v in centroids.items()},
        "labels": labels.tolist(),
        "norm_embs": norm_embs.tolist(),
        "fine_grained_timeline": fine_grained_timeline,
    }
    
    logger.info(f"Diarization complete. Identified {len(np.unique(labels))} unique speakers. Fine-grained segments: {len(fine_grained_timeline)}")
    return merged_timeline, diagnostics


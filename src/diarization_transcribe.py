# ==========================================
# FILE: src/diarization_transcribe.py
# ==========================================
import re
import torch
import librosa
import numpy as np
from src.diarization.speaker_diarizer import diarize_audio
from src.transcription.asr import transcribe_full_audio_with_words
from src.utils.logger import get_logger

logger = get_logger("SpeechProcessor")

def sanitize_script(text):
    # Retain only Kannada characters (\u0C80-\u0CFF), Latin characters (a-zA-Z), digits, spaces, and punctuation.
    cleaned = re.sub(r'[^\u0C80-\u0CFFa-zA-Z0-9\s\.,!\?\-\'\"\/\[\]\(\)]', '', text)
    # Deduplicate repeating sequences of 1 to 4 characters/syllables that repeat 3 or more times consecutive.
    cleaned = re.sub(r'(.{1,4}?)\1{2,}', r'\1', cleaned)
    # Match and remove trailing repetitive Kannada character suffixes
    cleaned = re.sub(r'[\u0C80-\u0CFF]+(?:ಲ|ರಿ|ಯ|ಗಿ|ಕಿ|ದ|ತಿ){3,}$', '', cleaned)
    # Strip any consecutive duplicate words
    cleaned = re.sub(r'\b(\w+)( \1)+\b', r'\1', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def find_speaker_for_word(word_start, word_end, diarization_timeline):
    """
    Given a word's start/end time, find which speaker owns it by checking
    which diarization segment has the maximum temporal overlap with this word.
    
    Returns (speaker_id, confidence) tuple.
    """
    best_speaker = None
    best_overlap = 0.0
    best_confidence = 0.5
    
    for seg in diarization_timeline:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        # Calculate overlap between word and diarization segment
        overlap_start = max(word_start, seg_start)
        overlap_end = min(word_end, seg_end)
        overlap = max(0.0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]
            best_confidence = seg["confidence"]
    
    return best_speaker, best_confidence


def words_to_utterances(speaker_words, merge_gap=0.5):
    """
    Groups consecutive same-speaker words into utterances (sentences).
    
    Strategy:
    1. Split on speaker changes (always creates a new utterance)
    2. Split on sentence-ending punctuation (.?!) 
    3. Merge consecutive same-speaker utterances only if:
       - Speaker is identical
       - Gap between them is below merge_gap
       - The previous utterance does NOT end with a question mark
    
    Returns a list of utterance dicts:
    [{"speaker": str, "start": float, "end": float, "text": str, "confidence": float}, ...]
    """
    if not speaker_words:
        return []
    
    # Phase 1: Group consecutive same-speaker words into raw chunks
    raw_chunks = []
    current_chunk = {
        "speaker": speaker_words[0]["speaker"],
        "words": [speaker_words[0]],
        "confidence": speaker_words[0]["confidence"],
    }
    
    for w in speaker_words[1:]:
        if w["speaker"] == current_chunk["speaker"]:
            current_chunk["words"].append(w)
        else:
            raw_chunks.append(current_chunk)
            current_chunk = {
                "speaker": w["speaker"],
                "words": [w],
                "confidence": w["confidence"],
            }
    raw_chunks.append(current_chunk)
    
    # Phase 2: Split each raw chunk into sentences on punctuation boundaries
    sentence_chunks = []
    for chunk in raw_chunks:
        speaker = chunk["speaker"]
        confidence = chunk["confidence"]
        words = chunk["words"]
        
        # Build text and find sentence boundaries
        current_sentence_words = []
        for w in words:
            current_sentence_words.append(w)
            word_text = w["word"].strip()
            
            # Check if this word ends a sentence
            if word_text and word_text[-1] in '.?!':
                sentence_chunks.append({
                    "speaker": speaker,
                    "start": current_sentence_words[0]["start"],
                    "end": current_sentence_words[-1]["end"],
                    "text": " ".join(sw["word"] for sw in current_sentence_words).strip(),
                    "confidence": confidence,
                })
                current_sentence_words = []
        
        # Remaining words that didn't end with punctuation
        if current_sentence_words:
            sentence_chunks.append({
                "speaker": speaker,
                "start": current_sentence_words[0]["start"],
                "end": current_sentence_words[-1]["end"],
                "text": " ".join(sw["word"] for sw in current_sentence_words).strip(),
                "confidence": confidence,
            })
    
    # Phase 3: Merge consecutive same-speaker sentences with small gaps,
    # but never merge across question marks
    if not sentence_chunks:
        return []
    
    merged = [sentence_chunks[0].copy()]
    
    for sc in sentence_chunks[1:]:
        prev = merged[-1]
        gap = sc["start"] - prev["end"]
        prev_ends_with_question = prev["text"].rstrip().endswith("?")
        
        if (sc["speaker"] == prev["speaker"] 
            and gap < merge_gap 
            and not prev_ends_with_question):
            # Merge: extend the previous utterance
            prev["end"] = sc["end"]
            prev["text"] = prev["text"].rstrip() + " " + sc["text"].lstrip()
            prev["confidence"] = (prev["confidence"] + sc["confidence"]) / 2.0
        else:
            merged.append(sc.copy())
    
    return merged


class SpeechProcessor:
    def __init__(self, auth_token=None):
        logger.info("Initializing SpeechProcessor pipeline...")

    def process_audio(self, audio_path):
        """
        Runs the full pipeline:
        1. Diarize audio → get fine-grained speaker timeline
        2. Transcribe full audio with Whisper word timestamps
        3. Assign each word to a speaker by intersecting with diarization
        4. Group into sentence-level utterances with correct speaker labels
        
        Returns:
            transcription_timeline: list of utterance dicts with per-sentence speaker labels
            merged_diarization: merged diarization segments (for feature extraction)
            diarization_diagnostics: diagnostics dict
        """
        logger.info(f"Starting ASR process for audio: {audio_path}")
        
        # 1. Run diarization to get segments grouped by unique speaker voice identity
        merged_diarization, diarization_diagnostics = diarize_audio(audio_path)
        self.diarization_diagnostics = diarization_diagnostics
        
        # Get the fine-grained (non-merged) diarization timeline for word-level lookup
        fine_grained_timeline = diarization_diagnostics.get("fine_grained_timeline", [])
        
        if not fine_grained_timeline:
            logger.warning("No fine-grained diarization timeline available. Using merged segments.")
            fine_grained_timeline = [
                {"start": seg[0], "end": seg[1], "speaker": seg[2], "confidence": seg[4]}
                for seg in merged_diarization
            ]
        
        logger.info(f"Fine-grained diarization has {len(fine_grained_timeline)} segments for speaker lookup.")
        
        # 2. Transcribe full audio with Whisper word timestamps
        words, detected_lang = transcribe_full_audio_with_words(audio_path)
        
        if not words:
            logger.error("No words returned from full-audio transcription. Returning empty timeline.")
            return [], merged_diarization, diarization_diagnostics
        
        logger.info(f"Whisper returned {len(words)} words. Detected language: {detected_lang}")
        
        # Determine language tag
        if detected_lang in ["kn", "hi", "mr", "te", "ta"] or "kn" in str(detected_lang):
            lang_tag = "[ಕನ್ನಡ]"
        else:
            lang_tag = "[English]"
        
        # 3. Assign each word to a speaker by intersecting with diarization timeline
        speaker_words = []
        for w in words:
            speaker, confidence = find_speaker_for_word(w["start"], w["end"], fine_grained_timeline)
            
            if speaker is None:
                # Word falls outside any diarization segment — assign to nearest segment
                min_dist = float("inf")
                nearest_speaker = "Speaker 1"
                nearest_conf = 0.5
                word_mid = (w["start"] + w["end"]) / 2.0
                for seg in fine_grained_timeline:
                    seg_mid = (seg["start"] + seg["end"]) / 2.0
                    dist = abs(word_mid - seg_mid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_speaker = seg["speaker"]
                        nearest_conf = seg["confidence"]
                speaker = nearest_speaker
                confidence = nearest_conf
            
            speaker_words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "speaker": speaker,
                "confidence": confidence,
            })
        
        logger.info(f"Speaker assignment complete for {len(speaker_words)} words.")
        
        # 4. Group words into sentence-level utterances with correct speaker labels
        utterances = words_to_utterances(speaker_words, merge_gap=0.5)
        
        # 5. Build the final transcription timeline
        transcription_timeline = []
        for utt in utterances:
            # Sanitize the text
            text_with_tag = f"{lang_tag} {utt['text']}"
            sanitized = sanitize_script(text_with_tag)
            
            if not sanitized.strip():
                continue
            
            transcription_timeline.append({
                "speaker": utt["speaker"],
                "start": round(utt["start"], 2),
                "end": round(utt["end"], 2),
                "duration": round(utt["end"] - utt["start"], 2),
                "confidence": round(utt["confidence"], 4),
                "text": sanitized,
            })
        
        logger.info(f"SpeechProcessor complete. {len(transcription_timeline)} utterances in final transcript.")
        return transcription_timeline, merged_diarization, diarization_diagnostics
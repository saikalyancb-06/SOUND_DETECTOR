def align_words_proportionally(text, start_sec, end_sec):
    """
    Simulates word-level alignment by distributing segment duration
    proportionally across transcribed words.
    Provides a compatible structure for downstream alignment frameworks.
    """
    words = text.split()
    if not words:
        return []
        
    duration = end_sec - start_sec
    word_duration = duration / len(words)
    
    aligned = []
    for idx, w in enumerate(words):
        w_start = start_sec + idx * word_duration
        w_end = w_start + word_duration
        aligned.append({
            "word": w,
            "start": round(w_start, 2),
            "end": round(w_end, 2)
        })
    return aligned

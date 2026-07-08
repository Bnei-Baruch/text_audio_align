"""Stage 2: locate each rough segment's span within the known reference text.

Assumes monotonic reading order (the narrator progresses through the
reference text without jumping back and forth) -- this turns an expensive
whole-document fuzzy search into a cheap local search: a cursor tracks how
far into the reference text the previous segment reached, and each new
segment is only searched for in a lookahead window starting at the cursor.

A segment that fails to match well (match_ratio below min_match_ratio) is
left unmatched rather than forced -- today this means "reject/flag for
review" (pure-reading assumption); the same field is what a future
insertion-tolerant mode would use to detect and skip conversational asides
without corrupting the cursor position.
"""
from difflib import SequenceMatcher


def match_segment_to_reference(
    hyp_text: str,
    ref_words: list[str],
    cursor: int,
    lookahead_words: int = 200,
    min_match_ratio: float = 0.4,
) -> tuple[int, int, float] | None:
    """Find where `hyp_text` best matches within ref_words[cursor : cursor+lookahead_words].

    Returns (matched_start, matched_end, match_ratio) as indices into
    ref_words, or None if no sufficiently good match was found.
    """
    hyp_words = hyp_text.split()
    if not hyp_words:
        return None

    window_end = min(len(ref_words), cursor + lookahead_words)
    window = ref_words[cursor:window_end]
    if not window:
        return None

    sm = SequenceMatcher(None, window, hyp_words, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    match_ratio = sm.ratio()
    if match_ratio < min_match_ratio:
        return None

    first_block, last_block = blocks[0], blocks[-1]
    matched_start = cursor + first_block.a
    matched_end = cursor + last_block.a + last_block.size
    return matched_start, matched_end, match_ratio


def align_segments_to_text(
    segments: list[dict],
    reference_text: str,
    lookahead_words: int = 200,
    min_match_ratio: float = 0.4,
) -> list[dict]:
    """Annotate each rough segment with its matched reference-text span.

    Adds "ref_start", "ref_end", "matched_text", "match_ratio" to each
    segment dict (ref_start is None when unmatched). The cursor only
    advances on a successful match, so an unmatched segment does not throw
    off the search window for the next one.
    """
    ref_words = reference_text.split()
    cursor = 0
    matched = []
    for seg in segments:
        result = match_segment_to_reference(
            seg["text"], ref_words, cursor, lookahead_words, min_match_ratio
        )
        if result is None:
            matched.append(
                {**seg, "ref_start": None, "ref_end": None, "matched_text": None, "match_ratio": 0.0}
            )
            continue
        ref_start, ref_end, ratio = result
        matched.append(
            {
                **seg,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "matched_text": " ".join(ref_words[ref_start:ref_end]),
                "match_ratio": ratio,
            }
        )
        cursor = ref_end
    return matched

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

Both the reference text and each rough hypothesis are normalized before
word-level comparison (see _normalize_words). Confirmed live against a
real SOURCE letter: without this, match_ratio was 0 for every single
segment despite the ASR hypothesis and reference text being genuinely
very close in content -- SequenceMatcher compares whole word tokens for
exact equality, so a single stray character (gershayim/quote marks around
honorific abbreviations, niqqud, punctuation) makes an otherwise-identical
word count as a total mismatch. The archival text convention of
"old_spelling [modern_spelling]" (a bracketed alternate reading) is
resolved to just the bracketed form, which is what's actually read aloud
-- confirmed by comparing a real hypothesis/reference pair live.
"""
import json
import os
import re
from difflib import SequenceMatcher

_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_QUOTE_RE = re.compile(r"[\"'׳״‘’“”]")
_PUNCT_RE = re.compile(r"[,.;:!?()]")
_BRACKET_GLOSS_RE = re.compile(r"\S+\s*\[([^\]]+)\]")

# A Hebrew ראשי תיבות abbreviation always has a gershayim/quote mark between
# two Hebrew letters (תו"מ, הקב"ה, שע"י, בעמ"נ...) -- that's the shape
# _ABBREV_CANDIDATE_RE flags as "worth trying to auto-expand", checked
# against the display token (quotes kept, niqqud stripped).
_ABBREV_CANDIDATE_RE = re.compile(r"[א-ת][\"'׳״][א-ת]")
_ANCHOR_LENGTHS = (6, 5, 4, 3, 2)
_MAX_EXPANSION_GAP = 6


def _expand_bracket_glosses(text: str) -> str:
    """Replace "old_spelling [modern_spelling]" with just "modern_spelling"."""
    return _BRACKET_GLOSS_RE.sub(r"\1", text)


def _strip_niqqud(word: str) -> str:
    return _NIQQUD_RE.sub("", word)


def _normalize_word(word: str) -> str:
    word = _strip_niqqud(word)
    word = _QUOTE_RE.sub("", word)
    word = _PUNCT_RE.sub("", word)
    return word


def _tokenize(text: str) -> list[str]:
    return _expand_bracket_glosses(text).split()


def normalize_words(text: str) -> list[str]:
    """Text -> list of comparison/output-ready words: bracket glosses
    expanded, niqqud/quotes/basic punctuation stripped, empty tokens
    dropped. Used for both the match comparison and (for the reference
    text) as the source of matched_text handed to stage 3 -- a clean,
    quote-free string is exactly what CTC/uroman romanization needs
    anyway (a stray `"` character has previously crashed that stage)."""
    return [w for w in (_normalize_word(w) for w in _tokenize(text)) if w]


def tokenize_with_display(text: str) -> tuple[list[str], list[str]]:
    """Positionally-aligned (stripped, display) token pair from the same raw
    split -- display[i] keeps quotes/punctuation (only niqqud stripped).
    Unlike normalize_words, empty-after-stripping tokens are NOT dropped
    here, so index i means the same token in both lists; callers needing
    normalize_words-equivalent output must apply the same drop-empty mask
    to both lists (see align_segments_to_text)."""
    raw = _tokenize(text)
    stripped = [_normalize_word(w) for w in raw]
    display = [_strip_niqqud(w) for w in raw]
    return stripped, display


def _find_sublist_occurrences(haystack: list[str], needle: list[str]) -> list[int]:
    """Start indices where `needle` occurs contiguously in `haystack`."""
    if not needle:
        return []
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if haystack[i:i + n] == needle]


def load_abbreviation_dict(path: str) -> dict[str, list[str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_abbreviation_dict(path: str, abbrev_dict: dict[str, list[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(abbrev_dict, f, ensure_ascii=False, indent=2, sort_keys=True)


def discover_abbreviation_expansions(
    reference_text: str,
    hyp_texts: list[str],
    known_dict: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Auto-detect how Hebrew ראשי תיבות abbreviations (תו"מ, הקב"ה, שע"י,
    בעמ"נ, ...) in `reference_text` were actually read aloud, by cross-
    referencing the ASR rough-transcript text (`hyp_texts`, in document
    order) around each abbreviation occurrence.

    Why this works: an abbreviation is a single reference token standing
    in for words the reader speaks in full, which is exactly what breaks
    match_segment_to_reference's word-for-word comparison -- badly enough
    that the mismatched region can make SequenceMatcher latch onto a
    distant, unrelated occurrence of a common word and drag matched_end
    far past where the segment actually ends (confirmed live: a single
    הקב"ה dragged one segment's matched_end 30 words into the next
    segment's territory). Fixing it means learning what each abbreviation
    was actually read as.

    For each abbreviation-shaped reference token not already in
    `known_dict`, this takes the words immediately before and after it as
    "anchors" -- they should appear verbatim in the ASR text, since
    abbreviations are the thing that breaks verbatim matching, not
    ordinary words -- and searches for that anchor pair in the
    concatenated hyp word stream. The hyp words found *between* a matching
    anchor pair are the observed expansion. Anchors start at
    _ANCHOR_LENGTHS[0] words and shrink, because the reference and spoken
    text can diverge slightly even near an abbreviation (a synonym, a
    tense change) without changing how the abbreviation itself was read;
    a wide anchor missing a verbatim match doesn't mean the abbreviation
    is unreadable, just that this particular anchor was too greedy.

    Returns (new_entries, report) -- new_entries is only the newly
    discovered {abbrev_token: [expansion_words]} pairs (the caller merges
    them into known_dict and persists via save_abbreviation_dict).
    report has one entry per abbreviation occurrence encountered
    (discovered / ambiguous / no match), for debugging.
    """
    known_dict = known_dict or {}
    ref_stripped_all, ref_display_all = tokenize_with_display(reference_text)
    keep = [i for i, w in enumerate(ref_stripped_all) if w]
    ref_words = [ref_stripped_all[i] for i in keep]
    ref_display_words = [ref_display_all[i] for i in keep]

    hyp_stream: list[str] = []
    for t in hyp_texts:
        hyp_stream.extend(normalize_words(t))

    new_entries: dict[str, list[str]] = {}
    report = []
    for idx, (word, display) in enumerate(zip(ref_words, ref_display_words)):
        if word in known_dict or word in new_entries:
            continue
        if not _ABBREV_CANDIDATE_RE.search(display):
            continue

        status, expansion, anchor_len_used = "no_match", None, None
        for anchor_len in _ANCHOR_LENGTHS:
            before_len = min(anchor_len, idx)
            after_len = min(anchor_len, len(ref_words) - idx - 1)
            if before_len == 0 or after_len == 0:
                continue
            anchor_before = ref_words[idx - before_len:idx]
            anchor_after = ref_words[idx + 1:idx + 1 + after_len]
            ends_before = [
                i + before_len for i in _find_sublist_occurrences(hyp_stream, anchor_before)
            ]
            starts_after = _find_sublist_occurrences(hyp_stream, anchor_after)
            candidates = {
                tuple(hyp_stream[e1:p2])
                for e1 in ends_before
                for p2 in starts_after
                if 0 <= p2 - e1 <= _MAX_EXPANSION_GAP
            }
            if not candidates:
                continue  # anchors too strict at this length -- try shorter
            anchor_len_used = anchor_len
            if len(candidates) == 1:
                status, expansion = "discovered", list(next(iter(candidates)))
            else:
                status = "ambiguous"
            break

        report.append(
            {
                "token": word,
                "ref_index": idx,
                "status": status,
                "expansion": expansion,
                "anchor_len": anchor_len_used,
            }
        )
        if status == "discovered":
            new_entries[word] = expansion

    return new_entries, report


def match_segment_to_reference(
    hyp_text: str,
    ref_words: list[str],
    cursor: int,
    lookahead_words: int = 200,
    min_match_ratio: float = 0.4,
    mismatch_log: list[dict] | None = None,
    segment_context: dict | None = None,
) -> tuple[int, int, float] | None:
    """Find where `hyp_text` best matches within ref_words[cursor : cursor+lookahead_words].

    `ref_words` must already be normalized (see normalize_words) -- this
    function normalizes `hyp_text` the same way before comparing.

    Returns (matched_start, matched_end, match_ratio) as indices into
    ref_words, or None if no sufficiently good match was found.

    If `mismatch_log` is given, every failure path appends a diagnostic
    dict to it (reason, cursor, hyp_text, and whatever ratio/window info
    was computed before the decision to bail) merged with `segment_context`
    -- otherwise that info (e.g. a low match_ratio) is computed then
    silently discarded, leaving no trail for why a segment was rejected.
    """

    def _log(reason: str, **extra) -> None:
        if mismatch_log is not None:
            mismatch_log.append(
                {"reason": reason, "cursor": cursor, "hyp_text": hyp_text, **extra, **(segment_context or {})}
            )

    hyp_words = normalize_words(hyp_text)
    if not hyp_words:
        _log("empty_hyp_text")
        return None

    window_end = min(len(ref_words), cursor + lookahead_words)
    window = ref_words[cursor:window_end]
    if not window:
        _log("empty_window")
        return None

    sm = SequenceMatcher(None, window, hyp_words, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        _log("no_matching_blocks", hyp_word_count=len(hyp_words), window_preview=" ".join(window[:30]))
        return None

    # NOT sm.ratio(): that's 2*M / (len(window) + len(hyp_words)), which
    # is diluted by the search window's length -- with lookahead_words=200
    # and a ~30-word hypothesis, even a near-perfect match scores ~0.2
    # (confirmed live). What we actually want is "what fraction of the
    # hypothesis did we find somewhere in the window", independent of how
    # large the window itself is.
    matched_word_count = sum(b.size for b in blocks)
    match_ratio = matched_word_count / len(hyp_words)
    if match_ratio < min_match_ratio:
        _log(
            "low_match_ratio",
            hyp_word_count=len(hyp_words),
            matched_word_count=matched_word_count,
            match_ratio=round(match_ratio, 4),
            window_preview=" ".join(window[:30]),
        )
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
    debug_dir: str | None = None,
    abbrev_dict: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Annotate each rough segment with its matched reference-text span.

    Adds "ref_start", "ref_end", "matched_text", "matched_display_words",
    "match_ratio" to each segment dict (ref_start is None when unmatched).
    matched_text is the stripped, CTC-safe string (as before); aligned to
    it 1:1 by word position, matched_display_words restores quotes/
    punctuation for final display -- CTC alignment itself never sees
    punctuation. The cursor only advances on a successful match, so an
    unmatched segment does not throw off the search window for the next
    one.

    If `debug_dir` is given, every unmatched segment's diagnostic info
    (see match_segment_to_reference's mismatch_log) is written to
    debug_dir/text_match_mismatches.json, matching the debug-artifact
    convention rough_transcribe uses for debug_dir/vad_chunks.json.

    `abbrev_dict` (see discover_abbreviation_expansions) maps a raw
    abbreviation reference token to the words it should expand to before
    matching -- e.g. "הקבה" -> ["הקדוש", "ברוך", "הוא"]. Each occurrence is
    spliced into ref_words/ref_display_words in place, so match_ratio is
    computed against what was actually said instead of against a single
    token that can never word-for-word match its multi-word reading.
    """
    ref_stripped_all, ref_display_all = tokenize_with_display(reference_text)
    keep = [i for i, w in enumerate(ref_stripped_all) if w]
    ref_words = [ref_stripped_all[i] for i in keep]  # == normalize_words(reference_text)
    ref_display_words = [ref_display_all[i] for i in keep]  # 1:1 with ref_words

    if abbrev_dict:
        expanded_words, expanded_display = [], []
        for w, d in zip(ref_words, ref_display_words):
            expansion = abbrev_dict.get(w)
            if expansion:
                expanded_words.extend(expansion)
                expanded_display.extend(expansion)
            else:
                expanded_words.append(w)
                expanded_display.append(d)
        ref_words, ref_display_words = expanded_words, expanded_display

    mismatch_log = [] if debug_dir else None
    cursor = 0
    matched = []
    for i, seg in enumerate(segments):
        result = match_segment_to_reference(
            seg["text"],
            ref_words,
            cursor,
            lookahead_words,
            min_match_ratio,
            mismatch_log=mismatch_log,
            segment_context={"segment_index": i, "seg_start": seg.get("start"), "seg_end": seg.get("end")},
        )
        if result is None:
            matched.append(
                {
                    **seg,
                    "ref_start": None,
                    "ref_end": None,
                    "matched_text": None,
                    "matched_display_words": None,
                    "match_ratio": 0.0,
                }
            )
            continue
        ref_start, ref_end, ratio = result
        matched.append(
            {
                **seg,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "matched_text": " ".join(ref_words[ref_start:ref_end]),
                "matched_display_words": ref_display_words[ref_start:ref_end],
                "match_ratio": ratio,
            }
        )
        cursor = ref_end

    if debug_dir and mismatch_log:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "text_match_mismatches.json"), "w", encoding="utf-8") as f:
            json.dump(mismatch_log, f, ensure_ascii=False, indent=2)

    return matched

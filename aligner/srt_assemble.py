"""Stage 4: assemble per-window CTC word alignments into subtitle-level SRT
cues, with confidence-based QC flags.

Low-confidence cues are not silently dropped -- they are written to a
separate QC report so a human (or a downstream filter) decides whether to
exclude them, matching the project's general QC pattern of flag-then-decide
rather than flag-and-guess.
"""
import json
import os
import re

_HARD_END_RE = re.compile(r'[.!?]["\'׳״‘’“”]?$')
_SOFT_END_RE = re.compile(r'[,;:]["\'׳״‘’“”]?$')
_MIN_HARD_BREAK_WORDS = 3


def _format_ts(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, ms_total = divmod(ms_total, 3600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _display(word: dict) -> str:
    return word.get("display_word", word["word"])


def _finalize_cue(chunk: list[dict], gap_after: bool) -> dict:
    return {
        "start": chunk[0]["start"],
        "end": chunk[-1]["end"],
        "text": " ".join(_display(w) for w in chunk),
        "min_score": min(w["score"] for w in chunk),
        "gap_after": gap_after,
    }


def words_to_cues(
    all_words: list[dict],
    words_per_cue: int = 12,
    min_score: float = 0.5,
    max_gap_s: float = 3.0,
    max_cue_words: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Group a flat, time-ordered word list into subtitle cues, breaking on
    (in priority order) a time gap, sentence-ending punctuation, clause-
    ending punctuation once at target length, or a hard word-count cap.

    all_words: [{"word","start","end","score",...}, ...] with start/end
    already offset to absolute (whole-file) time; "display_word" (if
    present) is used for cue text instead of "word" so CTC alignment can
    keep operating on stripped text while cues show punctuation.

    A gap between consecutive words larger than max_gap_s always closes
    the current cue, regardless of word count -- this is what stops a
    dropped/unmatched/CTC-failed window from being silently bridged into
    one cue spanning the whole gap.

    Returns (cues, flagged) where flagged is the subset of cues containing
    at least one word below min_score.
    """
    max_cue_words = max_cue_words or 2 * words_per_cue

    cues, chunk = [], []
    for i, w in enumerate(all_words):
        chunk.append(w)
        is_last = i == len(all_words) - 1
        gap_break = (not is_last) and (all_words[i + 1]["start"] - w["end"] > max_gap_s)
        text = _display(w).rstrip()
        hard_break = len(chunk) >= _MIN_HARD_BREAK_WORDS and bool(_HARD_END_RE.search(text))
        soft_break = len(chunk) >= words_per_cue and bool(_SOFT_END_RE.search(text))
        size_break = len(chunk) >= max_cue_words

        if gap_break or hard_break or soft_break or size_break or is_last:
            cues.append(_finalize_cue(chunk, gap_after=gap_break))
            chunk = []

    flagged = [c for c in cues if c["min_score"] < min_score]
    return cues, flagged


def write_srt(cues: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, cue in enumerate(cues, 1):
            f.write(
                f"{idx}\n{_format_ts(cue['start'])} --> {_format_ts(cue['end'])}\n{cue['text']}\n\n"
            )


def write_qc_report(flagged: list[dict], skipped_windows: list[dict], path: str) -> None:
    report = {"flagged_cues": flagged, "skipped_windows": skipped_windows}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

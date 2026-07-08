"""Stage 4: assemble per-window CTC word alignments into subtitle-level SRT
cues, with confidence-based QC flags.

Low-confidence cues are not silently dropped -- they are written to a
separate QC report so a human (or a downstream filter) decides whether to
exclude them, matching the project's general QC pattern of flag-then-decide
rather than flag-and-guess.
"""
import json


def _format_ts(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, ms_total = divmod(ms_total, 3600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_cues(
    all_words: list[dict],
    words_per_cue: int = 12,
    min_score: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Group a flat, time-ordered word list into fixed-size subtitle cues.

    all_words: [{"word","start","end","score"}, ...] with start/end already
    offset to absolute (whole-file) time.
    Returns (cues, flagged) where flagged is the subset of cues containing
    at least one word below min_score.
    """
    cues, flagged = [], []
    for i in range(0, len(all_words), words_per_cue):
        chunk = all_words[i : i + words_per_cue]
        if not chunk:
            continue
        cue = {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(w["word"] for w in chunk),
            "min_score": min(w["score"] for w in chunk),
        }
        cues.append(cue)
        if cue["min_score"] < min_score:
            flagged.append(cue)
    return cues, flagged


def write_srt(cues: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, cue in enumerate(cues, 1):
            f.write(
                f"{idx}\n{_format_ts(cue['start'])} --> {_format_ts(cue['end'])}\n{cue['text']}\n\n"
            )


def write_qc_report(flagged: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)

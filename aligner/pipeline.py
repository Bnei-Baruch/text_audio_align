"""End-to-end orchestration of the 4-stage text-audio alignment pipeline.

    1. rough_transcribe  -- Whisper long-form pass -> approximate segments
    2. text_match        -- locate each segment's span in the reference text
    3. ctc_align         -- precise CTC forced alignment per matched window
    4. srt_assemble      -- collect word timings into SRT cues + QC report

Unmatched segments (stage 2) are excluded from alignment rather than forced
-- today (pure-reading assumption) this means "this file has a problem,
review it"; the unmatched count is reported so that decision is visible
rather than silent.
"""
import os

import soundfile as sf
import torch

from .ctc_align import align_window
from .rough_transcribe import rough_transcribe
from .srt_assemble import words_to_cues, write_qc_report, write_srt
from .text_match import align_segments_to_text


def run_pipeline(cfg: dict) -> dict:
    audio_path = cfg["audio_path"]
    text_path = cfg["reference_text_path"]
    output_srt_path = cfg.get("output_srt_path", "output/output.srt")
    output_qc_path = cfg.get("output_qc_path", "output/output_qc.json")
    whisper_model = cfg.get("whisper_model", "openai/whisper-large-v3")
    language = cfg.get("language", "he")
    mms_lang_code = cfg.get("mms_lang_code", "heb")
    chunk_length_s = cfg.get("chunk_length_s", 30)
    min_silence_duration_ms = cfg.get("min_silence_duration_ms", 100)
    max_speech_duration_s = cfg.get("max_speech_duration_s")
    lookahead_words = cfg.get("lookahead_words", 200)
    min_match_ratio = cfg.get("min_match_ratio", 0.4)
    min_ctc_score = cfg.get("min_ctc_score", 0.5)
    words_per_cue = cfg.get("words_per_cue", 12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_dir = os.path.join(os.path.dirname(output_srt_path), "debug") if cfg.get("debug", False) else None

    with open(text_path, "r", encoding="utf-8") as f:
        reference_text = f.read()

    print(f"[1/4] Rough transcription: {audio_path}")
    segments = rough_transcribe(
        audio_path,
        model_name=whisper_model,
        language=language,
        chunk_length_s=chunk_length_s,
        device=device,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        debug_dir=debug_dir,
    )
    print(f"  -> {len(segments)} rough segments")
    if debug_dir:
        print(f"  -> debug: VAD chunks + log saved to {debug_dir}")

    print("[2/4] Matching segments to reference text")
    matched = align_segments_to_text(segments, reference_text, lookahead_words, min_match_ratio)
    n_unmatched = sum(1 for m in matched if m["ref_start"] is None)
    print(
        f"  -> {len(matched) - n_unmatched}/{len(matched)} segments matched "
        f"({n_unmatched} unmatched -- likely insertions or a data/ASR problem)"
    )

    print("[3/4] CTC forced alignment per matched window")
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # [N, channels]
    except Exception as e:
        raise RuntimeError(
            f"Could not read audio_path={audio_path!r} as WAV/FLAC/OGG via soundfile "
            f"({e}). This pipeline expects a plain, directly-readable audio file, not a "
            "raw compressed download -- e.g. the WAV files fetch_data/download.py already "
            "produces via ffmpeg. If you supplied your own file per the README's manual "
            '"Usage" step, convert it first:\n'
            f"  ffmpeg -y -i {audio_path!r} -ac 1 -ar 16000 <path-to-your-file>.wav"
        ) from e
    waveform = torch.from_numpy(data.T.copy())  # [channels, N], matches torchaudio.load's layout
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    all_words = []
    n_failed = 0
    for m in matched:
        if m["ref_start"] is None:
            continue
        start_sample = int(m["start"] * sr)
        end_sample = int(m["end"] * sr)
        window_wave = waveform[:, start_sample:end_sample]
        try:
            words = align_window(window_wave, sr, m["matched_text"], lang_code=mms_lang_code)
        except Exception as e:
            print(f"  ! CTC alignment failed for window {m['start']:.1f}-{m['end']:.1f}s: {e}")
            n_failed += 1
            continue
        for w in words:
            w["start"] += m["start"]
            w["end"] += m["start"]
        all_words.extend(words)

    print(f"[4/4] Assembling SRT ({len(all_words)} aligned words, {n_failed} windows failed)")
    cues, flagged = words_to_cues(all_words, words_per_cue=words_per_cue, min_score=min_ctc_score)
    write_srt(cues, output_srt_path)
    write_qc_report(flagged, output_qc_path)
    print(f"  -> {output_srt_path} ({len(cues)} cues, {len(flagged)} flagged for review)")

    return {
        "rough_segments": len(segments),
        "matched_segments": len(matched) - n_unmatched,
        "unmatched_segments": n_unmatched,
        "ctc_failed_windows": n_failed,
        "aligned_words": len(all_words),
        "cues": len(cues),
        "flagged_cues": len(flagged),
    }

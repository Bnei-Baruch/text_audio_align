"""Voice-activity-detection based chunk boundaries.

Chunking a long recording at a fixed time interval (e.g. exactly every
30.000s) routinely cuts mid-word, degrading both the rough transcription
(stage 1) and, more importantly, precise CTC alignment (stage 3) for
whichever word straddles the cut. Confirmed live: transformers'
pipeline(chunk_length_s=...) path (its own alternative to fixed-interval
slicing) produced a 217-second runaway "chunk" for one window -- both
approaches share the same underlying problem of not respecting where
speech actually pauses.

Silero VAD (via torch.hub) finds real speech/silence boundaries directly
in the signal, with no transcription involved. Chunk boundaries are built
by greedily merging consecutive VAD speech segments up to ~target_s each,
so every cut lands in an actual silence gap between two segments, never
inside one. This is the same role VAD plays in WhisperX and other
long-form ASR pipelines.
"""
import torch

_vad_model = None
_get_speech_timestamps = None


def _load_vad():
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        _vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        _get_speech_timestamps = utils[0]
    return _vad_model, _get_speech_timestamps


def get_speech_chunks(
    waveform: torch.Tensor,
    sr: int,
    target_s: float = 30.0,
    min_silence_duration_ms: int = 100,
    max_speech_duration_s: float | None = None,
) -> list[tuple[int, int]]:
    """Returns [(start_sample, end_sample), ...] chunk boundaries.

    Each boundary falls in a silence gap between two VAD-detected speech
    segments -- never mid-word.

    min_silence_duration_ms: how long a quiet stretch must last for Silero
    to treat it as a real gap between speech segments, rather than bridging
    over it as part of one continuous speech run. Lower values make VAD
    willing to cut on shorter pauses; too low risks false positives on
    brief in-word closures (stop consonants, breaths), reintroducing the
    mid-word-cut problem this function exists to avoid.

    max_speech_duration_s: hard cap on a single Silero-detected speech
    segment; defaults to target_s. Silero's own default is unbounded, so a
    long uninterrupted speech run with no detected pause becomes one
    oversized chunk -- rare in practice, but stage 3's CTC aligner is not
    designed to handle a window that long (see ctc_align.py). When a
    segment would exceed this cap, Silero picks the least-speech-like
    point within it to split (via min_silence_at_max_speech), not an
    arbitrary sample boundary.

    Falls back to a single (0, len) chunk if VAD finds no speech at all
    (e.g. a music-only intro with no spoken content).
    """
    model, get_speech_timestamps = _load_vad()
    if sr != 16000:
        raise ValueError(f"Silero VAD expects 16kHz audio, got {sr}Hz")
    if max_speech_duration_s is None:
        max_speech_duration_s = target_s

    speech = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sr,
        min_silence_duration_ms=min_silence_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
    )
    if not speech:
        return [(0, waveform.shape[-1])]

    target_samples = int(target_s * sr)
    chunks: list[tuple[int, int]] = []
    cur_start, cur_end = speech[0]["start"], speech[0]["end"]
    for seg in speech[1:]:
        if seg["end"] - cur_start <= target_samples:
            cur_end = seg["end"]  # still within target once merged -> keep merging
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
    chunks.append((cur_start, cur_end))
    return chunks

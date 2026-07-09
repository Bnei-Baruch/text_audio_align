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
    waveform: torch.Tensor, sr: int, target_s: float = 30.0
) -> list[tuple[int, int]]:
    """Returns [(start_sample, end_sample), ...] chunk boundaries.

    Each boundary falls in a silence gap between two VAD-detected speech
    segments -- never mid-word. A single uninterrupted speech run longer
    than target_s (no detected pause) becomes its own oversized chunk
    rather than being force-cut mid-speech; this is rare in practice
    (natural reading has breathing pauses) and preferred over
    reintroducing the mid-word-cut problem this function exists to avoid.

    Falls back to a single (0, len) chunk if VAD finds no speech at all
    (e.g. a music-only intro with no spoken content).
    """
    model, get_speech_timestamps = _load_vad()
    if sr != 16000:
        raise ValueError(f"Silero VAD expects 16kHz audio, got {sr}Hz")

    speech = get_speech_timestamps(waveform, model, sampling_rate=sr)
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

"""Stage 1: rough ASR pass over a long recording.

Purpose: produce approximate (start, end, text) segments covering the whole
file. The text does not need to be accurate -- it is only used as a search
key in stage 2 (text_match.py) to locate the corresponding span in the known
reference text. Timing only needs to be roughly right, since stage 3
(ctc_align.py) re-derives precise word timestamps per window.

Uses HF transformers' chunked ASR pipeline, which internally splits long
audio into `chunk_length_s`-second windows and stitches results back
together -- no manual slicing needed.
"""
from transformers import pipeline


def rough_transcribe(
    audio_path: str,
    model_name: str = "openai/whisper-large-v3",
    language: str = "he",
    chunk_length_s: int = 30,
    device: str = "cuda",
) -> list[dict]:
    """Returns a list of {"start": float, "end": float, "text": str} segments."""
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=chunk_length_s,
        device=device,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    result = asr(audio_path, return_timestamps=True)

    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        if start is None or end is None:
            # Whisper occasionally fails to close the last segment's timestamp
            # on truncated/edge-case audio; drop rather than guess.
            continue
        text = chunk["text"].strip()
        if not text:
            continue
        segments.append({"start": float(start), "end": float(end), "text": text})
    return segments

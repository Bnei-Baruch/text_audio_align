"""Stage 1: rough ASR pass over a long recording.

Purpose: produce approximate (start, end, text) segments covering the whole
file. The text does not need to be accurate -- it is only used as a search
key in stage 2 (text_match.py) to locate the corresponding span in the known
reference text. Timing only needs to be roughly right, since stage 3
(ctc_align.py) re-derives precise word timestamps per window.

Chunk boundaries come from VAD (vad_chunk.get_speech_chunks), not a fixed
time interval or transformers' pipeline(chunk_length_s=...) path. The
latter is explicitly flagged by transformers itself as "very experimental
with seq2seq models", and was observed live to produce wildly uneven
segments (as few as 1 for an 809s file, and one window spanning 200+
seconds) -- unusable both as a stage-2 search key and, worse, as a stage-3
CTC window (CTC drift risk grows with window length). VAD sidesteps this
by cutting only in actual silence gaps, never mid-word.

Each VAD-derived chunk is decoded independently via a plain short-form
generate() call -- exactly what Whisper (and this project's fine-tuned
checkpoint) is trained on -- the same decode pattern whisper_trainer's
eval.py _decode_batch uses for its own 30s windows.

Reads audio via soundfile rather than torchaudio.load(): files reaching
this function are always plain 16kHz mono WAV (fetch_data/download.py
normalizes on download), so no container parsing is needed here, and this
avoids newer torchaudio versions' hard dependency on the separate
torchcodec package for .load().

Note: a VAD chunk longer than chunk_length_s (rare -- only when no pause
is detected within that span) is still passed whole to the feature
extractor, which pads/truncates to 30s by default -- so content beyond 30s
inside such an oversized chunk is silently dropped from the *rough*
transcript. This only weakens stage 2's search key for that one chunk; it
does not affect stage 3's precision elsewhere.
"""
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .vad_chunk import get_speech_chunks

_model_cache: dict[tuple, tuple] = {}


def _get_model_and_processor(model_name: str, device: str):
    key = (model_name, device)
    if key not in _model_cache:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
        model.eval()
        _model_cache[key] = (model, processor)
    return _model_cache[key]


def rough_transcribe(
    audio_path: str,
    model_name: str = "openai/whisper-large-v3",
    language: str = "he",
    chunk_length_s: int = 30,
    device: str = "cuda",
) -> list[dict]:
    """Returns a list of {"start": float, "end": float, "text": str} segments."""
    model, processor = _get_model_and_processor(model_name, device)

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # [N, channels]
    array = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != 16000:
        raise ValueError(
            f"{audio_path}: expected 16kHz audio (fetch_data/download.py should "
            f"guarantee this) -- got {sr}Hz. Re-run the fetch step, or resample "
            f"before calling rough_transcribe."
        )

    waveform = torch.from_numpy(array.copy())
    chunk_bounds = get_speech_chunks(waveform, sr, target_s=chunk_length_s)

    segments = []
    with torch.inference_mode():
        for start_sample, end_sample in chunk_bounds:
            chunk_array = array[start_sample:end_sample]

            inputs = processor(chunk_array, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=model.dtype)

            gen_ids = model.generate(
                input_features=input_features,
                language=language,
                task="transcribe",
                do_sample=False,
                num_beams=1,
            )
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            if not text:
                continue
            segments.append(
                {"start": start_sample / sr, "end": end_sample / sr, "text": text}
            )
    return segments

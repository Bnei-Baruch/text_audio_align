"""Stage 3: precise CTC forced alignment of a short audio window against its
matched reference-text span (from stage 2), using Meta's MMS forced-alignment
model via torchaudio.

Deliberately operates on short (<=~30s) windows, not the whole recording:
CTC/Viterbi forced alignment is known to drift once it hits a mismatch or
noisy region, and that drift can propagate through the rest of a long pass.
Keeping windows short bounds the damage to one window.

NOTE -- verify this against your installed torchaudio version before
trusting it. This follows the official "Forced alignment for multilingual
data" tutorial API shape as of when this was written; torchaudio's
pipelines API has changed across versions before. Run the smoke test in
README.md first.

MMS_FA expects romanized (Latin-script) text for non-Latin languages, hence
the `uroman` step -- Hebrew text is romanized before alignment, and the
resulting per-token spans are mapped back to the original Hebrew words by
position (words and their romanized counterparts stay 1:1 as long as
`text.split()` and the romanized string have the same word count, which
holds for uroman's word-preserving transliteration).
"""
import torch
import torchaudio

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BUNDLE = torchaudio.pipelines.MMS_FA

_model = None
_tokenizer = None
_aligner = None
_uroman = None


def _get_model():
    global _model, _tokenizer, _aligner
    if _model is None:
        _model = _BUNDLE.get_model(with_star=False).to(_DEVICE)
        _tokenizer = _BUNDLE.get_tokenizer()
        _aligner = _BUNDLE.get_aligner()
    return _model, _tokenizer, _aligner


def _get_uroman():
    global _uroman
    if _uroman is None:
        import uroman as ur

        _uroman = ur.Uroman()
    return _uroman


def _romanize(text: str, lang_code: str = "heb") -> str:
    return _get_uroman().romanize_string(text, lcode=lang_code)


def align_window(
    waveform: torch.Tensor,
    sample_rate: int,
    text: str,
    lang_code: str = "heb",
) -> list[dict]:
    """Force-align `text` (known-correct, original language) against `waveform`.

    waveform: 1D or [1, N] float tensor covering just this window (not the
    full recording).
    Returns [{"word": str, "start": float_sec, "end": float_sec, "score": float}, ...]
    in window-relative time (caller offsets to absolute file time).
    """
    model, tokenizer, aligner = _get_model()

    if sample_rate != _BUNDLE.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, _BUNDLE.sample_rate)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    orig_words = text.split()
    roman_text = _romanize(text, lang_code=lang_code)
    roman_words = roman_text.split()
    if len(roman_words) != len(orig_words):
        # uroman is expected to preserve word count; if it doesn't for some
        # input, alignment-to-original-word mapping below would be wrong.
        raise ValueError(
            f"Romanization word-count mismatch ({len(orig_words)} orig vs "
            f"{len(roman_words)} romanized) -- cannot safely map spans back "
            f"to original words for text: {text!r}"
        )

    with torch.inference_mode():
        emission, _ = model(waveform.to(_DEVICE))
    token_spans = aligner(emission[0], tokenizer(roman_words))

    num_frames = emission.size(1)
    seconds_per_frame = waveform.size(1) / num_frames / _BUNDLE.sample_rate

    results = []
    for i, spans in enumerate(token_spans):
        start_sec = spans[0].start * seconds_per_frame
        end_sec = spans[-1].end * seconds_per_frame
        total_len = sum(s.length for s in spans) if hasattr(spans[0], "length") else len(spans)
        score = sum(s.score * getattr(s, "length", 1) for s in spans) / max(1, total_len)
        results.append(
            {
                "word": orig_words[i],
                "start": float(start_sec),
                "end": float(end_sec),
                "score": float(score),
            }
        )
    return results

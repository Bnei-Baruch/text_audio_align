"""Download audio and companion text for content units found via db.py.

Audio: GET https://cdn.kabbalahmedia.info/{file_uid}.mp3
Text: reuses the doc2text API already used successfully by the sibling
project (trlAi/src/prepare_data.py's fetch_files_by_uid), since that's the
only confirmed-working way to pull text content out of this database.

Assumes the URL's "[file id]" is the file's `uid` (string), not its
numeric `id` -- matches the sibling project's convention of using `uid` in
its own kabbalahmedia URL. Verify this if downloads 404.

Audio is re-encoded to WAV via ffmpeg immediately after download, not kept
as the raw downloaded bytes: files served from this URL despite the .mp3
extension have been observed to actually be ISO-BMFF/MP4 containers (AAC
audio) with the moov atom at the end of the file. Readers that pipe bytes
in via stdin (e.g. transformers' ffmpeg_read, some torchaudio decode
paths) can't seek to find it and fail with a "malformed" error, even
though the file is perfectly valid and `ffmpeg -i <path>` (which can seek
a real file) reads it without complaint. Normalizing to WAV once here
means every downstream consumer just opens a plain WAV file -- no
per-consumer special-casing needed.
"""
import logging
import os
import subprocess
import tempfile

import requests

logger = logging.getLogger(__name__)

AUDIO_URL_TEMPLATE = "https://cdn.kabbalahmedia.info/{file_id}.mp3"
DOC2TEXT_URL_TEMPLATE = "https://kabbalahmedia.info/assets/api/doc2text/{uid}"


def download_audio(file_uid: str, dest_path: str, chunk_size: int = 1 << 16) -> bool:
    """Download the audio and write it to dest_path as 16kHz mono WAV
    (dest_path should end in .wav; the source format/extension doesn't
    matter, ffmpeg re-encodes whatever it actually is)."""
    url = AUDIO_URL_TEMPLATE.format(file_id=file_uid)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    raw_fd, raw_path = tempfile.mkstemp(suffix=".src", dir=os.path.dirname(dest_path))
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with os.fdopen(raw_fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
    except requests.RequestException as e:
        logger.error(f"Failed to download audio {file_uid} from {url}: {e}")
        os.remove(raw_path)
        return False

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", raw_path,
        "-ac", "1", "-ar", "16000",
        dest_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(raw_path)
    if proc.returncode != 0:
        logger.error(
            f"ffmpeg failed to re-encode downloaded audio {file_uid}: "
            f"{proc.stderr.decode(errors='replace')}"
        )
        return False
    return True


def fetch_text(uid: str) -> str | None:
    url = DOC2TEXT_URL_TEMPLATE.format(uid=uid)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch text {uid} from {url}: {e}")
        return None

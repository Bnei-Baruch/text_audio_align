"""Download audio and companion text for content units found via db.py.

Audio: GET https://cdn.kabbalahmedia.info/{file_uid}.mp3
Text: reuses the doc2text API already used successfully by the sibling
project (trlAi/src/prepare_data.py's fetch_files_by_uid), since that's the
only confirmed-working way to pull text content out of this database.

Assumes the URL's "[file id]" is the file's `uid` (string), not its
numeric `id` -- matches the sibling project's convention of using `uid` in
its own kabbalahmedia URL. Verify this if downloads 404.
"""
import logging
import os

import requests

logger = logging.getLogger(__name__)

AUDIO_URL_TEMPLATE = "https://cdn.kabbalahmedia.info/{file_id}.mp3"
DOC2TEXT_URL_TEMPLATE = "https://kabbalahmedia.info/assets/api/doc2text/{uid}"


def download_audio(file_uid: str, dest_path: str, chunk_size: int = 1 << 16) -> bool:
    url = AUDIO_URL_TEMPLATE.format(file_id=file_uid)
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download audio {file_uid} from {url}: {e}")
        return False


def fetch_text(uid: str) -> str | None:
    url = DOC2TEXT_URL_TEMPLATE.format(uid=uid)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch text {uid} from {url}: {e}")
        return None

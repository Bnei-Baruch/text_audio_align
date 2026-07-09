#!/usr/bin/env python
"""Fetch SOURCE content units that have a linked audio file from the mdb
database, and download audio (+ companion text, if present) into
data/<content_unit_uid>/, ready for run_align.py.

Usage:
    python fetch_source_audio.py [fetch_data/config.json] [--limit N] [--offset N]

`limit` (max content units to fetch) can be set in the config file
("limit": null means no limit -- fetch all); --limit on the command line
overrides the config value when given.

Requires fetch_data/db_secrets.json (gitignored) -- copy
fetch_data/db_secrets.json.example and fill in real host/port/dbname/user/password.

Before trusting this against the real database, run the schema check:
    python -c "
from fetch_data.db import get_db_connection, explore_schema
conn = get_db_connection()
explore_schema(conn)
"
"""
import argparse
import json
import logging
import os

from fetch_data.db import (
    count_source_audio_units,
    fetch_source_audio_batch,
    fetch_text_files_for_unit,
    get_db_connection,
)
from fetch_data.download import download_audio, fetch_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(cfg: dict, limit: int | None, offset: int) -> None:
    conn = get_db_connection()
    cu_type = cfg.get("content_unit_type", "SOURCE")
    file_type = cfg.get("file_type", "audio")
    preferred_lang = cfg.get("preferred_text_language", "he")
    batch_size = cfg.get("batch_size", 100)
    data_dir = cfg.get("data_dir", "data")

    try:
        total = count_source_audio_units(conn, cu_type, file_type)
        logger.info(f"Found {total} content units of type '{cu_type}' with a '{file_type}' file")
        remaining = total - offset if limit is None else min(limit, total - offset)

        fetched = 0
        cur_offset = offset
        while fetched < remaining:
            batch_limit = min(batch_size, remaining - fetched)
            rows = fetch_source_audio_batch(conn, cu_type, file_type, cur_offset, batch_limit)
            if not rows:
                break

            for row in rows:
                unit_dir = os.path.join(data_dir, row["cu_uid"])
                if os.path.isdir(unit_dir):
                    logger.info(f"[{row['cu_uid']}] output dir already exists, skipping")
                    continue

                audio_path = os.path.join(unit_dir, f"{row['file_uid']}.wav")
                logger.info(f"[{row['cu_uid']}] downloading audio {row['file_uid']}")
                if not download_audio(row["file_uid"], audio_path):
                    continue

                text_files = fetch_text_files_for_unit(conn, row["cu_id"])
                if not text_files:
                    logger.warning(f"[{row['cu_uid']}] no companion text file found")
                    continue

                chosen = next(
                    (t for t in text_files if t["language"] == preferred_lang), text_files[0]
                )
                text_path = os.path.join(unit_dir, "reference_text.txt")
                text = fetch_text(chosen["uid"])
                if text:
                    os.makedirs(unit_dir, exist_ok=True)
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    logger.info(f"[{row['cu_uid']}] saved text ({chosen['language']})")
                else:
                    logger.warning(f"[{row['cu_uid']}] text fetch failed")

            fetched += len(rows)
            cur_offset += len(rows)
            logger.info(f"Progress: {fetched}/{remaining}")

        logger.info("Done")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="fetch_data/config.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    limit = args.limit if args.limit is not None else cfg.get("limit")
    run(cfg, limit=limit, offset=args.offset)


if __name__ == "__main__":
    main()

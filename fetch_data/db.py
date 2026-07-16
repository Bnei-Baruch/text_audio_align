"""Database connection and queries against the mdb Postgres database.

Schema confirmed against a live connection (localhost:5432/davgur):
  - `content_units.type_id` is a bigint FK into `content_types(id, name)`
    -- NOT a plain string column. `content_types.name` holds values like
    'SOURCE' (id=47), 'LECTURE', 'DAILY_LESSON', etc. -- confirmed via
    `explore_schema()` + `SELECT * FROM content_types`.
  - `files.type` IS a plain string column (e.g. 'audio', 'text') --
    confirmed both here and in the sibling project
    trlAi/src/prepare_data.py, which compares `f.type = 'text'` directly.

So content_units needs a join against content_types; files does not.
`SELECT COUNT(*) ... WHERE ct.name = 'SOURCE' AND f.type = 'audio'` was
tested against the live database and returned 1352 rows.
"""
import json
import logging
import os

import psycopg2

logger = logging.getLogger(__name__)

QUERY_COUNT_SOURCE_AUDIO = """
SELECT COUNT(*)
FROM content_units cu
INNER JOIN content_types ct ON ct.id = cu.type_id
INNER JOIN files f ON f.content_unit_id = cu.id
WHERE ct.name = %(cu_type)s
    AND f.type = %(file_type)s
"""

QUERY_SOURCE_AUDIO = """
SELECT cu.id AS cu_id, cu.uid AS cu_uid, f.id AS file_id, f.uid AS file_uid
FROM content_units cu
INNER JOIN content_types ct ON ct.id = cu.type_id
INNER JOIN files f ON f.content_unit_id = cu.id
WHERE ct.name = %(cu_type)s
    AND f.type = %(file_type)s
    AND f.language = 'he'
    AND f.removed_at IS NULL
ORDER BY cu.id
OFFSET %(offset)s
LIMIT %(limit)s
"""

QUERY_TEXT_FOR_UNIT = """
SELECT uid, language
FROM files
WHERE content_unit_id = %(cu_id)s
    AND type = 'text'
    AND language = 'he'
    AND removed_at IS NULL
    AND (name ILIKE '%%.doc' OR name ILIKE '%%.docx')
"""


def _load_secrets() -> dict:
    """Read connection secrets (host, port, user, password) from
    fetch_data/db_secrets.json.

    This file is gitignored (see .gitignore) -- host/user/password never
    live in config.json, which is git-tracked. Mirrors the
    settings.toml / .secrets.toml split already used by the sibling trlAi
    project. Copy db_secrets.json.example to db_secrets.json and fill in
    real values before running anything in this package."""
    secrets_path = os.path.join(os.path.dirname(__file__), "db_secrets.json")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(
            f"{secrets_path} not found -- copy db_secrets.json.example to "
            "db_secrets.json and fill in host/port/user/password."
        )
    with open(secrets_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_db_connection():
    """Connect using host/port/dbname/user/password, all read from
    db_secrets.json (gitignored). config.json (git-tracked) only carries
    non-connection settings (content_unit_type, batch_size, etc.) and is
    not needed here."""
    secrets = _load_secrets()
    try:
        return psycopg2.connect(
            host=secrets["host"],
            port=secrets["port"],
            dbname=secrets["dbname"],
            user=secrets["user"],
            password=secrets.get("password"),
        )
    except Exception as e:
        logger.error(f"Failed to connect to {secrets['host']}:{secrets['port']}/{secrets['dbname']}: {e}")
        raise


def explore_schema(conn) -> None:
    """Print column names/types for content_units and files. Kept for
    diagnosing schema drift -- the content_units.type_id / content_types
    join documented at the top of this file has already been confirmed
    against a live connection."""
    with conn.cursor() as cur:
        for table in ("content_units", "files"):
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = %s ORDER BY ordinal_position",
                (table,),
            )
            print(f"--- {table} ---")
            for name, dtype in cur.fetchall():
                print(f"  {name}: {dtype}")


def count_source_audio_units(conn, cu_type: str, file_type: str) -> int:
    with conn.cursor() as cur:
        cur.execute(QUERY_COUNT_SOURCE_AUDIO, {"cu_type": cu_type, "file_type": file_type})
        row = cur.fetchone()
        return row[0] if row else 0


def fetch_source_audio_batch(
    conn, cu_type: str, file_type: str, offset: int, limit: int
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            QUERY_SOURCE_AUDIO,
            {"cu_type": cu_type, "file_type": file_type, "offset": offset, "limit": limit},
        )
        rows = cur.fetchall()
    return [{"cu_id": r[0], "cu_uid": r[1], "file_id": r[2], "file_uid": r[3]} for r in rows]


def fetch_text_files_for_unit(conn, cu_id: int) -> list[dict]:
    """Companion text file(s) for the same content unit, if any -- used to
    get the reference text that goes with a SOURCE unit's audio.

    Restricted to .doc/.docx files with removed_at IS NULL: SOURCE units
    were found (live) to carry multiple text-type files per language --
    old revisions with removed_at set (superseded), and occasionally a
    non-.doc format alongside the .docx source document. Filtering to the
    current .doc/.docx gives exactly one row per language in every case
    checked."""
    with conn.cursor() as cur:
        cur.execute(QUERY_TEXT_FOR_UNIT, {"cu_id": cu_id})
        rows = cur.fetchall()
    return [{"uid": r[0], "language": r[1]} for r in rows]

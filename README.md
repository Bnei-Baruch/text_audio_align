# text_audio_align

Force-align a known reference text to a long (5-60 min) audio recording of
someone reading it aloud, producing an SRT file with precise word-grouped
timestamps. Standalone project — not part of `whisper_trainer`, though it
can feed that project's `dataset_build.build` (which already consumes
CSV+SRT).

## Why this design

A single global forced-alignment pass (Viterbi/CTC) over a long recording is
risky: if the model stumbles anywhere (a misread word, noise, or — in the
future — a spoken aside not in the script), the error can drift and corrupt
alignment for everything downstream in that pass. This is not theoretical:
practitioners aligning Hebrew read-aloud audio (e.g. Torah readings) have
reported exactly this failure mode with wav2vec2-CTC alignment.

The pipeline avoids this by never running forced alignment on more than a
short window (~30s) at a time, and by first figuring out *which* part of the
reference text belongs to each window before doing precise alignment:

1. **`rough_transcribe`** — a normal (non-forced) Whisper pass over the
   whole file, chunked into ~30s windows. Produces approximate
   `(start, end, text)` segments. The text only needs to be roughly right —
   it is a search key, not the final output.
2. **`text_match`** — locates each rough segment inside the known reference
   text. Assumes monotonic reading order (no jumping back and forth in the
   script), so instead of a full-document fuzzy search per segment, a cursor
   tracks how far into the reference text the previous segment reached, and
   each new segment is searched only in a lookahead window from there. A
   segment that doesn't match well enough is left unmatched rather than
   forced.
3. **`ctc_align`** — precise CTC forced alignment (Meta's MMS model via
   `torchaudio.pipelines.MMS_FA`) of each matched window's audio against the
   *actual* reference text for that window (not the rough Whisper
   hypothesis). Produces per-word timestamps + a confidence score per word.
   Operating on short windows bounds any alignment drift to that window.
4. **`srt_assemble`** — groups the aligned words into subtitle-style cues
   and writes an SRT file. Cues containing a low-confidence word are written
   to a separate QC report instead of being silently accepted.

### Current scope: pure reading only

`text_match`'s unmatched segments are currently treated as a data problem
(flag the file for review) — the pipeline assumes the whole recording is a
straight reading of the reference text. Handling recordings with
conversational insertions/asides is planned as a follow-up: the same
unmatched-segment signal already produced by `text_match` is what that would
key off (treat as "skip this segment, don't advance/rewind the cursor"
instead of "this file is broken"), so no architecture change is expected —
just a change in how unmatched segments are handled downstream.

## Setup

Run on a machine with a GPU (this downloads and runs both a Whisper model
and Meta's MMS alignment model — do not expect this to run practically on
CPU for a 5-60 min file).

```bash
pip install -r requirements.txt
```

**Before relying on this for real data**, verify `ctc_align.py` against your
installed `torchaudio` version — its `torchaudio.pipelines.MMS_FA` API
(`get_model`/`get_tokenizer`/`get_aligner`, and the `TokenSpan` fields used
for scoring) has changed across torchaudio releases before. Smoke test:

```bash
python3 -c "
import torchaudio
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
print('MMS_FA loaded OK, sample_rate =', bundle.sample_rate)
"
```

Also verify `uroman`'s Python API (`ur.Uroman().romanize_string(text,
lcode="heb")`) matches the installed package version — this project was
written against the documented API shape but wasn't executed against a real
install in this environment.

## Step 0: fetching data from the mdb database (optional)

If your text+audio pairs live in the `mdb` Postgres database (as SOURCE
content units with a linked audio file) rather than as local files already,
`fetch_source_audio.py` finds them and downloads both the audio and its
companion text into `data/<content_unit_uid>/`.

**Schema (confirmed against a live connection)**: `content_units` has no
plain string `type` column — it has `type_id` (bigint FK into
`content_types(id, name)`, where `name` holds values like `'SOURCE'`,
`'LECTURE'`, etc. — `'SOURCE'` is `content_types.id = 47`). `files.type`,
by contrast, *is* a plain string column (`'audio'`, `'text'`, ...),
matching the sibling project `trlAi/src/prepare_data.py`'s direct
`f.type = 'text'` comparison. `fetch_data/db.py`'s queries join
`content_units` to `content_types` accordingly; the count query was tested
live and returned 1352 SOURCE content units with an audio file. If the
schema changes, `explore_schema()` is still there for diagnosing drift:

```bash
python3 -c "
from fetch_data.db import get_db_connection, explore_schema
conn = get_db_connection()
explore_schema(conn)
"
```

**Credentials**: all connection details (`host`, `port`, `dbname`, `user`,
`password`) live in `fetch_data/db_secrets.json`, which is gitignored and
never committed. `fetch_data/config.json` (committed) only holds
non-connection settings (`content_unit_type`, `file_type`, `batch_size`,
etc.). Before running anything in `fetch_data/`, copy the template and fill
in real values:

```bash
cp fetch_data/db_secrets.json.example fetch_data/db_secrets.json
# then edit fetch_data/db_secrets.json with the real host/port/dbname/user/password
```

This mirrors the `settings.toml` / `.secrets.toml` split already used by
the sibling `trlAi` project against the same database — except here *all*
connection fields are secret, not just the password, since `host`/`user`
are specific to your access to this DB.

**Audio URL assumption**: downloads use
`https://cdn.kabbalahmedia.info/{file_uid}.mp3`, where `{file_uid}` is the
file's `uid` string column (not its numeric `id`) — matches the `uid`
convention `trlAi/src/prepare_data.py` uses for its own kabbalahmedia URL.
If downloads 404, this is the first thing to check.

`fetch_data/config.json`'s `"limit"` caps how many content units to fetch
(`null` = no limit, fetch all matching units). `--limit N` on the command
line overrides the config value for a one-off run.

Run it:

```bash
python fetch_source_audio.py fetch_data/config.json
# or, to override the config's limit for a quick test:
python fetch_source_audio.py fetch_data/config.json --limit 10
```

## Usage

1. Either run step 0 above, or put your own audio file and reference text
   (`.txt`, plain UTF-8) directly under `data/`.
2. Edit `align_config.json` — at minimum `audio_path` and
   `reference_text_path`.
3. Run:

```bash
python run_align.py align_config.json
```

Output: an SRT file at `output_srt_path`, and a JSON QC report at
`output_qc_path` listing every subtitle cue that contains a word below
`min_ctc_score` (default 0.5) — review these before trusting the file
downstream.

## Config reference (`align_config.json`)

| Key | Default | Meaning |
|---|---|---|
| `audio_path` | — | Path to the long recording. |
| `reference_text_path` | — | Path to the plain-text known-correct script. |
| `output_srt_path` | `output/output.srt` | Where to write the resulting SRT. |
| `output_qc_path` | `output/output_qc.json` | Where to write flagged low-confidence cues. |
| `whisper_model` | `openai/whisper-large-v3` | Model for the rough stage-1 pass. Any Hebrew-capable Whisper checkpoint works, including a fine-tuned one. |
| `language` | `he` | Passed to Whisper's generation config. |
| `mms_lang_code` | `heb` | ISO 639-3 code passed to `uroman` / MMS for the CTC stage. |
| `chunk_length_s` | `30` | Rough-pass window size (also the CTC stage's max window size). |
| `lookahead_words` | `200` | How far ahead of the cursor to search the reference text per segment. |
| `min_match_ratio` | `0.4` | Below this text-match ratio, a segment is left unmatched (excluded from alignment). |
| `min_ctc_score` | `0.5` | Below this CTC confidence, a cue is flagged in the QC report. |
| `words_per_cue` | `12` | How many aligned words to group into one SRT cue. |

## Layout

```
fetch_data/
  db.py                  # DB connection + queries (step 0)
  download.py            # audio/text download (step 0)
  config.json            # non-secret settings: content_unit_type, limit, etc. (committed)
  db_secrets.json.example  # copy to db_secrets.json (gitignored) for real host/port/dbname/user/password
fetch_source_audio.py    # step 0 CLI entry point
aligner/
  rough_transcribe.py   # stage 1
  text_match.py          # stage 2 (pure Python, no ML deps — see below)
  ctc_align.py           # stage 3
  srt_assemble.py        # stage 4
  pipeline.py            # orchestrates all 4 stages
run_align.py              # CLI entry point
align_config.json         # example config
```

`text_match.py` has no external dependencies (stdlib `difflib` only) and was
smoke-tested standalone during development — it correctly matches
well-aligned segments via a sequentially-advancing cursor and correctly
rejects an injected mismatched segment without losing its place in the
reference text for the segment after it. `rough_transcribe.py` and
`ctc_align.py` depend on downloading and running real models and have not
been executed end-to-end in this environment — validate them on your own
data before trusting the output.

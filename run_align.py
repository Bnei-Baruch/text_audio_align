#!/usr/bin/env python
"""CLI entry point.

Batch-runs the 4-stage alignment pipeline over content units under
data_dir -- the layout fetch_source_audio.py produces: one subdirectory
per unit, named by its uid, containing a single <file_uid>.wav and a
reference_text.txt. By default every subdirectory of data_dir is
processed; set the config's "unit_ids" to a list of uids to restrict the
run to just those units.

Usage:
    python run_align.py align_config.json
"""
import argparse
import glob
import json
import os
import shutil

from aligner.pipeline import run_pipeline

_UNIT_CFG_KEYS = ("data_dir", "output_dir", "unit_ids", "audio_path", "reference_text_path",
                  "output_srt_path", "output_qc_path")


def _find_unit_files(unit_dir: str) -> tuple[str, str] | None:
    wavs = glob.glob(os.path.join(unit_dir, "*.wav"))
    text_path = os.path.join(unit_dir, "reference_text.txt")
    if len(wavs) != 1 or not os.path.exists(text_path):
        return None
    return wavs[0], text_path


def run_unit(unit_id: str, audio_path: str, text_path: str, base_cfg: dict, output_dir: str) -> dict:
    unit_output_dir = os.path.join(output_dir, unit_id)
    os.makedirs(unit_output_dir, exist_ok=True)

    cfg = {
        **base_cfg,
        "audio_path": audio_path,
        "reference_text_path": text_path,
        "output_srt_path": os.path.join(unit_output_dir, "output.srt"),
        "output_qc_path": os.path.join(unit_output_dir, "output_qc.json"),
    }
    result = run_pipeline(cfg)

    shutil.copy2(audio_path, os.path.join(unit_output_dir, os.path.basename(audio_path)))
    shutil.copy2(text_path, os.path.join(unit_output_dir, "reference_text.txt"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_dir = cfg.get("data_dir", "data")
    output_dir = cfg.get("output_dir", "output")
    unit_ids = cfg.get("unit_ids")
    base_cfg = {k: v for k, v in cfg.items() if k not in _UNIT_CFG_KEYS}

    if unit_ids is None:
        unit_ids = sorted(os.listdir(data_dir))

    results = {}
    for unit_id in unit_ids:
        unit_dir = os.path.join(data_dir, unit_id)
        if not os.path.isdir(unit_dir):
            print(f"[{unit_id}] skipping -- no such directory under {data_dir!r}")
            continue

        found = _find_unit_files(unit_dir)
        if found is None:
            print(f"[{unit_id}] skipping -- expected exactly one .wav and a reference_text.txt")
            continue
        audio_path, text_path = found

        print(f"=== {unit_id} ===")
        try:
            results[unit_id] = run_unit(unit_id, audio_path, text_path, base_cfg, output_dir)
        except Exception as e:
            print(f"[{unit_id}] failed: {e}")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

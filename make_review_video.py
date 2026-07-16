#!/usr/bin/env python
"""Convert a WAV file into an MP4 with a plain black background video track
-- e.g. for uploading audio somewhere that requires a video file, or for
eyeballing forced-alignment output by burning in the generated SRT so
alignment quality can be reviewed in any standard video player.

Output path is always <audio_path stem>.mp4, next to the audio file. If
audio_path is omitted, runs over every unit dir under align_config.json's
output_dir instead (the layout run_align.py produces: one <uid>.wav +
output.srt per unit dir).

Usage:
    python make_review_video.py [audio.wav] [--srt subtitles.srt] [--config align_config.json]
"""
import argparse
import glob
import json
import os
import subprocess

# Smallest size that still keeps burned-in subtitles legible for review --
# not the true minimum (e.g. 2x2), which would make the SRT unreadable.
_RESOLUTION = "320x180"


def make_video(
    audio_path: str,
    output_path: str,
    srt_path: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={_RESOLUTION}:r=1",
        "-i", audio_path,
    ]
    if srt_path:
        # ffmpeg's subtitles filter runs its own mini-parser on this path
        # string -- colons/backslashes need escaping on top of normal
        # shell/argv escaping, or it misreads the path.
        escaped_srt = srt_path.replace("\\", "\\\\").replace(":", "\\:")
        cmd += ["-vf", f"subtitles={escaped_srt}"]
    cmd += [
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to build {output_path}: {proc.stderr.decode(errors='replace')}")


def _default_srt(audio_path: str, explicit_srt: str | None) -> str | None:
    if explicit_srt is not None:
        return explicit_srt
    candidate = os.path.join(os.path.dirname(audio_path) or ".", "output.srt")
    return candidate if os.path.exists(candidate) else None


def _process_one(audio_path: str, explicit_srt: str | None) -> str:
    output_path = os.path.splitext(audio_path)[0] + ".mp4"
    make_video(audio_path, output_path, srt_path=_default_srt(audio_path, explicit_srt))
    return output_path


def _iter_output_unit_wavs(output_dir: str):
    for unit_id in sorted(os.listdir(output_dir)):
        unit_dir = os.path.join(output_dir, unit_id)
        if not os.path.isdir(unit_dir):
            continue
        wavs = glob.glob(os.path.join(unit_dir, "*.wav"))
        if len(wavs) != 1:
            continue
        yield unit_id, wavs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=None)
    parser.add_argument(
        "--srt", default=None,
        help="SRT file to burn in as subtitles (default: output.srt next to audio_path, if it exists)",
    )
    parser.add_argument(
        "--config", default="align_config.json",
        help="Config to read output_dir from when audio_path is omitted (default: align_config.json)",
    )
    args = parser.parse_args()

    if args.audio_path is not None:
        output_path = _process_one(args.audio_path, args.srt)
        print(f"Saved: {output_path}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        output_dir = json.load(f).get("output_dir", "output")

    for unit_id, audio_path in _iter_output_unit_wavs(output_dir):
        try:
            output_path = _process_one(audio_path, args.srt)
            print(f"[{unit_id}] saved: {output_path}")
        except Exception as e:
            print(f"[{unit_id}] failed: {e}")


if __name__ == "__main__":
    main()

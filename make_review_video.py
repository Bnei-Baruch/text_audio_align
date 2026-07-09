#!/usr/bin/env python
"""Convert a WAV file into an MP4 with a plain black background video track
-- e.g. for uploading audio somewhere that requires a video file, or for
eyeballing forced-alignment output by burning in the generated SRT so
alignment quality can be reviewed in any standard video player.

Usage:
    python make_review_video.py <audio.wav> [output.mp4] [--srt subtitles.srt] [--resolution 640x360]
"""
import argparse
import os
import subprocess


def make_video(
    audio_path: str,
    output_path: str,
    srt_path: str | None = None,
    resolution: str = "640x360",
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={resolution}:r=1",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("output_path", nargs="?", default=None)
    parser.add_argument("--srt", default=None, help="Optional SRT file to burn in as subtitles")
    parser.add_argument("--resolution", default="640x360")
    args = parser.parse_args()

    output_path = args.output_path or os.path.splitext(args.audio_path)[0] + ".mp4"
    make_video(args.audio_path, output_path, srt_path=args.srt, resolution=args.resolution)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

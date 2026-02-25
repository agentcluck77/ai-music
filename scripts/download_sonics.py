#!/usr/bin/env python3
"""Download the SONICS dataset and the referenced real songs.

- Dataset: awsaf49/sonics (Hugging Face Datasets)
- Output: /mnt/aloy/sonics by default
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")  # basic unquote
        os.environ.setdefault(key, value)


def read_youtube_ids(real_csv: Path) -> List[str]:
    if not real_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {real_csv}")

    with real_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "youtube_id" not in reader.fieldnames:
            raise ValueError(f"Expected column 'youtube_id' in {real_csv}")
        ids = [row["youtube_id"].strip() for row in reader if row.get("youtube_id")]
    return [i for i in ids if i]


def run_yt_dlp(
    youtube_id: str,
    output_path: Path,
    audio_format: str,
    archive_path: Path,
    extract_audio: bool,
    js_runtime: str | None,
    retries: int,
    fragment_retries: int,
    extractor_retries: int,
    socket_timeout: int,
    log_file: Path,
) -> bool:
    if output_path.exists():
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp",
        url,
        "-c",
        "--retries",
        str(retries),
        "--fragment-retries",
        str(fragment_retries),
        "--extractor-retries",
        str(extractor_retries),
        "--socket-timeout",
        str(socket_timeout),
        "--download-archive",
        str(archive_path),
        "-o",
        str(output_path),
    ]
    if js_runtime:
        cmd.extend(["--js-runtimes", js_runtime])
    if extract_audio:
        cmd.extend(["-x", "--audio-format", audio_format])
    else:
        cmd.extend(["-f", "bestaudio"])
    try:
        with log_file.open("w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd,
                stdout=logf,
                stderr=logf,
                check=False,
            )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp not found. Install it in the venv first.") from exc

    return result.returncode == 0 and output_path.exists()


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/mnt/aloy/sonics",
        help="Output directory for the SONICS dataset",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. If omitted, uses env HF_TOKEN if set.",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        help="Audio format for yt-dlp extraction",
    )
    parser.add_argument(
        "--extract-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Extract audio with ffmpeg (default: auto-detect ffmpeg).",
    )
    parser.add_argument(
        "--hf-retries",
        type=int,
        default=5,
        help="Retries for Hugging Face dataset download",
    )
    parser.add_argument(
        "--hf-retry-wait",
        type=int,
        default=10,
        help="Seconds to wait between Hugging Face retries",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Progress batching size for logging",
    )
    parser.add_argument(
        "--yt-retries",
        type=int,
        default=10,
        help="yt-dlp retries for each video",
    )
    parser.add_argument(
        "--yt-fragment-retries",
        type=int,
        default=10,
        help="yt-dlp fragment retries",
    )
    parser.add_argument(
        "--yt-extractor-retries",
        type=int,
        default=3,
        help="yt-dlp extractor retries",
    )
    parser.add_argument(
        "--yt-socket-timeout",
        type=int,
        default=30,
        help="yt-dlp socket timeout (seconds)",
    )
    parser.add_argument(
        "--yt-js-runtime",
        default=None,
        help="yt-dlp JS runtime (e.g., node, node:/usr/bin/node, deno). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--yt-archive",
        default=None,
        help="Path to yt-dlp archive file (defaults to <output>/yt_dlp_archive.txt)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(Path(".env"))
    print(f"Downloading SONICS dataset into: {base_dir}")
    last_err: Exception | None = None
    for attempt in range(1, args.hf_retries + 1):
        try:
            snapshot_download(
                repo_id="awsaf49/sonics",
                repo_type="dataset",
                local_dir=str(base_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=args.hf_token or os.environ.get("HF_TOKEN"),
            )
            last_err = None
            break
        except Exception as exc:  # network/rate-limit errors
            last_err = exc
            print(f"HF download failed (attempt {attempt}/{args.hf_retries}): {exc}")
            if attempt < args.hf_retries:
                time.sleep(args.hf_retry_wait)
    if last_err is not None:
        raise last_err

    real_csv = base_dir / "real_songs.csv"
    real_dir = base_dir / "real_songs"
    real_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "yt_dlp_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_path = (
        Path(args.yt_archive).expanduser().resolve()
        if args.yt_archive
        else base_dir / "yt_dlp_archive.txt"
    )

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if args.extract_audio is None:
        extract_audio = ffmpeg_available
    else:
        extract_audio = args.extract_audio
    if extract_audio and not ffmpeg_available:
        print("ffmpeg not found; disabling audio extraction.")
        extract_audio = False

    js_runtime = args.yt_js_runtime
    if js_runtime is None:
        node_path = shutil.which("node")
        nodejs_path = shutil.which("nodejs")
        deno_path = shutil.which("deno")
        bun_path = shutil.which("bun")
        quickjs_path = shutil.which("qjs") or shutil.which("quickjs")
        if node_path:
            js_runtime = f"node:{node_path}"
        elif nodejs_path:
            js_runtime = f"node:{nodejs_path}"
        elif deno_path:
            js_runtime = f"deno:{deno_path}"
        elif bun_path:
            js_runtime = f"bun:{bun_path}"
        elif quickjs_path:
            js_runtime = f"quickjs:{quickjs_path}"

    youtube_ids = read_youtube_ids(real_csv)
    total = len(youtube_ids)
    print(f"Found {total} YouTube IDs")

    successes: List[str] = []
    failures: List[str] = []

    for batch in chunked(youtube_ids, args.batch_size):
        for youtube_id in tqdm(batch, desc="Downloading real songs", unit="song"):
            suffix = args.audio_format if extract_audio else "bestaudio"
            output_path = real_dir / f"{youtube_id}.{suffix}"
            log_file = logs_dir / f"{youtube_id}.log"
            ok = run_yt_dlp(
                youtube_id,
                output_path,
                args.audio_format,
                archive_path,
                extract_audio,
                js_runtime,
                args.yt_retries,
                args.yt_fragment_retries,
                args.yt_extractor_retries,
                args.yt_socket_timeout,
                log_file,
            )
            if ok:
                successes.append(youtube_id)
            else:
                failures.append(youtube_id)
        print(
            f"Progress: {len(successes)} ok, {len(failures)} failed, {total} total"
        )

    report_path = base_dir / "real_songs_download_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Success: {len(successes)}\n")
        f.write(f"Failed: {len(failures)}\n")
        if failures:
            f.write("\nFailed IDs:\n")
            for youtube_id in failures:
                f.write(f"{youtube_id}\n")

    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

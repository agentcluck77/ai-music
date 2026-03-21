# SONICS Dataset Downloader

Download the SONICS dataset metadata from Hugging Face and download/extract referenced real-song audio from YouTube.

## Repo Contents

- `scripts/setup_uv_venv.sh`: create `.venv` with `uv` and install Python deps.
- `scripts/download_sonics.py`: main downloader (dataset + real songs).
- `requirements.txt`: Python dependencies.
- `SC3021_Lab_1.ipynb`: reference notebook.
- `sonics.md`: source links.

## Prerequisites

- `uv`
- `node` (recommended JS runtime for `yt-dlp`)
- `ffmpeg` (required when using `--extract-audio`)

Example (Debian/Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y nodejs ffmpeg
```

## Quick Start

1. (Optional) add Hugging Face token in `.env`:

```bash
echo "HF_TOKEN=your_token_here" > .env
```

2. Create venv and install deps:

```bash
./scripts/setup_uv_venv.sh
```

3. Run downloader (default output is `/mnt/aloy/sonics`):

```bash
uv run scripts/download_sonics.py --extract-audio --yt-js-runtime node:/usr/bin/node
```

## Resume and Interruption Behavior

- Already downloaded songs are skipped.
- Existing output files and `yt_dlp_archive.txt` are reused.
- If interrupted, rerun the same command; it continues from current state.

## Output Layout

Written under `/mnt/aloy/sonics` by default:

- `fake_songs.csv`, `real_songs.csv`, etc. (dataset files)
- `real_songs/` extracted audio files (`.mp3` by default)
- `yt_dlp_logs/` one log per YouTube ID
- `yt_dlp_archive.txt` download archive for `yt-dlp`
- `real_songs_download_report.txt` progress summary

## Check Progress

```bash
grep -E '^(Total|Processed|Success|Failed|Interrupted):' /mnt/aloy/sonics/real_songs_download_report.txt
```

## Notes

- Some IDs will fail (removed/private/age-restricted/regional videos). This is expected.
- To disable extraction and download best available audio stream as-is:

```bash
uv run scripts/download_sonics.py --no-extract-audio
```

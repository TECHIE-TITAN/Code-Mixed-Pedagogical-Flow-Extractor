"""
pipeline/ingest.py — Ingestion Agent
Downloads audio from a video URL using yt-dlp, converts to 16kHz mono WAV
using ffmpeg. Returns the path to the audio file in the pipeline state.
"""

import subprocess
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_AUDIO_DIR, AUDIO_FORMAT, AUDIO_SAMPLE_RATE
from state import PipelineState

logger = logging.getLogger(__name__)


def ingest_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Download audio from video URL and convert to WAV.

    Input state keys used:  video_id, video_url
    Output state keys set:  audio_path
    """
    video_id = state["video_id"]
    video_url = state["video_url"]

    audio_path = RAW_AUDIO_DIR / f"{video_id}.{AUDIO_FORMAT}"

    if audio_path.exists():
        logger.info(f"[ingest] Audio already exists for {video_id}, skipping download.")
        return {**state, "audio_path": str(audio_path)}

    logger.info(f"[ingest] Downloading audio for video_id={video_id} from {video_url}")

    # ── Step 1: Download best audio with yt-dlp ─────────────────────────────
    temp_path = RAW_AUDIO_DIR / f"{video_id}_raw"
    ydlp_cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(temp_path) + ".%(ext)s",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        video_url,
    ]
    logger.info(f"[ingest] Running: {' '.join(ydlp_cmd)}")
    result = subprocess.run(ydlp_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"[ingest] yt-dlp failed for {video_url}:\n{result.stderr}"
        )

    # yt-dlp may produce .wav or .webm etc — find what it produced
    raw_files = list(RAW_AUDIO_DIR.glob(f"{video_id}_raw.*"))
    if not raw_files:
        raise FileNotFoundError(
            f"[ingest] yt-dlp ran but produced no output file in {RAW_AUDIO_DIR}"
        )
    raw_file = raw_files[0]
    logger.info(f"[ingest] Downloaded: {raw_file}")

    # ── Step 2: Convert to 16kHz mono WAV with ffmpeg ───────────────────────
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                           # overwrite if exists
        "-i", str(raw_file),
        "-ar", str(AUDIO_SAMPLE_RATE),  # 16000 Hz
        "-ac", "1",                     # mono
        "-c:a", "pcm_s16le",            # standard PCM WAV
        str(audio_path),
    ]
    logger.info(f"[ingest] Converting audio: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"[ingest] ffmpeg conversion failed:\n{result.stderr}")

    # Clean up temp file if it's different from the output
    if raw_file != audio_path and raw_file.exists():
        raw_file.unlink()

    logger.info(f"[ingest] Audio ready: {audio_path}")
    return {**state, "audio_path": str(audio_path)}

"""
pipeline/transcribe.py — Transcription Agent
Uses faster-whisper to produce a raw text transcript + timestamped segments.
Caches results to disk so re-runs don't redo expensive ASR.
"""

import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TRANSCRIPTS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    CACHE_DIR,
)
from state import PipelineState

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Transcribe audio using faster-whisper.

    Input state keys:  video_id, audio_path
    Output state keys: raw_transcript, timestamped_segments
    """
    from faster_whisper import WhisperModel  # lazy import — only loaded when needed

    video_id = state["video_id"]
    audio_path = state["audio_path"]

    # ── Cache check ─────────────────────────────────────────────────────────
    cache_file = CACHE_DIR / f"{video_id}_transcript.json"
    if cache_file.exists():
        logger.info(f"[transcribe] Loading cached transcript for {video_id}")
        cached = json.loads(cache_file.read_text())
        return {
            **state,
            "raw_transcript": cached["raw_transcript"],
            "timestamped_segments": cached["timestamped_segments"],
        }

    logger.info(
        f"[transcribe] Loading Whisper model: {WHISPER_MODEL_SIZE} "
        f"on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})"
    )
    logger.info(
        "  ⏳ This will take a few minutes on CPU. "
        "Model downloads on first run (~1.5 GB for 'medium')."
    )

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    logger.info(f"[transcribe] Transcribing: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,         # None = auto-detect
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,                    # Voice Activity Detection — skips silence
        vad_parameters=dict(
            min_silence_duration_ms=500,    # merge short pauses
        ),
    )

    logger.info(
        f"[transcribe] Detected language: {info.language} "
        f"(probability={info.language_probability:.2f})"
    )

    timestamped_segments = []
    full_text_parts = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        entry = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "start_fmt": _format_timestamp(seg.start),
            "end_fmt": _format_timestamp(seg.end),
            "text": text,
        }
        timestamped_segments.append(entry)
        full_text_parts.append(text)

    raw_transcript = " ".join(full_text_parts)

    # ── Persist transcript to disk ───────────────────────────────────────────
    transcript_txt = TRANSCRIPTS_DIR / f"{video_id}_raw.txt"
    transcript_json = TRANSCRIPTS_DIR / f"{video_id}_segments.json"
    transcript_txt.write_text(raw_transcript, encoding="utf-8")
    transcript_json.write_text(
        json.dumps(timestamped_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Write cache ──────────────────────────────────────────────────────────
    cache_file.write_text(
        json.dumps(
            {
                "raw_transcript": raw_transcript,
                "timestamped_segments": timestamped_segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        f"[transcribe] Done. {len(timestamped_segments)} segments, "
        f"{len(raw_transcript.split())} words."
    )

    return {
        **state,
        "raw_transcript": raw_transcript,
        "timestamped_segments": timestamped_segments,
    }

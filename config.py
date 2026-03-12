"""
config.py — Central configuration for the pipeline.
All tunable parameters and paths live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUTS_DIR = DATA_DIR / "outputs"
GLOSSARY_PATH = BASE_DIR / "glossary.json"
VIDEOS_YAML_PATH = BASE_DIR / "videos.yaml"
CACHE_DIR = BASE_DIR / ".cache"

# Ensure directories exist
for d in [RAW_AUDIO_DIR, TRANSCRIPTS_DIR, OUTPUTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LLM SETTINGS
# ──────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# LLM_PROVIDER: "openai" or "gemini"
# Switch to "gemini" if OpenAI quota is exhausted — Gemini 1.5 Flash is FREE
LLM_PROVIDER = "gemini" if GEMINI_API_KEY and not OPENAI_API_KEY else "openai"

# OpenAI models (used when LLM_PROVIDER = "openai")
LLM_FAST_MODEL = "gpt-4o-mini"          # normalization, glossary expansion
LLM_SMART_MODEL = "gpt-4o"             # concept extraction, prerequisite mapping, validation critique

# Gemini models (used when LLM_PROVIDER = "gemini")
# Note: gemini-1.5-flash and gemini-2.0-flash have 0 quota on free tier keys;
# gemini-2.5-flash-lite is the available free-tier model on this key.
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"   # free tier, confirmed working
GEMINI_SMART_MODEL = "gemini-2.5-flash-lite"  # same — use for all tasks

LLM_TEMPERATURE = 0.1                   # low temp for structured/factual tasks
LLM_MAX_RETRIES = 3

# ──────────────────────────────────────────────
# WHISPER SETTINGS
# ──────────────────────────────────────────────
# On your machine (no GPU, 16 GB RAM):
#   "tiny"   — fastest, least accurate
#   "base"   — good balance for demos
#   "small"  — recommended: decent code-mix accuracy, fits in RAM
#   "medium" — best CPU-feasible accuracy (~3–4 GB RAM peak)
#   "large-v3" — requires 10+ GB VRAM, DO NOT use on this machine without GPU
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"                  # "cuda" if GPU available
WHISPER_COMPUTE_TYPE = "int8"           # "int8" for CPU (fastest), "float16" for GPU
WHISPER_LANGUAGE = None                 # None = auto-detect per file (best for code-mix)
WHISPER_BEAM_SIZE = 5

# ──────────────────────────────────────────────
# PIPELINE SETTINGS
# ──────────────────────────────────────────────
MAX_VALIDATION_RETRIES = 3              # max times validation loop can retry extraction
CONCEPT_CHUNK_TOKENS = 800             # transcript chunk size for concept extraction
MIN_PREREQUISITE_EDGES = 2             # minimum edges required to pass validation
CONFIDENCE_THRESHOLD = 0.6            # minimum edge confidence to keep

# ──────────────────────────────────────────────
# AUDIO SETTINGS
# ──────────────────────────────────────────────
AUDIO_FORMAT = "wav"                    # wav works best with whisper
AUDIO_SAMPLE_RATE = 16000              # 16kHz mono is what whisper expects

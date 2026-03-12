"""
pipeline/normalize.py — Normalization Agent
Detects code-mixed segments in the transcript and maps colloquial Indic /
code-mixed terms to their standard English academic equivalents using GPT.
Also expands the shared glossary.json file with new findings.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_MAX_RETRIES,
    GLOSSARY_PATH,
    CACHE_DIR,
    TRANSCRIPTS_DIR,
)
from pipeline.llm_client import get_completion
from state import PipelineState

logger = logging.getLogger(__name__)

# ── Seed glossary (expanded by LLM at runtime) ────────────────────────────────
DEFAULT_GLOSSARY: Dict[str, str] = {
    # Hinglish CS / Math seeds
    "function": "Function",
    "recursion wala concept": "Recursion",
    "loop lagao": "Loop iteration",
    "array ka size": "Array length",
    "dhara": "Electric current",
    "bel": "Bell curve / Normal distribution",
    "uska matlab": "that means",
    "isliye": "therefore",
    "seedha": "directly / linear",
    "ulta": "inverse / reverse",
    "matlab": "meaning",
    "samajh lo": "understand that",
    # Telugu-English seeds
    "function ante": "Function means",
    "loop esthe": "if we apply a loop",
    "variable petti": "by setting a variable",
    # Tamil-English seeds
    "function nu solvom": "we call it a function",
    "input kuduthal": "if we give input",
}


def _load_glossary() -> Dict[str, str]:
    """Load glossary from disk, merging with defaults."""
    if GLOSSARY_PATH.exists():
        disk_glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_GLOSSARY, **disk_glossary}
    return DEFAULT_GLOSSARY.copy()


def _save_glossary(glossary: Dict[str, str]) -> None:
    """Persist glossary to disk."""
    GLOSSARY_PATH.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _chunk_text(text: str, max_words: int = 300) -> List[str]:
    """Split text into chunks of ~max_words words, preserving sentence boundaries."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], []
    current_words = 0
    for sent in sentences:
        words = len(sent.split())
        if current_words + words > max_words and current:
            chunks.append(". ".join(current) + ".")
            current, current_words = [], 0
        current.append(sent)
        current_words += words
    if current:
        chunks.append(". ".join(current))
    return chunks


def _normalize_chunk(
    chunk: str,
    glossary: Dict[str, str],
    language_mix: List[str],
) -> tuple[str, List[dict]]:
    """
    Call LLM to normalize code-mixed text in a single chunk.
    Returns (normalized_chunk, new_glossary_entries).
    """
    glossary_str = json.dumps(glossary, ensure_ascii=False, indent=2)
    language_str = " + ".join(language_mix)

    system_prompt = f"""You are an expert in {language_str} code-mixed language normalization for educational content.
Your job is to:
1. Replace ALL code-mixed, colloquial, or Indic-language phrases with their standard English academic equivalents.
2. Keep the meaning and sentence structure intact.
3. Do NOT translate English words that are already standard academic terms.
4. Return a JSON object with two keys:
   - "normalized_text": the cleaned English version of the input text
   - "new_glossary_entries": a list of {{"raw": "...", "normalized": "...", "domain": "..."}} objects for any NEW mappings you made that aren't already in the glossary.

Existing glossary (do not re-add these):
{glossary_str}

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation."""

    user_prompt = f"Normalize this code-mixed educational transcript chunk:\n\n{chunk}"

    try:
        raw = get_completion(system_prompt, user_prompt, mode="fast")
        data = json.loads(raw)
        normalized = data.get("normalized_text", chunk)
        new_entries = data.get("new_glossary_entries", [])
        return normalized, new_entries
    except Exception as e:
        logger.error(f"[normalize] Chunk normalization failed: {e}. Returning original.")
        return chunk, []


def normalize_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Normalize code-mixed transcript using LLM.

    Input state keys:  video_id, raw_transcript, language_mix
    Output state keys: normalized_transcript, glossary_additions
    """

    video_id = state["video_id"]
    raw_transcript = state["raw_transcript"]
    language_mix = state.get("language_mix", ["Hindi", "English"])

    # ── Cache check ─────────────────────────────────────────────────────────
    cache_key = hashlib.md5(raw_transcript.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"{video_id}_normalized_{cache_key}.json"

    if cache_file.exists():
        logger.info(f"[normalize] Using cached normalization for {video_id}")
        cached = json.loads(cache_file.read_text())
        return {
            **state,
            "normalized_transcript": cached["normalized_transcript"],
            "glossary_additions": cached["glossary_additions"],
        }

    glossary = _load_glossary()

    chunks = _chunk_text(raw_transcript, max_words=300)
    logger.info(f"[normalize] Processing {len(chunks)} chunks for {video_id}")

    normalized_parts = []
    all_new_entries = []

    for i, chunk in enumerate(chunks):
        logger.info(f"[normalize] Chunk {i + 1}/{len(chunks)} ...")
        norm_chunk, new_entries = _normalize_chunk(chunk, glossary, language_mix)
        normalized_parts.append(norm_chunk)
        all_new_entries.extend(new_entries)
        # Update in-memory glossary so subsequent chunks benefit
        for entry in new_entries:
            glossary[entry.get("raw", "")] = entry.get("normalized", "")

    normalized_transcript = " ".join(normalized_parts)

    # ── Persist ──────────────────────────────────────────────────────────────
    _save_glossary(glossary)
    norm_path = TRANSCRIPTS_DIR / f"{video_id}_normalized.txt"
    norm_path.write_text(normalized_transcript, encoding="utf-8")

    cache_file.write_text(
        json.dumps(
            {
                "normalized_transcript": normalized_transcript,
                "glossary_additions": all_new_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        f"[normalize] Done. {len(all_new_entries)} new glossary entries added."
    )

    return {
        **state,
        "normalized_transcript": normalized_transcript,
        "glossary_additions": all_new_entries,
    }

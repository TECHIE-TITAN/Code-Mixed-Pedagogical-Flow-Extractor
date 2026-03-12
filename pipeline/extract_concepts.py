"""
pipeline/extract_concepts.py — Concept Extraction Agent
Chunks the normalized transcript and calls GPT to extract structured
ConceptNode objects. Deduplicates concepts using sentence-transformer
embeddings to catch near-duplicates with different phrasing.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CONCEPT_CHUNK_TOKENS,
    CACHE_DIR,
)
from pipeline.llm_client import get_completion
from state import PipelineState, ConceptNode

logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert educational content analyst specializing in technical concept extraction.
Given a normalized educational transcript chunk, identify every distinct technical concept taught.

For each concept, return:
- "id": a short snake_case identifier (e.g., "recursion", "binary_search")
- "raw_term": the exact phrase used in the transcript (may be informal)
- "normalized_term": the standard academic English term
- "domain": subject area (e.g., "Computer Science", "Physics", "Mathematics")
- "first_mentioned_at": timestamp "HH:MM:SS" if available in the text, else ""
- "description": one concise sentence describing what this concept is

Return a JSON object: {"concepts": [...]}
IMPORTANT: Return ONLY valid JSON. No markdown, no explanation outside the JSON."""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token."""
    return int(len(text.split()) / 0.75)


def _chunk_transcript_with_timestamps(
    segments: List[dict], max_tokens: int = 800
) -> List[Dict]:
    """
    Group timestamped segments into chunks of ~max_tokens.
    Returns list of {"text": ..., "start_fmt": ..., "end_fmt": ...}
    """
    chunks = []
    current_texts = []
    current_tokens = 0
    chunk_start = None

    for seg in segments:
        tokens = _estimate_tokens(seg["text"])
        if current_tokens + tokens > max_tokens and current_texts:
            chunks.append({
                "text": " ".join(current_texts),
                "start_fmt": chunk_start,
                "end_fmt": seg.get("start_fmt", ""),
            })
            current_texts, current_tokens = [], 0
            chunk_start = None

        if chunk_start is None:
            chunk_start = seg.get("start_fmt", "")
        current_texts.append(seg["text"])
        current_tokens += tokens

    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "start_fmt": chunk_start,
            "end_fmt": segments[-1].get("end_fmt", "") if segments else "",
        })
    return chunks


def _deduplicate_concepts(concepts: List[ConceptNode]) -> List[ConceptNode]:
    """
    Remove near-duplicate concepts using sentence-transformer cosine similarity.
    Falls back to exact-string dedup if sentence-transformers not available.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch

        model = SentenceTransformer("all-MiniLM-L6-v2")
        terms = [c["normalized_term"] for c in concepts]
        embeddings = model.encode(terms, convert_to_tensor=True)

        keep = []
        kept_indices = []
        for i, concept in enumerate(concepts):
            is_dup = False
            for j in kept_indices:
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > 0.88:  # ~88% similarity → treat as duplicate
                    is_dup = True
                    break
            if not is_dup:
                keep.append(concept)
                kept_indices.append(i)
        logger.info(f"[extract] Dedup: {len(concepts)} → {len(keep)} concepts")
        return keep

    except ImportError:
        logger.warning(
            "[extract] sentence-transformers not available, using exact dedup."
        )
        seen = set()
        unique = []
        for c in concepts:
            key = c["normalized_term"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique


def _extract_from_chunk(chunk: dict, video_id_prefix: str) -> List[ConceptNode]:
    """Call LLM on a single chunk and return ConceptNode list."""
    text_with_ts = (
        f"[Timestamp: {chunk['start_fmt']} – {chunk['end_fmt']}]\n{chunk['text']}"
        if chunk.get("start_fmt")
        else chunk["text"]
    )

    try:
        raw = get_completion(SYSTEM_PROMPT, text_with_ts, mode="smart")
        data = json.loads(raw)
        raw_concepts = data.get("concepts", [])

        for i, c in enumerate(raw_concepts):
            if not c.get("id"):
                c["id"] = f"{video_id_prefix}_{i}"
            c.setdefault("raw_term", c.get("normalized_term", "unknown"))
            c.setdefault("normalized_term", c.get("raw_term", "unknown"))
            c.setdefault("domain", "General")
            c.setdefault("first_mentioned_at", chunk.get("start_fmt", ""))
            c.setdefault("description", "")

        return raw_concepts
    except Exception as e:
        logger.error(f"[extract] Chunk extraction failed: {e}")
        return []


def extract_concepts_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Extract technical concepts from normalized transcript.

    Input state keys:  video_id, normalized_transcript, timestamped_segments
    Output state keys: concepts
    """
    video_id = state["video_id"]
    normalized_transcript = state["normalized_transcript"]
    timestamped_segments = state.get("timestamped_segments", [])

    # ── Cache check ─────────────────────────────────────────────────────────
    cache_key = hashlib.md5(normalized_transcript.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"{video_id}_concepts_{cache_key}.json"

    if cache_file.exists():
        logger.info(f"[extract] Using cached concepts for {video_id}")
        cached = json.loads(cache_file.read_text())
        return {**state, "concepts": cached["concepts"]}

    # Prefer chunking by segments (preserves timestamps); fall back to plain text
    if timestamped_segments:
        chunks = _chunk_transcript_with_timestamps(
            timestamped_segments, max_tokens=CONCEPT_CHUNK_TOKENS
        )
    else:
        words = normalized_transcript.split()
        step = int(CONCEPT_CHUNK_TOKENS * 0.75)
        chunks = [
            {"text": " ".join(words[i: i + step]), "start_fmt": "", "end_fmt": ""}
            for i in range(0, len(words), step)
        ]

    logger.info(f"[extract] Extracting concepts from {len(chunks)} chunks ...")

    all_concepts: List[ConceptNode] = []
    for i, chunk in enumerate(chunks):
        logger.info(f"[extract] Chunk {i + 1}/{len(chunks)} ...")
        prefix = f"{video_id}_c{i}"
        concepts_from_chunk = _extract_from_chunk(chunk, prefix)
        all_concepts.extend(concepts_from_chunk)

    # Deduplicate
    unique_concepts = _deduplicate_concepts(all_concepts)

    # Re-assign clean sequential IDs after dedup
    for idx, concept in enumerate(unique_concepts):
        concept["id"] = f"{video_id}_concept_{idx + 1}"

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_file.write_text(
        json.dumps({"concepts": unique_concepts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"[extract] Extracted {len(unique_concepts)} unique concepts.")
    return {**state, "concepts": unique_concepts}

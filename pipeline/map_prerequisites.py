"""
pipeline/map_prerequisites.py — Prerequisite Mapping Agent
Given the ordered list of concepts (ordered by first_mentioned_at),
asks GPT to reason about which concepts are prerequisites of which,
strictly based on the pedagogical flow of the teacher in the transcript.
Builds and returns a list of PrerequisiteEdge objects.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CACHE_DIR
from pipeline.llm_client import get_completion
from state import PipelineState, ConceptNode, PrerequisiteEdge

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a learning-science expert analyzing the pedagogical flow of an educational lecture.

You are given:
1. An ordered list of concepts (ordered by when they first appeared in the lecture)
2. Relevant transcript excerpts for context

Your task: Identify PREREQUISITE relationships between concepts.
A concept A is a PREREQUISITE of concept B if:
- The teacher explained A BEFORE B
- AND understanding A is NECESSARY to understand B
- AND this dependency is evident from the transcript flow

Rules:
- Only create edges that are pedagogically justified by the transcript
- Do NOT create edges based on general domain knowledge alone
- Every edge must have a rationale grounded in the lecture content
- Assign a confidence score (0.0–1.0) based on how clearly the dependency is stated
- A confidence of 1.0 means the teacher explicitly said "you need to know X to understand Y"
- A confidence of 0.5 means the ordering implies it but is not explicitly stated

Return JSON: {"prerequisite_edges": [{"from_concept": "id1", "to_concept": "id2", "confidence": 0.9, "rationale": "..."}]}
IMPORTANT: Return ONLY valid JSON."""


def _sort_concepts_by_timestamp(concepts: List[ConceptNode]) -> List[ConceptNode]:
    """Sort concepts by first_mentioned_at (HH:MM:SS string sort works correctly)."""
    def ts_key(c):
        ts = c.get("first_mentioned_at", "")
        return ts if ts else "99:99:99"  # push empty timestamps to end

    return sorted(concepts, key=ts_key)


def _build_concept_list_str(concepts: List[ConceptNode]) -> str:
    """Format concepts as a numbered list for the prompt."""
    lines = []
    for i, c in enumerate(concepts):
        ts = c.get("first_mentioned_at", "")
        ts_str = f" [at {ts}]" if ts else ""
        lines.append(
            f"{i + 1}. ID={c['id']}{ts_str}\n"
            f"   Term: {c['normalized_term']}\n"
            f"   Domain: {c['domain']}\n"
            f"   Description: {c['description']}"
        )
    return "\n\n".join(lines)


def map_prerequisites_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Build prerequisite DAG from ordered concept list.

    Input state keys:  video_id, concepts, normalized_transcript
    Output state keys: prerequisite_edges
    """
    video_id = state["video_id"]
    concepts = state.get("concepts", [])
    normalized_transcript = state.get("normalized_transcript", "")

    if not concepts:
        logger.warning(f"[prereq] No concepts found for {video_id}, skipping.")
        return {**state, "prerequisite_edges": []}

    # ── Cache check ─────────────────────────────────────────────────────────
    ids_str = "|".join(c["id"] for c in concepts)
    cache_key = hashlib.md5(ids_str.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"{video_id}_prereqs_{cache_key}.json"

    if cache_file.exists():
        logger.info(f"[prereq] Using cached prerequisite edges for {video_id}")
        cached = json.loads(cache_file.read_text())
        return {**state, "prerequisite_edges": cached["prerequisite_edges"]}

    # Sort concepts chronologically
    ordered_concepts = _sort_concepts_by_timestamp(concepts)
    concept_list_str = _build_concept_list_str(ordered_concepts)

    # Include a snippet of the transcript for context (first 2000 words)
    transcript_snippet = " ".join(normalized_transcript.split()[:2000])

    user_prompt = (
        f"Here are the concepts extracted from the lecture, in the order they were taught:\n\n"
        f"{concept_list_str}\n\n"
        f"---\nTranscript excerpt for context:\n{transcript_snippet}"
    )

    logger.info(
        f"[prereq] Mapping prerequisites for {len(ordered_concepts)} concepts in {video_id}"
    )

    edges: List[PrerequisiteEdge] = []

    try:
        raw_response = get_completion(SYSTEM_PROMPT, user_prompt, mode="smart")
        data = json.loads(raw_response)
        raw_edges = data.get("prerequisite_edges", [])

        # Validate edges reference actual concept IDs
        valid_ids = {c["id"] for c in concepts}
        for edge in raw_edges:
            from_id = edge.get("from_concept", "")
            to_id = edge.get("to_concept", "")
            if from_id in valid_ids and to_id in valid_ids and from_id != to_id:
                edges.append({
                    "from_concept": from_id,
                    "to_concept": to_id,
                    "confidence": float(edge.get("confidence", 0.5)),
                    "rationale": edge.get("rationale", ""),
                })
            else:
                logger.warning(
                    f"[prereq] Skipping edge with invalid IDs: {from_id} → {to_id}"
                )
    except Exception as e:
        logger.error(f"[prereq] Failed to map prerequisites: {e}")

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_file.write_text(
        json.dumps({"prerequisite_edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"[prereq] Found {len(edges)} prerequisite edges for {video_id}")
    return {**state, "prerequisite_edges": edges}

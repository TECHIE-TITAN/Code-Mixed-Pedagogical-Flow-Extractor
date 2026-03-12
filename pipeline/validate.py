"""
pipeline/validate.py — Validation Agent (The Reflection Loop)
This is the agentic heart of the pipeline.

It checks the extracted concept graph for:
1. Cycles (invalid for a prerequisite DAG)
2. Isolated concept nodes (no edges → extraction probably failed)
3. Low-confidence edges below threshold
4. Minimum edge count

If checks fail, it asks GPT to critique the extraction and produces
a feedback string that gets fed back to the extraction agent on retry.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_PREREQUISITE_EDGES,
    CONFIDENCE_THRESHOLD,
    MAX_VALIDATION_RETRIES,
)
from pipeline.llm_client import get_completion
from state import PipelineState

logger = logging.getLogger(__name__)


CRITIQUE_SYSTEM_PROMPT = """You are a critical reviewer of pedagogical concept graphs.
You are given a directed graph of educational concepts and prerequisite edges, along with the source transcript.

Your job:
1. Identify any LOGICAL ERRORS in the prerequisite edges (e.g., A requires B but B was taught after A)
2. Identify MISSING prerequisite edges that are clearly implied by the transcript
3. Identify any SPURIOUS edges that don't have real pedagogical justification
4. Identify concepts that seem INCORRECTLY extracted or should be merged

Return JSON: {
    "issues": [{"type": "missing_edge|spurious_edge|wrong_direction|merge_concepts", "description": "...", "affected_ids": [...]}],
    "overall_quality": "poor|acceptable|good",
    "revision_instructions": "Specific instructions for the extraction agent on how to fix the issues."
}
IMPORTANT: Return ONLY valid JSON."""


def _check_dag_cycles(concepts, edges) -> List[str]:
    """Check for cycles using DFS. Returns list of error messages."""
    import networkx as nx

    G = nx.DiGraph()
    for c in concepts:
        G.add_node(c["id"])
    for e in edges:
        G.add_edge(e["from_concept"], e["to_concept"])

    try:
        cycle = nx.find_cycle(G)
        return [f"Cycle detected: {' → '.join(n for n, _ in cycle)} → {cycle[0][0]}"]
    except nx.NetworkXNoCycle:
        return []


def _check_isolated_nodes(concepts, edges) -> List[str]:
    """Check for concepts with no edges at all."""
    connected = set()
    for e in edges:
        connected.add(e["from_concept"])
        connected.add(e["to_concept"])

    isolated = [c["id"] for c in concepts if c["id"] not in connected]
    if len(isolated) > len(concepts) * 0.5:  # More than 50% isolated is suspicious
        return [
            f"{len(isolated)}/{len(concepts)} concepts are isolated (no prerequisite edges). "
            "Extraction may have been incomplete."
        ]
    return []


def _check_low_confidence(edges) -> List[str]:
    """Flag edges below confidence threshold."""
    low = [e for e in edges if e["confidence"] < CONFIDENCE_THRESHOLD]
    if low:
        return [
            f"{len(low)} edges have confidence below {CONFIDENCE_THRESHOLD}: "
            + ", ".join(f"{e['from_concept']}→{e['to_concept']} ({e['confidence']:.2f})" for e in low)
        ]
    return []


def _check_minimum_edges(concepts, edges) -> List[str]:
    """Ensure there are enough prerequisite edges for the graph to be meaningful."""
    if len(concepts) >= 3 and len(edges) < MIN_PREREQUISITE_EDGES:
        return [
            f"Only {len(edges)} prerequisite edge(s) found for {len(concepts)} concepts. "
            f"Expected at least {MIN_PREREQUISITE_EDGES}."
        ]
    return []


def _llm_critique(state: PipelineState) -> Tuple[str, str]:
    """
    Ask LLM to critique the current graph and provide revision instructions.
    Returns (overall_quality, revision_instructions).
    """
    concepts = state.get("concepts", [])
    edges = state.get("prerequisite_edges", [])
    transcript_snippet = " ".join(
        state.get("normalized_transcript", "").split()[:1500]
    )

    concepts_str = json.dumps(concepts, ensure_ascii=False, indent=2)
    edges_str = json.dumps(edges, ensure_ascii=False, indent=2)

    user_prompt = (
        f"Concepts extracted:\n{concepts_str}\n\n"
        f"Prerequisite edges:\n{edges_str}\n\n"
        f"Transcript excerpt:\n{transcript_snippet}"
    )

    try:
        raw_response = get_completion(CRITIQUE_SYSTEM_PROMPT, user_prompt, mode="smart")
        data = json.loads(raw_response)
        quality = data.get("overall_quality", "acceptable")
        instructions = data.get("revision_instructions", "")
        issues = data.get("issues", [])
        logger.info(f"[validate] LLM critique: quality={quality}, issues={len(issues)}")
        return quality, instructions
    except Exception as e:
        logger.warning(f"[validate] Critique failed: {e}")
        return "acceptable", ""


def validate_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Validate the extracted prerequisite graph.
    Sets validation_errors, validation_passed, and retry_count.

    Input state keys:  concepts, prerequisite_edges, retry_count
    Output state keys: validation_errors, validation_passed, retry_count
    """
    video_id = state["video_id"]
    concepts = state.get("concepts", [])
    edges = state.get("prerequisite_edges", [])
    retry_count = state.get("retry_count", 0)

    logger.info(
        f"[validate] Validating graph for {video_id} "
        f"({len(concepts)} concepts, {len(edges)} edges, retry={retry_count})"
    )

    # ── Structural checks ────────────────────────────────────────────────────
    errors: List[str] = []
    errors.extend(_check_dag_cycles(concepts, edges))
    errors.extend(_check_isolated_nodes(concepts, edges))
    errors.extend(_check_minimum_edges(concepts, edges))
    # Note: low-confidence edges are flagged but don't fail validation
    low_conf_warnings = _check_low_confidence(edges)
    for w in low_conf_warnings:
        logger.warning(f"[validate] ⚠ {w}")

    if errors:
        logger.warning(
            f"[validate] ❌ Structural errors found:\n" + "\n".join(f"  - {e}" for e in errors)
        )

        # If we've hit max retries, pass anyway with warnings
        if retry_count >= MAX_VALIDATION_RETRIES:
            logger.warning(
                f"[validate] Max retries ({MAX_VALIDATION_RETRIES}) reached. "
                "Passing with errors logged."
            )
            return {
                **state,
                "validation_errors": errors,
                "validation_passed": True,  # force pass to avoid infinite loop
                "retry_count": retry_count,
            }

        # Get LLM critique to feed back to extraction agent
        quality, revision_instructions = _llm_critique(state)
        if revision_instructions:
            errors.append(f"[LLM Critique] {revision_instructions}")

        return {
            **state,
            "validation_errors": errors,
            "validation_passed": False,
            "retry_count": retry_count + 1,
        }

    # ── All checks passed ────────────────────────────────────────────────────
    logger.info(f"[validate] ✅ Graph for {video_id} passed validation.")
    return {
        **state,
        "validation_errors": [],
        "validation_passed": True,
        "retry_count": retry_count,
    }


def should_retry(state: PipelineState) -> str:
    """
    LangGraph conditional edge function.
    Returns "retry" → routes back to extract_concepts
    Returns "output" → routes forward to output agent
    """
    if state.get("validation_passed", False):
        return "output"
    return "retry"

"""
state.py — Defines the shared LangGraph pipeline state.
Every agent node reads from and writes to this TypedDict.
"""

from typing import TypedDict, List, Optional, Annotated
import operator


class ConceptNode(TypedDict):
    id: str
    raw_term: str               # as it appeared in the transcript (possibly code-mixed)
    normalized_term: str        # standard English academic term
    domain: str                 # e.g., "Computer Science", "Physics"
    first_mentioned_at: str     # timestamp "HH:MM:SS"
    description: str            # one-line description from context


class PrerequisiteEdge(TypedDict):
    from_concept: str           # concept id
    to_concept: str             # concept id
    confidence: float           # 0.0 – 1.0
    rationale: str              # why this prerequisite relationship exists


class PipelineState(TypedDict):
    # ── Input ──────────────────────────────
    video_id: str
    video_url: str
    language_mix: List[str]     # e.g., ["Hindi", "English"]
    title: str

    # ── Ingestion ──────────────────────────
    audio_path: str

    # ── Transcription ──────────────────────
    raw_transcript: str
    timestamped_segments: List[dict]    # [{start, end, text}, ...]

    # ── Normalization ──────────────────────
    normalized_transcript: str
    glossary_additions: List[dict]      # [{raw, normalized, domain}, ...]

    # ── Concept Extraction ─────────────────
    concepts: List[ConceptNode]

    # ── Prerequisite Mapping ───────────────
    prerequisite_edges: List[PrerequisiteEdge]

    # ── Validation / Reflection ────────────
    validation_errors: List[str]
    validation_passed: bool
    retry_count: int

    # ── Output ─────────────────────────────
    output_json_path: str
    output_graphml_path: str
    output_html_path: str

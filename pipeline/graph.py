"""
pipeline/graph.py — LangGraph Orchestrator
Wires all agent nodes into a StateGraph with the agentic reflection loop.

Flow:
  ingest → transcribe → normalize → extract_concepts → map_prerequisites
  → validate ──[pass]──→ output → END
              └──[fail]──→ extract_concepts (retry with feedback)
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import PipelineState
from pipeline.ingest import ingest_agent
from pipeline.transcribe import transcribe_agent
from pipeline.normalize import normalize_agent
from pipeline.extract_concepts import extract_concepts_agent
from pipeline.map_prerequisites import map_prerequisites_agent
from pipeline.validate import validate_agent, should_retry
from pipeline.output import output_agent

logger = logging.getLogger(__name__)


def build_pipeline():
    """
    Construct and compile the LangGraph StateGraph for the full pipeline.
    Returns a compiled app ready to invoke.
    """
    from langgraph.graph import StateGraph, END  # lazy import

    workflow = StateGraph(PipelineState)

    # ── Register nodes ───────────────────────────────────────────────────────
    workflow.add_node("ingest", ingest_agent)
    workflow.add_node("transcribe", transcribe_agent)
    workflow.add_node("normalize", normalize_agent)
    workflow.add_node("extract_concepts", extract_concepts_agent)
    workflow.add_node("map_prerequisites", map_prerequisites_agent)
    workflow.add_node("validate", validate_agent)
    workflow.add_node("output", output_agent)

    # ── Define edges (sequential flow) ───────────────────────────────────────
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "transcribe")
    workflow.add_edge("transcribe", "normalize")
    workflow.add_edge("normalize", "extract_concepts")
    workflow.add_edge("extract_concepts", "map_prerequisites")
    workflow.add_edge("map_prerequisites", "validate")

    # ── Agentic reflection loop ───────────────────────────────────────────────
    # validate → "output" if passed, → "extract_concepts" if failed (retry)
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "output": "output",
            "retry": "extract_concepts",   # feed validation errors back into extraction
        },
    )

    workflow.add_edge("output", END)

    app = workflow.compile()
    logger.info("[graph] LangGraph pipeline compiled successfully.")
    return app


def run_pipeline_for_video(video_config: dict) -> PipelineState:
    """
    Run the full pipeline for a single video config dict.
    video_config should have: video_id, video_url, language_mix, title

    Returns the final state dict.
    """
    app = build_pipeline()

    initial_state: PipelineState = {
        "video_id": video_config["video_id"],
        "video_url": video_config["url"],
        "language_mix": video_config.get("language_mix", ["Hindi", "English"]),
        "title": video_config.get("title", video_config["video_id"]),
        # Initialize all other keys to empty defaults
        "audio_path": "",
        "raw_transcript": "",
        "timestamped_segments": [],
        "normalized_transcript": "",
        "glossary_additions": [],
        "concepts": [],
        "prerequisite_edges": [],
        "validation_errors": [],
        "validation_passed": False,
        "retry_count": 0,
        "output_json_path": "",
        "output_graphml_path": "",
        "output_html_path": "",
    }

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Starting pipeline for: {video_config.get('title', video_config['video_id'])}")
    logger.info(f"# URL: {video_config['url']}")
    logger.info(f"{'#'*60}")

    final_state = app.invoke(initial_state)
    return final_state

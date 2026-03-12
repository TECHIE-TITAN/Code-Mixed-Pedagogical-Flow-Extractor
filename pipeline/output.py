"""
pipeline/output.py — Output Agent
Serializes the final pipeline state to:
  1. JSON  — primary machine-readable output
  2. GraphML — standard graph interchange format
  3. (HTML visualization is handled by visualization/visualize.py)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR
from state import PipelineState

logger = logging.getLogger(__name__)


def output_agent(state: PipelineState) -> PipelineState:
    """
    LangGraph node: Serialize final results to JSON + GraphML.

    Input state keys:  all populated state fields
    Output state keys: output_json_path, output_graphml_path
    """
    import networkx as nx  # lazy import

    video_id = state["video_id"]
    concepts = state.get("concepts", [])
    edges = state.get("prerequisite_edges", [])

    logger.info(f"[output] Serializing results for {video_id}")

    # ── 1. Build JSON output ─────────────────────────────────────────────────
    output_data = {
        "video_id": video_id,
        "source_url": state.get("video_url", ""),
        "title": state.get("title", ""),
        "language_mix": state.get("language_mix", []),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_metadata": {
            "concepts_extracted": len(concepts),
            "prerequisite_edges": len(edges),
            "validation_errors": state.get("validation_errors", []),
            "retry_count": state.get("retry_count", 0),
            "glossary_additions": len(state.get("glossary_additions", [])),
        },
        "concepts": concepts,
        "prerequisite_edges": edges,
    }

    json_path = OUTPUTS_DIR / f"{video_id}_output.json"
    json_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[output] JSON saved: {json_path}")

    # ── 2. Build GraphML output ──────────────────────────────────────────────
    G = nx.DiGraph()

    # Add concept nodes with attributes
    for c in concepts:
        G.add_node(
            c["id"],
            label=c["normalized_term"],
            raw_term=c.get("raw_term", ""),
            domain=c.get("domain", ""),
            first_mentioned_at=c.get("first_mentioned_at", ""),
            description=c.get("description", ""),
        )

    # Add prerequisite edges with attributes
    for e in edges:
        G.add_edge(
            e["from_concept"],
            e["to_concept"],
            confidence=e.get("confidence", 0.0),
            rationale=e.get("rationale", ""),
        )

    graphml_path = OUTPUTS_DIR / f"{video_id}_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    logger.info(f"[output] GraphML saved: {graphml_path}")

    # ── 3. Print summary to console ─────────────────────────────────────────
    logger.info(
        f"\n{'='*60}\n"
        f"✅ Pipeline complete for: {state.get('title', video_id)}\n"
        f"   URL     : {state.get('video_url', '')}\n"
        f"   Lang    : {', '.join(state.get('language_mix', []))}\n"
        f"   Concepts: {len(concepts)}\n"
        f"   Edges   : {len(edges)}\n"
        f"   Retries : {state.get('retry_count', 0)}\n"
        f"   JSON    : {json_path}\n"
        f"   GraphML : {graphml_path}\n"
        f"{'='*60}"
    )

    return {
        **state,
        "output_json_path": str(json_path),
        "output_graphml_path": str(graphml_path),
    }

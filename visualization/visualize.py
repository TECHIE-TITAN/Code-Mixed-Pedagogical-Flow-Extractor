"""
visualization/visualize.py — Graph Visualizer
Generates an interactive HTML prerequisite graph for a single video
using pyvis. Each node is a concept, each directed edge is a prerequisite.
Color-coded by domain, sized by number of connections.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Domain → color mapping
DOMAIN_COLORS: Dict[str, str] = {
    "Computer Science": "#4A90D9",
    "Mathematics": "#E67E22",
    "Physics": "#27AE60",
    "Chemistry": "#8E44AD",
    "Engineering": "#C0392B",
    "General": "#95A5A6",
}


def generate_html_graph(
    video_id: str,
    title: str,
    concepts: List[dict],
    edges: List[dict],
    output_path: str = None,
) -> str:
    """
    Generate an interactive pyvis HTML graph for one video's concept DAG.
    Returns the path to the generated HTML file.
    """
    from pyvis.network import Network  # lazy import

    if output_path is None:
        output_path = str(OUTPUTS_DIR / f"{video_id}_graph.html")

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
    )

    # Physics layout options for better DAG readability
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 120
        }
      },
      "physics": {
        "enabled": false
      },
      "edges": {
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 1.2 }
        },
        "color": { "color": "#7f8c8d", "highlight": "#f39c12" },
        "smooth": { "type": "cubicBezier" }
      },
      "nodes": {
        "font": { "size": 14, "face": "Arial" },
        "borderWidth": 2
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    # Count connections per node (for sizing)
    degree: Dict[str, int] = {c["id"]: 0 for c in concepts}
    for e in edges:
        degree[e["from_concept"]] = degree.get(e["from_concept"], 0) + 1
        degree[e["to_concept"]] = degree.get(e["to_concept"], 0) + 1

    # Add nodes
    for concept in concepts:
        cid = concept["id"]
        domain = concept.get("domain", "General")
        color = DOMAIN_COLORS.get(domain, DOMAIN_COLORS["General"])
        size = 20 + degree.get(cid, 0) * 5  # bigger nodes = more connections

        tooltip = (
            f"<b>{concept['normalized_term']}</b><br>"
            f"Domain: {domain}<br>"
            f"First at: {concept.get('first_mentioned_at', 'N/A')}<br>"
            f"<i>{concept.get('description', '')}</i>"
        )

        net.add_node(
            cid,
            label=concept["normalized_term"],
            title=tooltip,
            color=color,
            size=size,
            shape="dot",
        )

    # Add edges
    for edge in edges:
        confidence = edge.get("confidence", 0.5)
        width = 1 + confidence * 4  # thicker = higher confidence
        tooltip = (
            f"Prerequisite: {edge['from_concept']} → {edge['to_concept']}<br>"
            f"Confidence: {confidence:.0%}<br>"
            f"{edge.get('rationale', '')}"
        )
        net.add_edge(
            edge["from_concept"],
            edge["to_concept"],
            title=tooltip,
            width=width,
            color={"color": "#f39c12" if confidence >= 0.8 else "#7f8c8d"},
        )

    net.save_graph(output_path)
    logger.info(f"[visualize] HTML graph saved: {output_path}")
    return output_path


def generate_all_graphs() -> List[str]:
    """
    Find all *_output.json files in OUTPUTS_DIR and generate HTML graphs.
    Returns list of generated HTML paths.
    """
    json_files = list(OUTPUTS_DIR.glob("*_output.json"))
    if not json_files:
        logger.warning("[visualize] No output JSON files found.")
        return []

    html_paths = []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        html_path = generate_html_graph(
            video_id=data["video_id"],
            title=data.get("title", data["video_id"]),
            concepts=data.get("concepts", []),
            edges=data.get("prerequisite_edges", []),
        )
        html_paths.append(html_path)

    return html_paths

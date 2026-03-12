"""
visualization/dashboard.py — Streamlit Dashboard
Interactive web dashboard showing all 5 videos' concept graphs side by side.
Run with: streamlit run visualization/dashboard.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR
from visualization.visualize import generate_html_graph

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pedagogical Flow Extractor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Code-Mixed Pedagogical Flow Extractor")
st.markdown(
    "Interactive visualization of concept prerequisite graphs extracted "
    "from code-mixed educational videos."
)

# ── Load all output JSONs ────────────────────────────────────────────────────
@st.cache_data
def load_all_outputs():
    results = {}
    for jf in sorted(OUTPUTS_DIR.glob("*_output.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        results[data["video_id"]] = data
    return results


all_outputs = load_all_outputs()

if not all_outputs:
    st.warning(
        "⚠️ No pipeline outputs found. Run `python main.py` first to process videos."
    )
    st.stop()

# ── Sidebar: Video selector ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📹 Select Video")
    video_ids = list(all_outputs.keys())
    selected_id = st.selectbox(
        "Choose a video",
        video_ids,
        format_func=lambda vid: all_outputs[vid].get("title", vid),
    )
    st.divider()

    # Global stats
    total_concepts = sum(len(v["concepts"]) for v in all_outputs.values())
    total_edges = sum(len(v["prerequisite_edges"]) for v in all_outputs.values())
    st.metric("Total Videos Processed", len(all_outputs))
    st.metric("Total Concepts Extracted", total_concepts)
    st.metric("Total Prerequisite Edges", total_edges)

# ── Main panel ───────────────────────────────────────────────────────────────
data = all_outputs[selected_id]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Concepts", len(data["concepts"]))
col2.metric("Prerequisite Edges", len(data["prerequisite_edges"]))
col3.metric("Language Mix", " + ".join(data.get("language_mix", [])))
col4.metric("Pipeline Retries", data["pipeline_metadata"].get("retry_count", 0))

st.markdown(f"**Source:** [{data.get('title', selected_id)}]({data['source_url']})")
st.divider()

# ── Tabs: Graph | Concepts | Edges | Raw JSON ────────────────────────────────
tab_graph, tab_concepts, tab_edges, tab_json = st.tabs(
    ["🗺️ Graph", "📚 Concepts", "🔗 Edges", "📄 Raw JSON"]
)

with tab_graph:
    st.subheader("Prerequisite Dependency Graph")
    st.caption(
        "→ Arrows point FROM prerequisites TO dependent concepts. "
        "Thicker/orange edges = higher confidence. Node size ∝ connectivity."
    )

    # Generate / load HTML graph
    html_path = OUTPUTS_DIR / f"{selected_id}_graph.html"
    if not html_path.exists():
        with st.spinner("Generating graph..."):
            generate_html_graph(
                video_id=data["video_id"],
                title=data.get("title", selected_id),
                concepts=data["concepts"],
                edges=data["prerequisite_edges"],
                output_path=str(html_path),
            )

    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=800, scrolling=False)
    else:
        st.error("Could not generate graph. Make sure pyvis is installed.")


with tab_concepts:
    st.subheader(f"Extracted Concepts ({len(data['concepts'])})")
    for concept in data["concepts"]:
        with st.expander(
            f"**{concept['normalized_term']}** — {concept.get('domain', 'N/A')} "
            f"[{concept.get('first_mentioned_at', '?')}]"
        ):
            col_a, col_b = st.columns(2)
            col_a.markdown(f"**Raw term:** `{concept.get('raw_term', 'N/A')}`")
            col_b.markdown(f"**ID:** `{concept['id']}`")
            st.markdown(f"*{concept.get('description', 'No description')}*")


with tab_edges:
    st.subheader(f"Prerequisite Edges ({len(data['prerequisite_edges'])})")

    # Build concept ID → name map
    id_to_name = {c["id"]: c["normalized_term"] for c in data["concepts"]}

    for edge in sorted(
        data["prerequisite_edges"], key=lambda e: e.get("confidence", 0), reverse=True
    ):
        from_name = id_to_name.get(edge["from_concept"], edge["from_concept"])
        to_name = id_to_name.get(edge["to_concept"], edge["to_concept"])
        confidence = edge.get("confidence", 0.0)
        color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
        with st.expander(
            f"{color} **{from_name}** → **{to_name}** ({confidence:.0%})"
        ):
            st.markdown(f"*{edge.get('rationale', 'No rationale provided')}*")


with tab_json:
    st.subheader("Raw JSON Output")
    st.json(data)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "iREL 2026 | Code-Mixed Pedagogical Flow Extractor | "
    "LangGraph + Whisper + GPT-4o + NetworkX + Pyvis"
)

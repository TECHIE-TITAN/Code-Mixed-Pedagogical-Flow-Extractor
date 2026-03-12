"""
main.py — Pipeline Entrypoint
Loads video configurations from videos.yaml and runs the LangGraph
agentic pipeline for each video sequentially.

Usage:
    python main.py                        # process all videos in videos.yaml
    python main.py --video-id v001        # process only one specific video
    python main.py --skip-ingest          # skip download (audio already exists)
    python main.py --visualize-only       # only regenerate HTML graphs from existing outputs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import VIDEOS_YAML_PATH, OUTPUTS_DIR
from pipeline.graph import run_pipeline_for_video
from visualization.visualize import generate_all_graphs


def load_videos(yaml_path: Path) -> list:
    """Load video configs from videos.yaml."""
    if not yaml_path.exists():
        logger.error(f"videos.yaml not found at {yaml_path}")
        sys.exit(1)
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("videos", [])


def run_all(video_id_filter: str = None):
    """Run the full pipeline for all (or filtered) videos."""
    videos = load_videos(VIDEOS_YAML_PATH)

    if video_id_filter:
        videos = [v for v in videos if v["video_id"] == video_id_filter]
        if not videos:
            logger.error(f"No video found with video_id='{video_id_filter}'")
            sys.exit(1)

    logger.info(f"Processing {len(videos)} video(s)...")

    results = []
    failed = []

    for i, video in enumerate(videos):
        logger.info(f"\n[{i+1}/{len(videos)}] Starting: {video.get('title', video['video_id'])}")
        try:
            final_state = run_pipeline_for_video(video)
            results.append({
                "video_id": video["video_id"],
                "status": "success",
                "concepts": len(final_state.get("concepts", [])),
                "edges": len(final_state.get("prerequisite_edges", [])),
                "json_output": final_state.get("output_json_path", ""),
            })
        except Exception as e:
            logger.exception(f"Pipeline failed for {video['video_id']}: {e}")
            failed.append({"video_id": video["video_id"], "error": str(e)})

    # ── Generate HTML visualizations ─────────────────────────────────────────
    logger.info("\nGenerating HTML visualizations...")
    html_paths = generate_all_graphs()
    for hp in html_paths:
        logger.info(f"  HTML graph: {hp}")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"  ✅ {r['video_id']}: {r['concepts']} concepts, "
            f"{r['edges']} edges → {r['json_output']}"
        )
    for f in failed:
        logger.info(f"  ❌ {f['video_id']}: {f['error']}")
    logger.info("=" * 60)

    if failed:
        logger.warning(f"{len(failed)} video(s) failed. Check pipeline.log for details.")

    return results, failed


def main():
    parser = argparse.ArgumentParser(
        description="Code-Mixed Pedagogical Flow Extractor — Agentic Pipeline"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process only a specific video by its video_id (e.g., v001)",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip pipeline, only regenerate HTML graphs from existing JSON outputs",
    )
    args = parser.parse_args()

    if args.visualize_only:
        logger.info("Visualize-only mode: regenerating HTML graphs...")
        html_paths = generate_all_graphs()
        for hp in html_paths:
            logger.info(f"  Generated: {hp}")
        logger.info("Done. Launch dashboard with: streamlit run visualization/dashboard.py")
        return

    run_all(video_id_filter=args.video_id)
    logger.info(
        "\n🚀 All done! Launch the dashboard with:\n"
        "   streamlit run visualization/dashboard.py"
    )


if __name__ == "__main__":
    main()

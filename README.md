# 🎓 Code-Mixed Pedagogical Flow Extractor

An agentic NLP pipeline that ingests code-mixed educational videos (Hinglish, Telugu-English, Tamil-English), transcribes them with Whisper, normalizes colloquial Indic terms via GPT-4o, extracts technical concepts, and builds a prerequisite dependency graph — all orchestrated by **LangGraph** with a self-reflection validation loop.

---

## 🏗️ Architecture

```
videos.yaml (5 URLs)
      │
      ▼
┌─────────────────────────────────────────────────┐
│              LangGraph StateGraph               │
│                                                 │
│  ingest → transcribe → normalize →              │
│  extract_concepts → map_prerequisites →         │
│  validate ──[pass]──→ output → END              │
│           └──[fail]──→ extract_concepts (retry) │
└─────────────────────────────────────────────────┘
      │
      ▼
data/outputs/  →  JSON + GraphML + HTML graphs
      │
      ▼
streamlit dashboard (interactive visual)
```

### Agent Roles
| Agent | Responsibility |
|-------|---------------|
| **Ingestion** | `yt-dlp` download + `ffmpeg` convert to 16kHz WAV |
| **Transcription** | `faster-whisper` ASR with word timestamps |
| **Normalization** | GPT-4o-mini maps Hinglish/Tanglish → standard English |
| **Concept Extraction** | GPT-4o extracts structured `ConceptNode` objects |
| **Prerequisite Mapping** | GPT-4o reasons about pedagogical flow → DAG edges |
| **Validation** | Cycle detection + LLM critique + reflection retry loop |
| **Output** | Serializes JSON + GraphML |

---

## ⚙️ Setup

### 1. Clone & create virtual environment
```bash
git clone <your-repo-url>
cd code-mixed-pedagogy-extractor
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install system dependencies
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Install yt-dlp
pip install yt-dlp
```

### 3. Install Python packages
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# OPENAI_API_KEY=sk-...
```

### 5. Configure your 5 videos
Edit `videos.yaml` — replace the placeholder URLs with real YouTube video URLs of code-mixed educational content.

---

## 🚀 Running the Pipeline

```bash
# Process all 5 videos
python main.py

# Process only one video
python main.py --video-id v001

# Regenerate HTML graphs from existing outputs (no API calls)
python main.py --visualize-only

# Launch the interactive dashboard
streamlit run visualization/dashboard.py
```

---

## 📁 Project Structure

```
code-mixed-pedagogy-extractor/
│
├── pipeline/
│   ├── ingest.py           # Audio download & conversion
│   ├── transcribe.py       # Whisper ASR
│   ├── normalize.py        # Code-mix normalization
│   ├── extract_concepts.py # LLM concept extraction
│   ├── map_prerequisites.py# Prerequisite DAG construction
│   ├── validate.py         # DAG validation + reflection loop
│   ├── output.py           # JSON + GraphML serialization
│   └── graph.py            # LangGraph orchestrator
│
├── visualization/
│   ├── visualize.py        # Pyvis HTML graph generator
│   └── dashboard.py        # Streamlit web dashboard
│
├── data/
│   ├── raw_audio/          # Downloaded WAV files
│   ├── transcripts/        # Raw + normalized transcripts
│   └── outputs/            # JSON + GraphML + HTML outputs
│
├── config.py               # All configuration & paths
├── state.py                # LangGraph PipelineState definition
├── main.py                 # CLI entrypoint
├── videos.yaml             # Video sources configuration
├── glossary.json           # Cumulative code-mix term mappings (auto-generated)
├── requirements.txt
└── .env.example
```

---

## 📤 Output Format

### `data/outputs/{video_id}_output.json`
```json
{
  "video_id": "v001",
  "source_url": "https://youtube.com/...",
  "title": "Recursion in Hinglish",
  "language_mix": ["Hindi", "English"],
  "concepts": [
    {
      "id": "v001_concept_1",
      "raw_term": "function wala concept",
      "normalized_term": "Function",
      "domain": "Computer Science",
      "first_mentioned_at": "00:01:23",
      "description": "A reusable block of code that performs a specific task"
    }
  ],
  "prerequisite_edges": [
    {
      "from_concept": "v001_concept_1",
      "to_concept": "v001_concept_3",
      "confidence": 0.92,
      "rationale": "Teacher explained Functions before introducing Recursion, explicitly stating 'recursion ek function hi hai'"
    }
  ]
}
```

---

## 🎬 Video Sources

| ID | Title | Language Mix | URL |
|----|-------|-------------|-----|
| v001 | ... | Hindi-English | ... |
| v002 | ... | Hindi-English | ... |
| v003 | ... | Telugu-English | ... |
| v004 | ... | Tamil-English | ... |
| v005 | ... | Hindi-English | ... |

---

## 💡 Design Decisions

**Why Whisper `medium` on CPU?**
Your system (i7-1360P, no GPU) can run `medium` in ~15 min per 10-min video with `int8` quantization. It handles code-mixed speech better than `base` while still fitting in 16 GB RAM.

**Why LangGraph for orchestration?**
LangGraph's `StateGraph` natively supports cycles (the validation→retry loop). State is immutable and fully traceable, making debugging straightforward.

**Why GPT-4o for concept extraction but GPT-4o-mini for normalization?**
Normalization is a simpler substitution task — mini handles it at 10x lower cost. Prerequisite reasoning requires deeper chain-of-thought, justifying the stronger model.

**Why JSON + GraphML?**
JSON for LLM I/O and human readability. GraphML for compatibility with graph tools (Gephi, Neo4j, Cytoscape). Both are machine-readable and standard.

---

## ⚠️ Hardware Notes

| Component | Recommended | This Repo Default |
|-----------|------------|-------------------|
| GPU | NVIDIA (any) | CPU (works, slower) |
| Whisper model | large-v3 with GPU | `medium` on CPU |
| RAM | 16 GB+ | ✅ Works on 16 GB |
| Processing time | ~2 min/video (GPU) | ~20 min/video (CPU) |

Change `WHISPER_MODEL_SIZE` in `config.py` to `"base"` for faster but less accurate transcription.

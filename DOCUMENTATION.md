# Project Development Documentation
## Code-Mixed Pedagogical Flow Extractor

> A chronological account of how this project was conceived, designed, broken, fixed, and extended — from a PDF task description to a working agentic NLP pipeline.

---

## Table of Contents

1. [The Original Task](#1-the-original-task)
2. [Phase 1 — Strategy & Architecture Design](#2-phase-1--strategy--architecture-design)
3. [Phase 2 — Hardware Constraints & Technology Choices](#3-phase-2--hardware-constraints--technology-choices)
4. [Phase 3 — Full Codebase Implementation](#4-phase-3--full-codebase-implementation)
5. [Phase 4 — First Real Run & The OpenAI Quota Wall](#5-phase-4--first-real-run--the-openai-quota-wall)
6. [Phase 5 — LLM Migration: OpenAI → Google Gemini](#6-phase-5--llm-migration-openai--google-gemini)
7. [Phase 6 — Two-Pass Parallel Transcription](#7-phase-6--two-pass-parallel-transcription)
8. [Final Architecture](#8-final-architecture)
9. [File Inventory](#9-file-inventory)
10. [Key Configuration Reference](#10-key-configuration-reference)
11. [Verified Output: v002](#11-verified-output-v002)
12. [Lessons Learned](#12-lessons-learned)

---

## 1. The Original Task

The starting point was `iREL_Recruitment_Task_2026.pdf` — a recruitment challenge from the **Information Retrieval and Extraction Lab (iREL), IIT Hyderabad**.

**The task in one sentence:**
> Given a set of 5 code-mixed (e.g., Hindi-English, Telugu-English, Tamil-English) educational YouTube videos, extract the pedagogical flow of concepts taught and represent it as a prerequisite dependency graph.

**What "code-mixed" means in this context:**
Indian educators on YouTube often teach in a hybrid language — English technical terms embedded in Hindi/Telugu/Tamil sentence structures (called *Hinglish*, *Tenglish*, etc.). For example:
> *"Ab hum linked list ka concept dekhenge — basically ek node ke andar data aur next pointer hota hai"*
> *(Now we will see the concept of linked list — basically a node contains data and a next pointer)*

**Required outputs:**
- A structured graph where nodes = technical concepts and directed edges = prerequisite relationships
- For each concept: its name, domain, timestamp of first mention, description
- For each edge: which concept must be understood before which, with a confidence score and rationale
- Deliverable formats: JSON + GraphML

---

## 2. Phase 1 — Strategy & Architecture Design

### Initial Idea: Simple Sequential Script

The naive approach was a single Python script that would:
1. Download audio with `yt-dlp`
2. Transcribe with Whisper
3. Feed the transcript to a single GPT-4 prompt
4. Parse the JSON response

**Why this was rejected:**
- A single monolithic prompt for a 25-minute Hindi-English lecture would exceed context windows
- Code-mixed normalization and concept extraction are distinct cognitive tasks — conflating them degrades output quality
- No mechanism for self-correction if the extraction produced a bad graph
- Not extensible to 5 videos or different language pairs

### Evolved Idea: Agentic LangGraph Pipeline

After evaluating the problem more carefully, the design evolved to a **multi-agent pipeline** where each agent is responsible for exactly one cognitive task, communicates through a shared typed state, and the system can reflect on its own outputs.

The core insight was that **prerequisite graph extraction is a reasoning task**, not just an extraction task. It requires:
1. Understanding what the teacher said (ASR + normalization)
2. Identifying what concepts were introduced (extraction)
3. Reasoning about which concepts depend on which (dependency inference)
4. Validating that the resulting graph is logically consistent (reflection)

These four concerns became four separate agents, with LangGraph as the orchestrator.

**Two strategy documents were written at this stage:**
- `IMPLEMENTATION_STRATEGY.md` — technology choices and data flow
- `AGENTIC_APPROACH.md` — the LangGraph reflection loop design

---

## 3. Phase 2 — Hardware Constraints & Technology Choices

### Machine Specs (the deployment target)
| Component | Spec |
|---|---|
| CPU | Intel i7-1360P (12 cores, 2.2–5.0 GHz) |
| RAM | 16 GB |
| GPU | **None** (Intel Iris Xe integrated graphics only) |
| OS | Ubuntu 24.04 |
| Free Disk | ~34 GB |
| Python | 3.10.16 |

This had major implications:

**Whisper model selection:**
- `large-v3` was ruled out immediately (needs 10+ GB VRAM)
- `medium` was borderline (3–4 GB RAM peak, very slow on CPU)
- **`small` was chosen**: decent code-mix accuracy, ~480 MB, fits comfortably in RAM
- Compute type: `int8` (fastest CPU quantization mode in `faster-whisper`)

**LLM selection:**
- Local LLMs (Ollama/LM Studio) were considered but rejected — a 7B model on CPU would take 3–5 minutes per inference call, making the pipeline impractical
- **OpenAI GPT-4o / GPT-4o-mini** were chosen as the LLM backend
- `gpt-4o-mini` for cheap/fast tasks (normalization)
- `gpt-4o` for reasoning tasks (extraction, prerequisite mapping, validation)

**Framework stack:**
```
yt-dlp + ffmpeg          — audio ingestion
faster-whisper (small)   — ASR, CPU int8
LangGraph 0.2.27+        — agent orchestration + reflection loop
OpenAI SDK               — LLM calls
sentence-transformers    — concept deduplication (all-MiniLM-L6-v2)
networkx                 — DAG cycle detection + GraphML export
pyvis + Streamlit        — visualization layer
```

---

## 4. Phase 3 — Full Codebase Implementation

The entire project structure was built in a single implementation session:

```
code-mixed-pedagogy-extractor/
├── config.py                    ← all tunable parameters
├── state.py                     ← LangGraph TypedDict state definition
├── main.py                      ← CLI entrypoint
├── videos.yaml                  ← 5 video configs (IDs, URLs, language pairs)
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── ingest.py                ← yt-dlp + ffmpeg audio download
│   ├── transcribe.py            ← faster-whisper ASR
│   ├── normalize.py             ← LLM code-mix → standard English
│   ├── extract_concepts.py      ← LLM concept extraction + embedding dedup
│   ├── map_prerequisites.py     ← LLM prerequisite DAG construction
│   ├── validate.py              ← DAG validation + LLM critique reflection loop
│   ├── output.py                ← JSON + GraphML serialization
│   └── graph.py                 ← LangGraph StateGraph orchestrator
├── visualization/
│   ├── visualize.py             ← Pyvis interactive HTML graph
│   └── dashboard.py             ← Streamlit multi-video dashboard
└── data/
    ├── raw_audio/
    ├── transcripts/
    └── outputs/
```

### The LangGraph Pipeline

The pipeline was wired as a `StateGraph` with 7 nodes and one conditional edge:

```
ingest → transcribe → normalize → extract_concepts → map_prerequisites
                                                            ↓
                                                        validate ──[pass]──→ output → END
                                                            ↑__[fail + LLM critique]__|
```

**The reflection loop** is the agentic core: if `validate` finds structural problems in the extracted graph (cycles, too few edges, disconnected nodes), it:
1. Calls the LLM to *critique* the current extraction
2. Embeds the critique as revision instructions in the state's `validation_errors` field
3. Routes back to `extract_concepts`, which now sees the feedback and tries again
4. This repeats up to `MAX_VALIDATION_RETRIES = 3` times before force-passing

### Shared State Schema (`state.py`)

Every agent reads from and writes to a single `PipelineState` TypedDict:

```python
class PipelineState(TypedDict):
    video_id, video_url, language_mix, title   # input
    audio_path                                  # after ingest
    raw_transcript, timestamped_segments        # after transcribe
    normalized_transcript, glossary_additions   # after normalize
    concepts                                    # after extract_concepts
    prerequisite_edges                          # after map_prerequisites
    validation_errors, validation_passed,       # after validate
    retry_count
    output_json_path, output_graphml_path,      # after output
    output_html_path
```

### Caching Strategy

Every expensive step caches its output to `.cache/`:
- Transcript: `.cache/{video_id}_transcript.json`
- Normalized text: `.cache/{video_id}_normalized_{md5}.json`
- Concepts: `.cache/{video_id}_concepts_{md5}.json`
- Prerequisites: `.cache/{video_id}_prereqs_{md5}.json`

The md5 suffix is derived from the input content — if the transcript changes (e.g., different audio), the cache is automatically invalidated.

### Concept Deduplication

After the LLM extracts concepts from multiple transcript chunks, near-duplicates are filtered using `sentence-transformers`:
- Embed all concept `normalized_term` strings using `all-MiniLM-L6-v2`
- Compute cosine similarity between all pairs
- Drop any concept with >88% similarity to an already-kept concept

This prevents cases like "Linked List", "linked lists", and "linked-list data structure" all appearing as separate nodes.

---

## 5. Phase 4 — First Real Run & The OpenAI Quota Wall

### What Was Tested

Video: `v002` — **"Data Structures - Linked Lists | CodeWithHarry"**
- URL: `https://www.youtube.com/watch?v=TWMCMvfEAv4`
- Duration: ~25 minutes
- Language: Hindi-English (Hinglish)

### What Worked

| Step | Result |
|---|---|
| Audio download (yt-dlp) | ✅ Success |
| ffmpeg 16kHz mono conversion | ✅ Success |
| Whisper `small` transcription | ✅ Success — 111 segments, 1181 words, detected Hindi (98% confidence) |
| Transcript cached to `.cache/v002_transcript.json` | ✅ Persisted |

**Time cost of transcription:** ~40 minutes on CPU for a 25-minute video. This ratio (~1.6× real-time) became a key motivation for later optimization work.

### What Failed

```
openai.RateLimitError: Error code: 429
  You exceeded your current quota, please check your plan and billing details.
```

**Every single LLM call across all 5 normalization chunks failed with HTTP 429.** The OpenAI account had zero remaining quota.

**Result:**  `data/outputs/v002_output.json` was written with:
```json
{ "concepts": [], "prerequisite_edges": [] }
```

The pipeline completed without crashing (the normalize agent caught errors and fell back to the original text), but produced zero useful output.

---

## 6. Phase 5 — LLM Migration: OpenAI → Google Gemini

### Decision

Rather than top up the OpenAI quota, the decision was made to migrate to **Google Gemini** (free tier), which offered sufficient quota for development and testing.

**Package installed:** `google-generativeai` (then later upgraded to `google-genai`)

### Implementation Approach

Instead of a find-and-replace across every agent file, a **thin adapter layer** was introduced: `pipeline/llm_client.py`. All agents import a single function:

```python
from pipeline.llm_client import get_completion

raw = get_completion(system_prompt, user_prompt, mode="fast")   # or "smart"
```

The adapter internally routes to the correct backend based on `LLM_PROVIDER` in `config.py`:

```python
LLM_PROVIDER = "gemini" if GEMINI_API_KEY and not OPENAI_API_KEY else "openai"
```

### Migration Complications (three separate issues)

**Issue 1 — Wrong SDK:**
The initial implementation used `google.generativeai` (the old `google-generativeai` package). This worked but threw deprecation warnings:
```
FutureWarning: All support for the google.generativeai package has ended.
```
**Fix:** Migrated to `google.genai` (the new `google-genai` SDK).

**Issue 2 — Deprecated model name:**
The config had `GEMINI_FAST_MODEL = "gemini-1.5-flash"`. With the new SDK this returned:
```
404: models/gemini-1.5-flash is not found for API version v1beta
```
**Investigation:** Called `client.models.list()` to enumerate available models. Found that the API key's free tier had zero quota on both `gemini-1.5-flash` and `gemini-2.0-flash`.

**Issue 3 — Model quota:**
Testing each available model in sequence:
```
FAIL: gemini-2.0-flash-lite → 429 RESOURCE_EXHAUSTED (limit: 0)
SUCCESS: gemini-2.5-flash-lite → {"ok": true}
```
**Fix:** Updated config to `gemini-2.5-flash-lite` for both fast and smart tiers.

### Final `config.py` LLM Section

```python
GEMINI_FAST_MODEL  = "gemini-2.5-flash-lite"   # confirmed working on free tier
GEMINI_SMART_MODEL = "gemini-2.5-flash-lite"
```

### Clearing the Bad Cache

The failed OpenAI run had written empty results to:
- `.cache/v002_concepts_*.json`  → `{"concepts": []}`
- `.cache/v002_normalized_*.json` → empty dict

These were deleted before re-running:
```bash
rm -f .cache/v002_concepts_*.json .cache/v002_normalized_*.json
rm -f data/outputs/v002_output.json
```

The transcript cache (`.cache/v002_transcript.json`) was intentionally preserved to skip the 40-minute Whisper step.

### Successful End-to-End Run

With Gemini in place, the full pipeline ran in under 2 minutes (skipping transcription from cache):

```
[normalize]  5 chunks processed, 75 new glossary entries
[extract]    2 chunks → 16 raw concepts → 15 unique after dedup
[prereq]     8 prerequisite edges found
[validate]   ✅ Graph passed on first attempt (0 retries)
[output]     v002_output.json, v002_graph.graphml, v002_graph.html saved
```

---

## 7. Phase 6 — Two-Pass Parallel Transcription

### The Problem

40 minutes of CPU time to transcribe a 25-minute video is the pipeline's dominant cost. With 5 videos, that's potentially **3+ hours of transcription alone** before any LLM work begins.

### Idea Proposed

The user proposed adding a preprocessing agent that speeds up audio (e.g., using `ffmpeg atempo` to run at 1.5×–2× speed) before passing it to Whisper.

### Why Audio Speed-Up Was Rejected

After analysis, this approach was rejected for three reasons specific to this project:

1. **Already solved downstream:** `vad_filter=True` in the transcribe agent already skips all silence frames before they reach Whisper's decoder — which is what speed-up primarily achieves.

2. **Code-mix accuracy degrades:** Time-stretching blurs consonant boundaries and compresses prosodic cues. The switches between Hindi and English are often marked by rhythm and intonation changes — exactly the signal that gets corrupted by `atempo`.

3. **Wrong level of abstraction:** The bottleneck is Whisper's autoregressive decoder on CPU, not audio duration per se. Halving audio duration gives at best a 1.5–2× speedup; parallelism gives 4–6×.

### Alternative: Cascaded Two-Pass + Parallelism

The accepted design combines two orthogonal optimizations:

**Option A (chunked parallelism):** Split the audio into N chunks and transcribe in parallel across CPU cores.

**Option B (model cascade):** Use the fast `tiny` model everywhere, then only upgrade uncertain segments to `small`.

**Combined approach:**
- Pass 1: `tiny` model × 6 workers (all chunks in parallel)
- Confidence filter: flag chunks where `avg_logprob < -0.6` OR `no_speech_prob > 0.3`
- Pass 2: `small` model × 3 workers (flagged chunks only, ~20% of total)
- Merge: combine results, deduplicate 0.5-second boundary overlap

**Expected result for a 25-min video:**

| Stage | Time |
|---|---|
| Split into 6 × 4-min chunks | <1 sec |
| Pass 1: tiny × 6 parallel | ~3–4 min |
| Confidence filter | <1 sec |
| Pass 2: small × 3 parallel (20% of chunks) | ~1–2 min |
| Merge | <1 sec |
| **Total** | **~5–7 min** |

Compared to the original **40 minutes** — roughly a **6–8× speedup**.

### Implementation

**`pipeline/transcribe_fast.py`** (445 lines) — the new agent. Five internal stages, all hidden from LangGraph:

```python
def transcribe_fast_agent(state: PipelineState) -> PipelineState:
    # 1. _split_audio()           — ffmpeg splits into 4-min WAV chunks
    # 2. _run_pass(tiny, ×6)      — ProcessPoolExecutor, all chunks parallel
    # 3. _flag_low_confidence()   — avg_logprob < -0.6 or no_speech_prob > 0.3
    # 4. _run_pass(small, ×3)     — parallel, flagged chunks only
    # 5. _merge_results()         — chronological merge + overlap dedup
```

**Key design decisions:**
- Workers are top-level functions (not closures) — required for `multiprocessing` pickle compatibility
- Chunk WAVs live in a `tempfile.TemporaryDirectory` — auto-cleaned when processing completes
- **Shares the same cache key** as `transcribe.py` — if `v002_transcript.json` exists, both agents skip entirely
- `logical_start` offset is passed to each worker so timestamps in the merged output are correct absolute positions in the original video

**`config.py` additions:**
```python
USE_FAST_TRANSCRIPTION     = true   # master switch (env var)
WHISPER_CHUNK_SECONDS      = 240    # 4 minutes per chunk
WHISPER_OVERLAP_SECONDS    = 0.5    # boundary padding to avoid cutting mid-word
WHISPER_TINY_WORKERS       = 6      # auto-capped at cpu_count
WHISPER_SMALL_WORKERS      = 3      # auto-capped at cpu_count // 2
WHISPER_LOGPROB_THRESHOLD  = -0.6   # flag segments below this log-probability
WHISPER_NO_SPEECH_THRESHOLD = 0.3   # flag segments with >30% non-speech probability
```

**`pipeline/graph.py` update:**
The LangGraph node named `"transcribe"` now routes to the fast or original agent based on the config flag, invisibly to all other pipeline components:

```python
if USE_FAST_TRANSCRIPTION:
    from pipeline.transcribe_fast import transcribe_fast_agent as _transcribe_node
else:
    from pipeline.transcribe import transcribe_agent as _transcribe_node
```

**To disable fast transcription** (e.g., for debugging a transcription issue):
```bash
# in .env
USE_FAST_TRANSCRIPTION=false
```

---

## 8. Final Architecture

### Pipeline Flow

```
                            ┌──────────────────────────────────────────┐
                            │            LANGGRAPH StateGraph           │
                            └──────────────────────────────────────────┘

videos.yaml ──→ main.py
                    │
                    ▼
              ┌──────────┐
              │  ingest  │  yt-dlp downloads audio, ffmpeg converts to 16kHz mono WAV
              └────┬─────┘
                   │ audio_path
                   ▼
         ┌──────────────────┐
         │   transcribe     │  Two-pass: tiny×6 workers → confidence filter → small×3 workers
         │  (fast or slow)  │  OR: sequential small (if USE_FAST_TRANSCRIPTION=false)
         └────────┬─────────┘
                  │ raw_transcript, timestamped_segments
                  ▼
           ┌───────────┐
           │ normalize │  LLM maps code-mixed phrases → standard English
           │           │  Grows shared glossary.json across videos
           └─────┬─────┘
                 │ normalized_transcript, glossary_additions
                 ▼
        ┌─────────────────┐
        │ extract_concepts│  LLM extracts ConceptNode list from transcript chunks
        │                 │  sentence-transformers deduplication (88% cosine threshold)
        └────────┬────────┘
                 │ concepts[]
                 ▼
      ┌────────────────────┐
      │ map_prerequisites  │  LLM reasons about pedagogical ordering
      │                    │  Returns PrerequisiteEdge list with confidence + rationale
      └──────────┬─────────┘
                 │ prerequisite_edges[]
                 ▼
           ┌──────────┐
           │ validate │  Structural checks: cycles, isolated nodes, edge count
           │          │  On failure: LLM critique → revision instructions → retry
           └────┬─────┘
                │
         ┌──────┴──────┐
    [pass]│             │[fail + feedback]
         ▼             ▼
    ┌────────┐    back to extract_concepts
    │ output │    (up to MAX_VALIDATION_RETRIES=3)
    └────────┘
         │
         ▼
   JSON + GraphML + HTML
```

### LLM Call Map

| Agent | Model tier | Approximate calls per video |
|---|---|---|
| normalize | `fast` (gemini-2.5-flash-lite) | 5 calls (one per ~300-word chunk) |
| extract_concepts | `smart` (gemini-2.5-flash-lite) | 2 calls (one per ~800-token chunk) |
| map_prerequisites | `smart` | 1 call |
| validate (critique, if retry) | `smart` | 0–3 calls |
| **Total per video** | | **~8–11 calls** |

---

## 9. File Inventory

| File | Role | Lines |
|---|---|---|
| `config.py` | All configuration knobs + `.env` loading | 103 |
| `state.py` | `PipelineState` + `ConceptNode` + `PrerequisiteEdge` TypedDicts | 59 |
| `main.py` | CLI entrypoint, YAML loading, result summary | 137 |
| `videos.yaml` | 5 video configs: IDs, URLs, language pairs, subjects | 50 |
| `requirements.txt` | Python dependencies | 25 |
| `pipeline/ingest.py` | yt-dlp + ffmpeg audio download | 88 |
| `pipeline/transcribe.py` | Original sequential Whisper-small agent | 141 |
| `pipeline/transcribe_fast.py` | **New** two-pass parallel tiny→small agent | 445 |
| `pipeline/normalize.py` | LLM code-mix normalization + glossary expansion | 193 |
| `pipeline/extract_concepts.py` | LLM concept extraction + embedding dedup | 212 |
| `pipeline/map_prerequisites.py` | LLM prerequisite DAG construction | 148 |
| `pipeline/validate.py` | DAG validation + LLM reflection loop | 214 |
| `pipeline/output.py` | JSON + GraphML + HTML serialization | 107 |
| `pipeline/llm_client.py` | Unified OpenAI/Gemini adapter (`get_completion`) | 136 |
| `pipeline/graph.py` | LangGraph StateGraph orchestrator | 124 |
| `visualization/visualize.py` | Pyvis interactive HTML graph generation | — |
| `visualization/dashboard.py` | Streamlit multi-video dashboard | — |

---

## 10. Key Configuration Reference

All settings live in `config.py` and can be overridden via `.env`:

```bash
# LLM Provider
OPENAI_API_KEY=sk-...          # if set without GEMINI_API_KEY → uses OpenAI
GEMINI_API_KEY=AIza...         # if set without OPENAI_API_KEY → uses Gemini

# Whisper
WHISPER_MODEL_SIZE=small       # tiny / base / small / medium
WHISPER_BEAM_SIZE=5            # lower = faster, less accurate

# Fast Transcription
USE_FAST_TRANSCRIPTION=true    # false = fall back to original transcribe.py
WHISPER_CHUNK_SECONDS=240      # chunk size (4 min default)
WHISPER_OVERLAP_SECONDS=0.5    # boundary overlap to prevent word cutting
WHISPER_TINY_WORKERS=6         # parallel workers for tiny pass
WHISPER_SMALL_WORKERS=3        # parallel workers for small refinement pass
WHISPER_LOGPROB_THRESHOLD=-0.6 # confidence cutoff (lower = less confident)
WHISPER_NO_SPEECH_THRESHOLD=0.3

# Pipeline
MAX_VALIDATION_RETRIES=3
CONCEPT_CHUNK_TOKENS=800
MIN_PREREQUISITE_EDGES=2
CONFIDENCE_THRESHOLD=0.6
```

---

## 11. Verified Output: v002

**Video:** Data Structures - Linked Lists | CodeWithHarry
**URL:** `https://www.youtube.com/watch?v=TWMCMvfEAv4`
**Language mix:** Hindi + English | **Duration:** ~25 min

### Extracted Concepts (15 unique)

| Timestamp | Concept | Domain |
|---|---|---|
| 00:03:45 | Linked List | Computer Science |
| 00:04:00 | Array | Computer Science |
| 00:04:35 | Stack | Computer Science |
| 00:04:35 | Heap | Computer Science |
| 00:04:37 | Stack Memory | Computer Science |
| 00:05:07 | Heap Memory | Computer Science |
| 00:05:30 | Array Declaration | Computer Science |
| 00:05:48 | Dynamic Memory Allocation | Computer Science |
| 00:05:55 | Pointer | Computer Science |
| 00:07:47 | Contiguous Memory Allocation | Computer Science |
| 00:08:00 | Memory Fragmentation | Computer Science |
| 00:10:00 | Pointer Arithmetic | Computer Science |
| 00:14:00 | Node | Computer Science |
| 00:17:00 | Self-Referential Structure | Computer Science |
| 00:17:10 | Data Structure | Computer Science |

### Prerequisite Edges (8 edges)

| From | To | Confidence |
|---|---|---|
| Array | Array Declaration | 0.70 |
| Array | Contiguous Memory Allocation | 0.80 |
| Dynamic Memory Allocation | Heap Memory | 0.60 |
| Pointer | Pointer Arithmetic | 0.80 |
| Linked List | Node | 0.90 |
| Pointer | Node | 0.70 |
| Node | Self-Referential Structure | 0.90 |
| Linked List | Data Structure | 0.50 |

**Validation:** Passed on first attempt. Zero retries needed.

---

## 12. Lessons Learned

### 1. Design for failure from the start
The OpenAI quota failure on the first run would have been catastrophic in a non-resilient pipeline. Because each agent catches its own exceptions and the expensive Whisper step was cached before the LLM calls, recovery required only clearing two small JSON files — not re-running 40 minutes of transcription.

### 2. Thin adapters beat big refactors
When the LLM backend needed to change from OpenAI to Gemini, **only one file needed to be created** (`llm_client.py`). The other 4 LLM-using agents were already written against the `get_completion()` interface, so the migration was a 15-minute job rather than touching 4 files simultaneously.

### 3. Verify model availability before committing to a provider
Three separate issues hit during the Gemini migration:
- The SDK package itself was deprecated (`google-generativeai` → `google-genai`)
- The model name was deprecated (`gemini-1.5-flash` → not found)
- The model had zero quota on the free tier (`gemini-2.0-flash` → limit: 0)

Each required a different fix. The lesson: always enumerate available models programmatically before hardcoding a model name.

### 4. Audio preprocessing must not corrupt the signal it's meant to help
The audio speed-up proposal (ffmpeg `atempo`) seemed like an obvious win but would have actively damaged the code-mixed transcription accuracy — specifically because language-switching cues in Hinglish are encoded in prosody (rhythm, pitch), which time-stretching corrupts. The right solution (parallelism + model cascade) achieved a larger speedup without touching audio fidelity at all.

### 5. The agentic reflection loop earns its complexity
For v002, the validate agent passed on the first attempt — which might make the reflection loop seem unnecessary overhead. But the loop's value is in the *long tail*: videos where the LLM extracts contradictory edges, or where a technical concept appears without context, or where two near-identical concepts are extracted separately. The critique-and-retry mechanism handles these cases automatically rather than silently producing a bad graph.

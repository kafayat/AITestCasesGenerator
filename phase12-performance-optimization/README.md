# Phase 12 - Quality Scoring Agent (Rebuild)

Phase 12 is a from-scratch rebuild of Phase 11 focused on **speed**,
**code structure**, **caching**, and a **streaming UI** — while keeping
Ollama as the model backend and Streamlit as the UI, as requested.

## What changed vs Phase 11

### 1. Concurrent, coverage-group test case generation (biggest speed win)
Phase 11 generated all 12 coverage areas (happy path, negative, boundary,
security, etc.) in **one blocking LLM call** with `num_predict=6000`.
On a local Ollama model that can take minutes, and any JSON parse
failure triggered a **second full LLM call** just to repair the JSON.

Phase 12 splits generation into **4 smaller prompts**
(`prompts/generation_groups/*.txt`), each covering 2-3 related areas, and
runs them **concurrently** via `ThreadPoolExecutor`
(`agents/test_case_generator_agent.py`). Each group has a much lower
`num_predict` ceiling (1400 vs 6000). Wall-clock time becomes roughly
`max(group time)` instead of `sum(all 12 areas)`.

> **Important:** real concurrency requires Ollama itself to be willing to
> serve more than one request at a time. Set this before starting the
> Ollama server:
> ```
> OLLAMA_NUM_PARALLEL=4 ollama serve
> ```
> Without it, Ollama queues the 4 group requests and you won't see the
> speedup (though caching and the smaller per-call ceiling still help).

### 2. Pure-Python JSON repair before any LLM repair call
`utils/json_repair.py` now tries, in order:
1. Plain `json.loads`
2. The `json_repair` PyPI package (handles trailing commas, missing
   quotes, unbalanced brackets — fixes most real-world cases)
3. A dependency-free manual bracket-balancing pass
4. Only if all of that fails: one LLM repair call (same as Phase 11)

This removes the "double LLM call" tax for the common case of
near-valid JSON.

### 3. Response caching (`utils/cache.py`)
Every generation-group call and the quality/review/inference calls are
keyed by a hash of `(model, task, prompt)`. Re-running the same
requirement against the same knowledge base is **instant** on a cache
hit. Useful while iterating on prompts or demoing repeatedly. Clear it
any time from the sidebar ("Clear response cache") or by deleting the
`cache/` folder.

### 4. Task-based model routing (`utils/model_provider.py`)
Phase 11 used one model instance with one fixed `num_predict=6000` for
every agent. Phase 12 routes by task:

| Task        | Default num_predict | Notes                                   |
|-------------|---------------------|------------------------------------------|
| generation  | 1400                | called 4x concurrently per run           |
| analysis    | 500                 | only used when Fast Mode is off          |
| scoring     | 900                 | called once per run                      |
| review      | 700                 | called once per run, optional            |
| inference   | 900                 | only used when no requirement is found   |

The sidebar now exposes a separate **Generation Model** (fast, called
often) and **Quality / Review Model** (can be a larger model, called
once), instead of one model for everything.

### 5. Streaming UI for single-call agents
Requirement analysis, requirement inference, quality scoring, and
reviewer feedback now stream tokens into the page live
(`stream_text_response` in `app.py`) instead of a blank spinner followed
by a wall of text. True per-token streaming across the 4 concurrent
generation threads isn't practical in Streamlit's single-threaded
renderer, so generation instead shows **live per-group status chips**
(Queued → Generating... → Done) that update as each group finishes —
giving the same "something is happening" feedback without fighting the
framework.

### 6. Code structure
- `agents/test_case_generator_agent.py` is now organized around a single
  `generate_test_cases_concurrent()` entry point with a clear per-group
  helper (`_generate_single_group`) that's easy to unit test in
  isolation.
- The old single-call `generate_test_cases()` function is kept (unused
  by `app.py`) for backward compatibility / fallback.
- Merged test cases get deterministic, collision-free IDs
  (`TC-A01`, `TC-B01`, `TC-C01`, `TC-D01`, ...) keyed by coverage group,
  so two groups both returning "TC001" internally never collide after
  merging.

## What stayed the same
- Streamlit as the UI framework
- Ollama as the model backend
- RAG retrieval (`utils/rag_loader.py`), document classification
  (`utils/knowledge_classifier.py`), file extraction
  (`utils/file_handler.py`), and Excel/Word export
  (`utils/output_writer.py`) are unchanged — they weren't the bottleneck.
- `models/test_case_model.py` (Pydantic schema) unchanged.

## Known limitations / next steps
- Concurrency speedup depends on Ollama's `OLLAMA_NUM_PARALLEL` setting
  and available hardware (GPU memory in particular — running 4 model
  instances concurrently uses more VRAM than 1).
- The file-based cache is for development/demo speed, not a production
  cache (no eviction policy beyond manual clearing).
- True token-level streaming for the 4 concurrent generation groups was
  intentionally left out in favor of per-group status chips, since
  Streamlit can't easily render 4 simultaneous token streams without a
  more complex async/websocket setup.

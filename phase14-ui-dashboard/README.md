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

---

# Phase 13 - Coverage & Gap Analysis

Phase 13 adds a deterministic Coverage & Gap Analysis section, computed
entirely in Python (`utils/coverage_analysis.py`) from data Phase 12
already produces - no extra LLM calls, so it's instant and doesn't add
generation time.

## Why deterministic instead of LLM-judged

Phase 12's Quality Scoring Agent asks the LLM to self-report coverage
("Coverage Score: 95%", "Missing Scenarios: None"). In real testing this
was unreliable - it once said "Business Rule Coverage: Not applicable"
on a run that had 3 business-rule test cases, and "Missing Scenarios:
None" when a real acceptance criterion (confirmation email) had no
matching test case at all.

Phase 13 replaces what can be checked in code with code:

- **Coverage matrix** - every test case's `coverage_area` field is
  tallied against the fixed list of 12 known areas. Any area with zero
  test cases is a real, visible gap (`MISSING`), not a guess.
- **Acceptance criteria traceability** - bullet lines are extracted from
  the requirement text and matched against test case content using
  IDF-weighted keyword overlap (see below) - any criterion with no
  matching test case is flagged.
- **Business rule traceability** - same approach, applied to the bullet
  rules in `knowledge_base/business_rules/*.txt`.
- **Risk coverage** - test cases are bucketed into High/Medium/Low risk
  tiers based on their `coverage_area` (security, business rule, data
  integrity, and failure-handling areas are tier "High" by default -
  editable in `HIGH_RISK_AREAS`/`MEDIUM_RISK_AREAS` in
  `utils/coverage_analysis.py` without touching any LLM prompt).

The LLM-based Quality Scoring Agent from Phase 12 is kept as-is for the
parts that genuinely need judgment (qualitative recommendations) - it
just no longer has to be trusted for the parts that can be checked
mechanically.

## Matching approach and its limits

Matching is plain keyword overlap weighted by inverse document
frequency (a word that appears in nearly every test case, like
"payment" or "account" in this domain, counts for very little; a word
that appears in only one or two test cases, like "duplicate" or
"boundary", counts for a lot). This is intentionally simple, fast, and
auditable - no embeddings or extra dependencies - but it is NOT semantic
understanding. A criterion phrased very differently from the matching
test case's wording can still show as a false gap. Treat the report as
a strong starting point for a human reviewer, not a certified coverage
audit.

## What's in the UI

After test case generation, a new **"Coverage & Gap Analysis"** section
shows:
- 3 summary metrics (coverage areas hit, acceptance criteria covered,
  business rules covered)
- Expandable lists of uncovered criteria and uncovered rules
- The full coverage matrix and risk tier breakdown
- A **"Download Coverage & Gap Analysis Report (Markdown)"** button,
  alongside the existing Excel/Word downloads

## Files added/changed
- `utils/coverage_analysis.py` (new) - all gap analysis logic
- `app.py` - new Coverage & Gap Analysis section wired in after
  generation, before the Quality Scoring report

---

## Phase 14 update log

### Fix: false-positive contradiction flag on legitimate missing-field test cases

`utils/quality_checks.py`'s precondition/steps contradiction check
(added to catch real bugs like a test case saying "a valid submission
is made" while its own steps "enter invalid data") was too blunt: it
also flagged the standard, CORRECT way of writing a missing-mandatory-
field test case - e.g. preconditions "all fields completed **except**
exporter reference" + steps "leave exporter reference empty" - as a
contradiction. That's not a contradiction, it's exactly how this kind
of negative test case should be written.

Fixed by checking for partial-state language ("except", "excluding",
"other than", "apart from", "leaving out") in the preconditions text
before treating a "valid/successful" signal word as asserting full
validity. Verified: the legitimate missing-field pattern is no longer
flagged, while the original real bug (TC-D03-style genuine
contradiction) is still caught correctly.

### Generation collapse investigation (open)

A real run against the NEXDOC requirement produced only 1 test case
across 5 coverage groups (expected ~12-18). A full pipeline replay using
realistic, well-formed mock LLM responses for all 5 groups confirmed the
Python-side filtering, dedup, and merge logic are NOT the cause - 11/11
realistic test cases survived correctly with zero false drops. The
collapse is most likely happening at the Ollama response level (4 of 5
concurrent calls returning malformed/empty output, possibly from
resource contention running 5 concurrent model calls on limited local
hardware). Still needs the `outputs/generation_debug.log` from an actual
failing run to confirm definitively - this is the next thing to
diagnose with real run data.

### PDF upload confirmed working

`utils/file_handler.py`'s PDF extraction was verified directly against
the real NEXDOC PDF (6 pages, 6853 characters, all use cases and
acceptance criteria extracted correctly). PDF upload is not a known
issue.

### Critical fix: copy-detection filter was built but never actually enabled

The `use_copy_filter` flag in `generate_test_cases_concurrent()` /
`_generate_single_group()` defaulted to `False`, and `app.py` never
passed `True` - meaning the tested, working copy-detection filter from
the previous update had **never actually run in any real session**.

A real NEXDOC run confirmed this concretely: 3 of 5 groups (negative_
boundary, failure_security_error, lifecycle_operations) returned 100%
verbatim copies of their own prompt's worked examples - 7 fake "test
cases" with zero domain-specific content, all silently accepted because
the safety net that exists specifically to catch this was switched off.

Fixed by flipping the default to `use_copy_filter=True`. Verified against
the exact verbatim content from the real failing run - all 7 copies are
now correctly dropped - and re-verified that legitimate, domain-specific
content (e.g. mentioning "REX lodgement", "destination country") still
survives untouched.

### Quality polish after first qwen2.5:3b run (real improvement, smaller gaps remaining)

A real run on qwen2.5:3b produced 11 genuinely domain-specific test
cases (the model-size fix worked) but surfaced 3 smaller issues:

1. **Empty `test_data` fields** in several test cases (TC-B01, TC-C01,
   etc. had a blank Test Data column in the exported doc). The prompts
   already said "never leave empty", but a 3B model doesn't always
   comply. Fixed in code: `normalize_test_cases` now falls back to an
   explicit "Not explicitly specified by the model..." message for
   test_data, preconditions, and expected_result if the model leaves
   them blank - non-empty fields are left untouched.

2. **No security test case generated**, despite the requirement
   explicitly naming "Security and access control" as a required
   coverage area in its coverage-areas list, just without a detailed
   use case describing it. `group_4_failure_security_error.txt` now
   explicitly instructs the model not to skip a named-but-undetailed
   coverage area, and to infer a reasonable generic security test case
   (e.g. role-based access) from the requirement's named actors/roles.

3. **Quality Scoring report still self-contradicted reality**
   ("Business Rule Coverage: Not applicable" on a run with business-rule
   test cases). Added `format_facts_for_quality_prompt()` in
   `utils/coverage_analysis.py`, which computes real facts (coverage
   areas hit/missing, business-rule test case count) and injects them
   into the quality scoring prompt as instructed ground truth - the LLM
   now only exercises judgment on the parts that actually need it
   (Quality Issues, Recommendations), not on facts that are already
   computed.

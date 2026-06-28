"""
Phase 11 generated all 12 coverage areas in ONE blocking LLM call with
num_predict=6000. That call alone could take minutes on a local model,
and any JSON parse failure doubled it with a second full LLM repair call.

Phase 12 splits generation into 4 small coverage-group calls and runs
them CONCURRENTLY (ThreadPoolExecutor). Each group call:
  - asks for only 2-3 coverage areas instead of 12
  - has a much lower num_predict ceiling
  - is cached by content hash, so re-running the same requirement/RAG
    context is instant on a cache hit
  - is repaired with pure-Python JSON repair first (see utils/json_repair),
    only falling back to an LLM repair call if that truly fails

After a real run against Sydney Water sample data, two more problems
showed up that this version fixes:
  1. The model sometimes ignored "ONLY generate these coverage areas"
     and generated a different scenario (most often whichever rule was
     most prominent in the RAG context, e.g. duplicate payment) in 3 of
     4 groups. Each test case now must include a `coverage_area` field;
     any test case whose coverage_area isn't one this group is allowed
     to produce gets dropped before merging, instead of silently
     polluting the output.
  2. Truncated generations (model ran out of tokens mid-array) used to
     get silently bracket-padded into "valid but incomplete" JSON. Now
     `parse_json_safely` reports when that happened and the group is
     flagged as possibly truncated.

Net effect: wall-clock time is roughly max(group time) instead of
sum(all 12 areas in one call), as long as Ollama is configured to serve
more than one request at a time (OLLAMA_NUM_PARALLEL > 1).

Streaming: each group call now optionally streams tokens into a shared
thread-safe queue (stream_queue) as they're generated, so a caller (e.g.
a Streamlit UI polling the queue from the main thread) can show live
text per coverage group instead of a static "Generating..." chip.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from utils.prompt_loader import load_prompt
from utils.json_repair import parse_json_safely, normalize_test_cases
from utils.cache import make_cache_key, get_cached, set_cached


GROUPS_DIR = "prompts/generation_groups"

# (group_id, prompt_file, id_prefix, allowed_coverage_areas) - id_prefix
# keeps test_case_id's unique across groups. allowed_coverage_areas is
# the set of coverage_area values this group is permitted to produce;
# anything else gets dropped during merge (see _filter_to_allowed_areas).
GENERATION_GROUPS = [
    (
        "happy_path",
        "group_1_happy_path.txt",
        "A",
        {
            "Happy path / successful flow",
            "Confirmation or notification",
            "Receipt or reference generation",
        },
    ),
    (
        "negative_boundary",
        "group_2_negative_boundary.txt",
        "B",
        {
            "Invalid input",
            "Missing mandatory data",
            "Boundary condition",
        },
    ),
    (
        "business_rules_integrity",
        "group_3_business_rules_integrity.txt",
        "C",
        {
            "Business rule validation",
            "Duplicate processing",
            "Data integrity",
        },
    ),
    (
        "failure_security_error",
        "group_4_failure_security_error.txt",
        "D",
        {
            "Failed transaction or system failure",
            "Security or sensitive data handling",
            "Error handling",
        },
    ),
    (
        "lifecycle_operations",
        "group_5_lifecycle_operations.txt",
        "E",
        {
            "Amendment or modification of an existing record",
            "Replacement or reissue of an existing record",
            "Document or certificate generation tied to a lifecycle event",
        },
    ),
]


def _build_prompt(group_file, requirement, analysis, rag_context):

    prompt_template = load_prompt(
        os.path.join(GROUPS_DIR, group_file)
    )

    return (
        prompt_template
        .replace("{requirement}", requirement)
        .replace("{analysis}", analysis)
        .replace("{rag_context}", rag_context)
    )


_example_titles_cache = {}


def _extract_example_titles(group_file):
    """
    Pulls the title (and steps, joined) of every test case in a group
    prompt's own "Worked example" JSON block, so generated output can be
    checked against it later. This is what catches the real failure
    mode seen with the lifecycle_operations group: with a concrete,
    well-written example sitting in the prompt, a low-temperature small
    model's single most probable continuation is often to just recite
    that example back almost verbatim, with zero adaptation to the
    actual requirement. Comparing generated titles/steps against the
    prompt's OWN example content (rather than guessing at a fixed list)
    means this stays in sync automatically if the example is ever edited.

    Cached per group_file since the prompt file content doesn't change
    during a run.
    """

    if group_file in _example_titles_cache:
        return _example_titles_cache[group_file]

    raw_prompt = load_prompt(os.path.join(GROUPS_DIR, group_file))

    example_marker = "Worked example"
    end_marker = "Return valid JSON only"

    start_index = raw_prompt.find(example_marker)
    end_index = raw_prompt.find(end_marker, start_index if start_index != -1 else 0)

    example_entries = []

    if start_index != -1 and end_index != -1:

        block = raw_prompt[start_index:end_index]

        json_start = block.find("{")
        json_end = block.rfind("}")

        if json_start != -1 and json_end != -1:

            try:
                parsed = json.loads(block[json_start:json_end + 1])

                for test_case in parsed.get("test_cases", []):
                    example_entries.append({
                        "title": test_case.get("title", ""),
                        "steps_text": " ".join(test_case.get("steps", [])),
                    })

            except Exception:
                # If the example block can't be parsed for some reason,
                # just skip the copy-detection check for this group
                # rather than failing generation over it.
                pass

    _example_titles_cache[group_file] = example_entries

    return example_entries


def _filter_copied_examples(test_cases, group_file, threshold=0.88):
    """
    Drops any generated test case whose title OR steps are near-identical
    to one of the group prompt's own worked examples - the smoking-gun
    sign that the model recited the example instead of using the actual
    requirement. Real generated content about a specific domain (REX
    lodgements, dairy/honey, exporters) essentially never scores this
    high against a deliberately generic, abstract example.
    """

    example_entries = _extract_example_titles(group_file)

    if not example_entries:
        return test_cases, 0

    kept = []
    copied_count = 0

    for test_case in test_cases:

        title = test_case.get("title", "")
        steps_text = " ".join(test_case.get("steps", []))

        is_copied = any(
            _title_similarity(title, example["title"]) >= threshold
            or (steps_text and example["steps_text"]
                and _title_similarity(steps_text, example["steps_text"]) >= threshold)
            for example in example_entries
        )

        if is_copied:
            copied_count += 1
        else:
            kept.append(test_case)

    return kept, copied_count


def _normalize_area_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _coverage_area_matches(candidate_area, allowed_area):
    """
    Was previously an exact string-equality check after normalization,
    which turned out to be dangerously brittle: at a higher generation
    temperature (raised specifically to reduce verbatim example-copying),
    the model became more likely to phrase coverage_area with small
    variations - a trailing period, an extra word ("Invalid input
    scenario"), an added prefix ("Negative - Invalid input"). Any
    deviation at all caused a 100% match failure, silently dropping a
    test case that was actually correctly categorized. Across several
    groups this could wipe out almost the entire run, which is exactly
    what happened in a real test.

    This now accepts a match if the allowed area's normalized text is
    a substring of the candidate (or vice versa), which absorbs minor
    wording drift while still correctly rejecting a genuinely different
    area (e.g. "Happy path / successful flow" vs "Invalid input" share
    no meaningful substring).
    """

    candidate_norm = _normalize_area_text(candidate_area)
    allowed_norm = _normalize_area_text(allowed_area)

    if candidate_norm == allowed_norm:
        return True

    if allowed_norm in candidate_norm or candidate_norm in allowed_norm:
        return True

    return False


def _filter_to_allowed_areas(test_cases, allowed_areas, group_id):
    """
    Drops any test case whose coverage_area doesn't belong to this
    group's allowed set. If coverage_area is missing entirely (older
    prompt / model didn't include it), the test case is kept but
    flagged, since rejecting it outright would silently lose coverage
    just because of a missing label.
    """

    kept = []
    dropped_count = 0

    for test_case in test_cases:

        coverage_area = test_case.get("coverage_area", "").strip()

        if not coverage_area:
            test_case["coverage_area"] = "Unspecified"
            kept.append(test_case)
            continue

        matched_area = next(
            (area for area in allowed_areas if _coverage_area_matches(coverage_area, area)),
            None
        )

        if matched_area:
            # Snap to the canonical allowed-area string so downstream
            # exact-match lookups (e.g. the coverage matrix in
            # utils/coverage_analysis.py) still recognize it, even
            # though the model's wording drifted slightly.
            test_case["coverage_area"] = matched_area
            kept.append(test_case)
        else:
            dropped_count += 1

    return kept, dropped_count


def _title_similarity(text_a, text_b):
    return SequenceMatcher(
        None,
        _normalize_area_text(text_a),
        _normalize_area_text(text_b),
    ).ratio()


def _dedupe_by_title_similarity(test_cases, threshold=0.80, steps_threshold=0.92):
    """
    Drops test cases that are near-duplicates of one already kept, using
    two checks:
      1. Title similarity >= threshold (catches same-scenario, similar
         wording - e.g. 3 groups all generating "successful payment").
      2. Steps similarity >= steps_threshold even if titles differ
         (catches cases where the model reused identical steps for two
         supposedly-different behaviours, e.g. receipt vs confirmation
         email both just saying "complete payment, check screen").
    """

    kept = []
    duplicate_ids = []

    for test_case in test_cases:

        title = test_case.get("title", "")
        steps_text = " ".join(test_case.get("steps", []))

        is_duplicate = False

        for existing in kept:

            existing_title = existing.get("title", "")
            existing_steps_text = " ".join(existing.get("steps", []))

            if _title_similarity(title, existing_title) >= threshold:
                is_duplicate = True
                break

            if steps_text and existing_steps_text:
                if _title_similarity(steps_text, existing_steps_text) >= steps_threshold:
                    is_duplicate = True
                    break

        if is_duplicate:
            duplicate_ids.append(test_case.get("test_case_id", ""))
        else:
            kept.append(test_case)

    return kept, duplicate_ids


def _generate_single_group(
    llm,
    group_id,
    group_file,
    id_prefix,
    allowed_areas,
    requirement,
    analysis,
    rag_context,
    model_name,
    requirement_id,
    use_cache=True,
    stream_queue=None,
    use_copy_filter=True,
):
    """
    Runs ONE coverage-group generation call. Designed to be safe to call
    from a worker thread - returns a result dict rather than raising, so
    one failing group never crashes the others.

    If stream_queue is provided, pushes ("chunk", group_id, accumulated_text)
    messages to it as tokens arrive (or once, immediately, on a cache
    hit) so a caller polling the queue from the main thread can render
    live per-group text instead of a static status chip.
    """

    prompt = _build_prompt(group_file, requirement, analysis, rag_context)

    cache_key = make_cache_key(
        "generation", model_name, group_id, prompt
    )

    raw_text = get_cached(cache_key) if use_cache else None
    from_cache = raw_text is not None

    if raw_text is None:

        accumulated = ""

        try:
            for chunk in llm.stream(prompt):
                accumulated += chunk
                if stream_queue is not None:
                    stream_queue.put(("chunk", group_id, accumulated))

            raw_text = accumulated

        except Exception:
            # Streaming not supported by this LLM wrapper - fall back to
            # a single blocking call.
            raw_text = llm.invoke(prompt)
            if stream_queue is not None:
                stream_queue.put(("chunk", group_id, raw_text))

        if use_cache:
            set_cached(cache_key, raw_text)

    elif stream_queue is not None:
        # Cache hit - still post the full text once so the UI shows
        # something immediately instead of an empty panel.
        stream_queue.put(("chunk", group_id, raw_text))

    was_truncated = False
    dropped_off_topic = 0
    error = None

    try:
        parsed, was_truncated = parse_json_safely(raw_text, llm)
        parsed = normalize_test_cases(parsed, default_requirement_id=requirement_id)
    except Exception as exc:
        parsed = {"test_cases": []}
        error = str(exc)

    test_cases = parsed.get("test_cases", [])

    # Drop test cases generated for the wrong coverage area before they
    # ever reach the merge step.
    test_cases, dropped_off_topic = _filter_to_allowed_areas(
        test_cases, allowed_areas, group_id
    )

    # Drop test cases that are just the model reciting this group's own
    # worked example back verbatim instead of using the actual
    # requirement - see _filter_copied_examples for why this happens.
    # Opt-in (default off): this filter is newer and less proven than
    # the others, so it shouldn't risk dropping legitimate content by
    # default until it's been validated on more real runs.
    if use_copy_filter:
        test_cases, dropped_copied_example = _filter_copied_examples(
            test_cases, group_file
        )
    else:
        dropped_copied_example = 0

    # Force a single consistent requirement_id - never trust the model
    # to keep this consistent across 4 independent calls.
    for index, test_case in enumerate(test_cases, start=1):
        test_case["test_case_id"] = f"TC-{id_prefix}{index:02d}"
        test_case["requirement_id"] = requirement_id

    return {
        "group_id": group_id,
        "raw_text": raw_text,
        "test_cases": test_cases,
        "error": error,
        "from_cache": from_cache,
        "was_truncated": was_truncated,
        "dropped_off_topic": dropped_off_topic,
        "dropped_copied_example": dropped_copied_example,
    }


def generate_test_cases_concurrent(
    llm,
    requirement,
    analysis,
    rag_context,
    model_name="unknown-model",
    requirement_id="REQ001",
    use_cache=True,
    progress_callback=None,
    stream_queue=None,
    use_copy_filter=True,
):
    """
    Runs all coverage groups concurrently and merges the results.

    progress_callback(group_id, status) is called with status in
    {"started", "done", "error", "truncated"} so the UI can show live
    progress without needing true token-level streaming across threads.

    stream_queue, if provided, receives ("chunk", group_id, text) messages
    as each group streams tokens, and a final ("group_done", group_id, status)
    message when a group finishes. This is designed to be polled from a
    different thread than the one running this function (e.g. a
    Streamlit main thread, while this function itself runs in a
    background thread) - see app.py for the polling loop.

    use_copy_filter (default True): drops test cases near-identical to a
    group's own worked example. This used to default to False ("newer/
    less proven") - real runs against NEXDOC proved that was wrong: 3 of
    5 groups returned 100% verbatim copies of their own examples with
    this filter sitting unused. Now on by default; pass False explicitly
    only if you need to debug raw model output without any filtering.

    Returns: (merged_result_dict, group_results_list, duplicate_ids)
    """

    group_results = []

    with ThreadPoolExecutor(max_workers=len(GENERATION_GROUPS)) as executor:

        future_to_group = {}

        for group_id, group_file, id_prefix, allowed_areas in GENERATION_GROUPS:

            if progress_callback:
                progress_callback(group_id, "started")

            future = executor.submit(
                _generate_single_group,
                llm,
                group_id,
                group_file,
                id_prefix,
                allowed_areas,
                requirement,
                analysis,
                rag_context,
                model_name,
                requirement_id,
                use_cache,
                stream_queue,
                use_copy_filter,
            )

            future_to_group[future] = group_id

        for future in as_completed(future_to_group):

            group_id = future_to_group[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "group_id": group_id,
                    "raw_text": "",
                    "test_cases": [],
                    "error": str(exc),
                    "from_cache": False,
                    "was_truncated": False,
                    "dropped_off_topic": 0,
                    "dropped_copied_example": 0,
                }

            group_results.append(result)

            if progress_callback:
                if result["error"]:
                    status = "error"
                elif result["was_truncated"]:
                    status = "truncated"
                else:
                    status = "done"
                progress_callback(group_id, status)

            if stream_queue is not None:
                if result["error"]:
                    done_status = "error"
                elif result["was_truncated"]:
                    done_status = "truncated"
                else:
                    done_status = "done"
                stream_queue.put(("group_done", group_id, done_status))

    merged_test_cases = []

    for result in group_results:
        merged_test_cases.extend(result["test_cases"])

    merged_test_cases, duplicate_ids = _dedupe_by_title_similarity(
        merged_test_cases
    )

    merged_result = {
        "test_cases": merged_test_cases
    }

    if stream_queue is not None:
        stream_queue.put(("all_done", None, None))

    return merged_result, group_results, duplicate_ids


# Kept for backward compatibility / single-call use cases (e.g. if a
# caller wants the old "everything in one prompt" behaviour). Not used
# by the rebuilt app.py, but harmless to keep around.
def generate_test_cases(llm, requirement, analysis, rag_context):

    prompt_template = load_prompt(
        "prompts/test_case_generator_prompt.txt"
    )

    prompt = (
        prompt_template
        .replace("{requirement}", requirement)
        .replace("{analysis}", analysis)
        .replace("{rag_context}", rag_context)
    )

    return llm.invoke(prompt)


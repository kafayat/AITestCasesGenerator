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

Net effect: wall-clock time is roughly max(group time) instead of
sum(all 12 areas in one call), as long as Ollama is configured to serve
more than one request at a time (OLLAMA_NUM_PARALLEL > 1).
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.prompt_loader import load_prompt
from utils.json_repair import parse_json_safely, normalize_test_cases
from utils.cache import make_cache_key, get_cached, set_cached


GROUPS_DIR = "prompts/generation_groups"

# (group_id, prompt_file, id_prefix) - id_prefix keeps test_case_id's
# unique across groups even though each group is generated independently.
GENERATION_GROUPS = [
    ("happy_path", "group_1_happy_path.txt", "A"),
    ("negative_boundary", "group_2_negative_boundary.txt", "B"),
    ("business_rules_integrity", "group_3_business_rules_integrity.txt", "C"),
    ("failure_security_error", "group_4_failure_security_error.txt", "D"),
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


def _generate_single_group(
    llm,
    group_id,
    group_file,
    id_prefix,
    requirement,
    analysis,
    rag_context,
    model_name,
    use_cache=True,
):
    """
    Runs ONE coverage-group generation call. Designed to be safe to call
    from a worker thread - returns a result dict rather than raising, so
    one failing group never crashes the others.
    """

    prompt = _build_prompt(group_file, requirement, analysis, rag_context)

    cache_key = make_cache_key(
        "generation", model_name, group_id, prompt
    )

    raw_text = get_cached(cache_key) if use_cache else None
    from_cache = raw_text is not None

    if raw_text is None:
        raw_text = llm.invoke(prompt)

        if use_cache:
            set_cached(cache_key, raw_text)

    try:
        parsed = parse_json_safely(raw_text, llm)
        parsed = normalize_test_cases(parsed, default_requirement_id="REQ001")
        error = None
    except Exception as exc:
        parsed = {"test_cases": []}
        error = str(exc)

    # Re-prefix test_case_id's so merged output never collides across
    # groups, even if the model reused "TC001" in every group.
    for index, test_case in enumerate(parsed.get("test_cases", []), start=1):
        test_case["test_case_id"] = f"TC-{id_prefix}{index:02d}"

    return {
        "group_id": group_id,
        "raw_text": raw_text,
        "test_cases": parsed.get("test_cases", []),
        "error": error,
        "from_cache": from_cache,
    }


def generate_test_cases_concurrent(
    llm,
    requirement,
    analysis,
    rag_context,
    model_name="unknown-model",
    use_cache=True,
    progress_callback=None,
):
    """
    Runs all coverage groups concurrently and merges the results.

    progress_callback(group_id, status) is called with status in
    {"started", "done", "error"} so the UI can show live progress
    without needing true token-level streaming across threads.

    Returns: (merged_result_dict, group_results_list)
    """

    group_results = []

    with ThreadPoolExecutor(max_workers=len(GENERATION_GROUPS)) as executor:

        future_to_group = {}

        for group_id, group_file, id_prefix in GENERATION_GROUPS:

            if progress_callback:
                progress_callback(group_id, "started")

            future = executor.submit(
                _generate_single_group,
                llm,
                group_id,
                group_file,
                id_prefix,
                requirement,
                analysis,
                rag_context,
                model_name,
                use_cache,
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
                }

            group_results.append(result)

            if progress_callback:
                status = "error" if result["error"] else "done"
                progress_callback(group_id, status)

    merged_test_cases = []

    for result in group_results:
        merged_test_cases.extend(result["test_cases"])

    merged_result = {
        "test_cases": merged_test_cases
    }

    return merged_result, group_results


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

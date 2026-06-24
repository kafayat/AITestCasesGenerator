"""
Phase 11 problem: any time the model's JSON wasn't perfectly parseable,
parse_json_safely() fell back to a SECOND full LLM call just to "fix" the
JSON. On a local model that can double the wall-clock time of generation
for what is often a one-character problem (trailing comma, stray brace,
unescaped quote).

Phase 12 fix: try increasingly aggressive pure-Python repairs first
(cheap, milliseconds) and only call the LLM as an absolute last resort.

Repair order:
1. Strip markdown fences / leading-trailing junk (same as Phase 11).
2. Try the `json_repair` library if installed (handles trailing commas,
   missing quotes, unbalanced brackets, etc. - this fixes the vast
   majority of real-world malformed LLM JSON).
3. Try a manual bracket-balancing pass (no external dependency needed).
4. Only if all of that fails, fall back to an LLM repair call.
"""

import json
import re


def clean_ai_json(raw_text):

    cleaned = raw_text.strip()

    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")

    cleaned = re.sub(r"//.*", "", cleaned)

    cleaned = cleaned.replace("{{", "{")
    cleaned = cleaned.replace("}}", "}")

    start_index = cleaned.find("{")
    end_index = cleaned.rfind("}")

    if start_index != -1 and end_index != -1:
        cleaned = cleaned[start_index:end_index + 1]

    return cleaned


def _try_json_repair_library(text):
    """
    Uses the third-party `json_repair` package if it is installed.
    Returns a Python object on success, or None if unavailable / failed.
    """

    try:
        from json_repair import repair_json
    except ImportError:
        return None

    try:
        repaired_text = repair_json(text)
        return json.loads(repaired_text)
    except Exception:
        return None


def _try_manual_bracket_balance(text):
    """
    Last-resort, dependency-free repair: trims trailing commas before
    closing brackets and balances any missing closing braces/brackets.
    Cheap and catches a large share of truncated-output cases (the model
    ran out of tokens mid-array).
    """

    candidate = text

    # Remove trailing commas before a closing bracket/brace.
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")

    # Pad missing closing brackets/braces (handles truncated output).
    if open_brackets > close_brackets:
        candidate += "]" * (open_brackets - close_brackets)

    if open_braces > close_braces:
        candidate += "}" * (open_braces - close_braces)

    try:
        return json.loads(candidate)
    except Exception:
        return None


def parse_json_safely(raw_text, llm=None, allow_llm_repair=True):

    cleaned_json = clean_ai_json(raw_text)

    # 1. Straightforward parse.
    try:
        return json.loads(cleaned_json)
    except Exception:
        pass

    # 2. `json_repair` library, if installed.
    repaired = _try_json_repair_library(cleaned_json)
    if repaired is not None:
        return repaired

    # 3. Manual bracket balancing, no dependency required.
    repaired = _try_manual_bracket_balance(cleaned_json)
    if repaired is not None:
        return repaired

    # 4. Last resort: ask the LLM to fix it (slow - only happens when
    #    everything above failed).
    if not allow_llm_repair or llm is None:
        # Re-raise the original parse error so callers see a real
        # json.JSONDecodeError rather than silently failing.
        return json.loads(cleaned_json)

    repair_prompt = f"""
Return VALID JSON only.

Fix this content into valid JSON.

Rules:
- No markdown
- No explanation
- No comments
- Use single opening brace only
- Use single closing brace only
- Root must contain test_cases array

Content:
{cleaned_json}
"""

    repaired_raw = llm.invoke(repair_prompt)
    repaired_cleaned = clean_ai_json(repaired_raw)

    return json.loads(repaired_cleaned)


def convert_to_string(value):

    if isinstance(value, str):
        return value

    if value is None:
        return ""

    return json.dumps(
        value,
        indent=2
    )


def normalize_test_cases(
    test_case_data,
    default_requirement_id="REQ001"
):

    if "test_cases" not in test_case_data:
        test_case_data["test_cases"] = []

    for test_case in test_case_data["test_cases"]:

        test_case.setdefault("test_case_id", "")
        test_case.setdefault("requirement_id", default_requirement_id)
        test_case.setdefault("title", "")
        test_case.setdefault("priority", "")
        test_case.setdefault("test_type", "")
        test_case.setdefault("preconditions", "")
        test_case.setdefault("test_data", "")
        test_case.setdefault("steps", [])
        test_case.setdefault("expected_result", "")
        test_case.setdefault("automation_candidate", "Yes")
        test_case.setdefault("source_traceability", "")

        if not test_case.get("requirement_id"):
            test_case["requirement_id"] = default_requirement_id

        test_case["preconditions"] = convert_to_string(
            test_case.get("preconditions")
        )

        test_case["test_data"] = convert_to_string(
            test_case.get("test_data")
        )

        test_case["expected_result"] = convert_to_string(
            test_case.get("expected_result")
        )

        if isinstance(test_case.get("steps"), str):
            test_case["steps"] = [test_case["steps"]]

        if not isinstance(test_case.get("steps"), list):
            test_case["steps"] = [
                convert_to_string(
                    test_case.get("steps")
                )
            ]

        cleaned_steps = []

        for step in test_case["steps"]:

            cleaned_steps.append(
                convert_to_string(step).strip()
            )

        test_case["steps"] = cleaned_steps

        test_case["automation_candidate"] = convert_to_string(
            test_case.get("automation_candidate")
        )

        test_case["source_traceability"] = convert_to_string(
            test_case.get("source_traceability")
        )

    return test_case_data

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
    closing brackets and balances any missing closing braces/brackets
    in the CORRECT NESTED ORDER (using a stack), not just by appending
    all "]" then all "}" - naive count-based padding produces wrongly
    nested JSON like "]]}}" when the correct close order is "]}]}".

    IMPORTANT: padding in missing brackets keeps whatever was generated
    before the cutoff and silently drops everything after it (e.g. a
    half-written "steps" array just gets closed where it stopped). That
    produces JSON that parses fine but is INCOMPLETE - so this function
    reports whether padding was needed, and the caller must surface that
    as a possible-truncation warning rather than presenting it as clean
    output.

    Returns (parsed_object_or_none, was_truncated: bool).
    """

    candidate = re.sub(r",\s*([\]}])", r"\1", text)

    # Walk the string tracking only structural brackets/braces that are
    # OUTSIDE of string literals, so we don't get confused by "[" or "{"
    # appearing inside a quoted title/description.
    stack = []
    in_string = False
    escape_next = False

    for char in candidate:

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "{[":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if in_string:
        # An unterminated string literal - close it before padding
        # brackets, otherwise the padded brackets would be swallowed
        # into the open string.
        candidate += '"'

    was_truncated = len(stack) > 0

    # Close whatever is left open, innermost first (reverse of the
    # order they were opened), which is the only valid nesting order.
    for opener in reversed(stack):
        candidate += "}" if opener == "{" else "]"

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None, False

    return parsed, was_truncated


def parse_json_safely(raw_text, llm=None, allow_llm_repair=True):
    """
    Returns (parsed_object, was_truncated: bool).

    was_truncated is True only when the manual bracket-balancing path had
    to pad in missing closing brackets/braces - meaning the model's
    output was cut off mid-structure and some content (e.g. trailing
    steps) was lost. Callers should surface this to the user instead of
    treating the result as clean.
    """

    cleaned_json = clean_ai_json(raw_text)

    # 1. Straightforward parse.
    try:
        return json.loads(cleaned_json), False
    except Exception:
        pass

    # 2. `json_repair` library, if installed.
    repaired = _try_json_repair_library(cleaned_json)
    if repaired is not None:
        return repaired, False

    # 3. Manual bracket balancing, no dependency required.
    repaired, was_truncated = _try_manual_bracket_balance(cleaned_json)
    if repaired is not None:
        return repaired, was_truncated

    # 4. Last resort: ask the LLM to fix it (slow - only happens when
    #    everything above failed).
    if not allow_llm_repair or llm is None:
        # Re-raise the original parse error so callers see a real
        # json.JSONDecodeError rather than silently failing.
        return json.loads(cleaned_json), False

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

    return json.loads(repaired_cleaned), False


def convert_to_string(value):
    """
    Flattens a value into plain text. Despite the prompt rules telling
    the model never to put a JSON object into a plain-text field, small
    models occasionally do it anyway - in two different shapes seen in
    practice:
      1. An actual nested dict/list in the parsed JSON (e.g. test_data
         comes back as a real Python dict from json.loads).
      2. A STRING that merely looks like JSON (e.g. test_data is the
         literal text '{"key_reference": "invalid_value"}') - this is
         already technically a str, so it silently passed through the
         old version of this function unflattened, even though it
         renders just as badly in the exported Word/Excel doc.

    Both cases get flattened into a readable "key is value" sentence.
    """

    if isinstance(value, str):

        stripped = value.strip()

        looks_like_json = (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        )

        if looks_like_json:
            try:
                parsed = json.loads(stripped)
            except Exception:
                return value
            else:
                return convert_to_string(parsed)

        return value

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if value is None:
        return ""

    if isinstance(value, dict):
        parts = [
            f"{str(key).replace('_', ' ')} is {convert_to_string(val).rstrip('.')}"
            for key, val in value.items()
        ]
        return ", ".join(parts) + "." if parts else ""

    if isinstance(value, list):
        return ", ".join(convert_to_string(item) for item in value)

    return str(value)


def normalize_test_cases(
    test_case_data,
    default_requirement_id="REQ001"
):

    if "test_cases" not in test_case_data:
        test_case_data["test_cases"] = []

    for test_case in test_case_data["test_cases"]:

        test_case.setdefault("test_case_id", "")
        test_case.setdefault("requirement_id", default_requirement_id)
        test_case.setdefault("coverage_area", "Unspecified")
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

        # The prompt tells the model to never leave this blank, but small
        # models sometimes skip the field anyway (seen with NEXDOC's
        # TC-A01/TC-A02 coming back with an empty source_traceability).
        # Enforce the fallback in code rather than trusting the model.
        if not str(test_case.get("source_traceability", "")).strip():
            test_case["source_traceability"] = "Source: requirement"

        test_case["preconditions"] = convert_to_string(
            test_case.get("preconditions")
        )

        test_case["test_data"] = convert_to_string(
            test_case.get("test_data")
        )

        test_case["expected_result"] = convert_to_string(
            test_case.get("expected_result")
        )

        # The model sometimes leaves test_data genuinely blank (seen in
        # a real qwen2.5:3b run: TC-B01, TC-C01, TC-D01, etc. all had an
        # empty Test Data column in the exported doc). An empty cell in
        # an exported Word/Excel test case reads as broken/incomplete to
        # a QA reviewer, so fall back to something explicit instead of
        # silently leaving it blank - same pattern as the
        # source_traceability fallback above.
        if not test_case["test_data"].strip():
            test_case["test_data"] = (
                "Not explicitly specified by the model - refer to "
                "preconditions and steps for the data implied by this scenario."
            )

        if not test_case["preconditions"].strip():
            test_case["preconditions"] = "Not explicitly specified by the model."

        if not test_case["expected_result"].strip():
            test_case["expected_result"] = "Not explicitly specified by the model."

        if isinstance(test_case.get("steps"), str):
            test_case["steps"] = [test_case["steps"]]

        if not isinstance(test_case.get("steps"), list) or not test_case.get("steps"):
            test_case["steps"] = [
                "Not explicitly specified by the model - "
                "refer to preconditions and test data for the implied scenario."
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

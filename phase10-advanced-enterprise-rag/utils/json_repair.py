import json
import re


def clean_ai_json(raw_text):

    cleaned = raw_text.strip()

    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")

    # Remove comments
    cleaned = re.sub(r"//.*", "", cleaned)

    # Fix common LLM mistake: double braces
    cleaned = cleaned.replace("{{", "{")
    cleaned = cleaned.replace("}}", "}")

    # Extract JSON only
    start_index = cleaned.find("{")
    end_index = cleaned.rfind("}")

    if start_index != -1 and end_index != -1:
        cleaned = cleaned[start_index:end_index + 1]

    return cleaned


def parse_json_safely(raw_text, llm):

    cleaned_json = clean_ai_json(raw_text)

    try:
        return json.loads(cleaned_json)

    except Exception:

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

        repaired = llm.invoke(repair_prompt)

        repaired_cleaned = clean_ai_json(repaired)

        return json.loads(repaired_cleaned)


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

        if isinstance(test_case.get("preconditions"), dict):
            test_case["preconditions"] = json.dumps(
                test_case["preconditions"],
                indent=2
            )

        if isinstance(test_case.get("test_data"), dict):
            test_case["test_data"] = json.dumps(
                test_case["test_data"],
                indent=2
            )

        if isinstance(test_case.get("steps"), str):
            test_case["steps"] = [test_case["steps"]]

    return test_case_data
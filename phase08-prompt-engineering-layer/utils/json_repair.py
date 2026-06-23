import json
import re


def clean_ai_json(raw_text):
    cleaned = raw_text.strip()

    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")

    # Remove JavaScript-style comments
    cleaned = re.sub(r"//.*", "", cleaned)

    # Extract only JSON object
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
You are a JSON repair agent.

Convert the content below into VALID JSON ONLY.

Rules:
- Return JSON only
- No markdown
- No explanation
- No comments
- No trailing commas
- Must start with {{
- Must end with }}
- Root object must contain "test_cases"

Content:
{cleaned_json}
"""

        repaired = llm.invoke(repair_prompt)

        repaired_cleaned = clean_ai_json(repaired)

        return json.loads(repaired_cleaned)


def normalize_test_cases(test_case_data, default_requirement_id="REQ001"):

    if "test_cases" not in test_case_data:
        test_case_data["test_cases"] = []

    for test_case in test_case_data["test_cases"]:

        if "requirement_id" not in test_case or not test_case["requirement_id"]:
            test_case["requirement_id"] = default_requirement_id

        if "test_data" not in test_case:
            test_case["test_data"] = ""

        if isinstance(test_case.get("test_data"), dict):
            test_case["test_data"] = json.dumps(
                test_case["test_data"],
                indent=2
            )

        if "steps" not in test_case:
            test_case["steps"] = []

        if isinstance(test_case.get("steps"), str):
            test_case["steps"] = [test_case["steps"]]

    return test_case_data
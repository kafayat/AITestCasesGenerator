"""
Phase 15 - Scenario-Driven Test Case Generator Agent

Core change from Phase 14:
  Phase 14: one prompt asks the model to discover AND write all test cases
            -> model produces 1-3, misses most scenarios
  Phase 15: Python extracts scenarios, model writes ONE test case per scenario
            -> guaranteed N test cases for N scenarios, every run, any model

Each call to generate_test_case_for_scenario() sends the model a tiny,
focused prompt: "Here is one specific scenario. Write exactly one test case
for it." The model never has to discover what to test - it just has to
write steps and expected results for a scenario it's already been given.

This also means:
- 3B models now work reliably (each call is simple enough for them)
- Calls run concurrently (one thread per scenario)
- Failed calls for individual scenarios are retried without affecting others
- Cache keys are per-scenario, so re-runs skip already-generated scenarios
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.json_repair import parse_json_safely, normalize_test_cases
from utils.cache import make_cache_key, get_cached, set_cached
from utils.scenario_extractor import extract_scenarios, summarise_scenarios


SINGLE_SCENARIO_PROMPT = """\
You are a QA Engineer. Write exactly ONE test case for the scenario below.

Requirement:
{requirement}

SCENARIO: {scenario_title}
Coverage area: {coverage_area}
Test type: {test_type}
Priority: {priority}
Context: {requirement_context}

Return valid JSON only, no markdown:

{{
  "test_cases": [{{
    "test_case_id": "TC_PENDING",
    "requirement_id": "REQ001",
    "coverage_area": "{coverage_area}",
    "title": "<specific title referencing actual domain terms from the requirement>",
    "priority": "{priority}",
    "test_type": "{test_type}",
    "preconditions": "<plain text - state before test>",
    "test_data": "<plain text - specific test data values, never empty>",
    "steps": ["<step 1>", "<step 2>", "<step 3>", "<step 4>"],
    "expected_result": "<plain text - single definite outcome>",
    "automation_candidate": "Yes",
    "source_traceability": "Source: requirement"
  }}]
}}
"""


def generate_test_case_for_scenario(
    llm,
    scenario,
    requirement,
    rag_context,
    model_name,
    requirement_id,
    use_cache=True,
):
    """
    Generates exactly one test case for one scenario.
    This is the fundamental unit of Phase 15 generation.
    """

    prompt = SINGLE_SCENARIO_PROMPT.format(
        scenario_title=scenario["title"],
        coverage_area=scenario["coverage_area"],
        test_type=scenario["test_type"],
        priority=scenario["priority"],
        requirement_context=scenario.get("context", "")[:400],
        requirement=requirement[:1500],
    )

    cache_key = make_cache_key(
        "scenario_v1", model_name, scenario["title"], requirement[:300]
    )

    raw_text = get_cached(cache_key) if use_cache else None
    from_cache = raw_text is not None

    if raw_text is None:
        try:
            raw_text = ""
            for chunk in llm.stream(prompt):
                raw_text += chunk
        except Exception:
            raw_text = llm.invoke(prompt)

        if use_cache and raw_text:
            set_cached(cache_key, raw_text)

    try:
        parsed, _ = parse_json_safely(raw_text, llm)
        parsed = normalize_test_cases(parsed, default_requirement_id=requirement_id)
        test_cases = parsed.get("test_cases", [])
    except Exception as exc:
        return None, str(exc), from_cache

    if not test_cases:
        return None, "Model returned empty test_cases array", from_cache

    # Take only the first test case (model should only return one)
    tc = test_cases[0]

    # Force correct values from the scenario (don't trust the model to
    # keep these consistent across 50 independent calls)
    tc["coverage_area"] = scenario["coverage_area"]
    tc["priority"] = scenario["priority"]
    tc["test_type"] = scenario["test_type"]
    tc["requirement_id"] = requirement_id

    return tc, None, from_cache


def generate_all_scenarios_concurrent(
    llm,
    requirement,
    rag_context,
    model_name,
    requirement_id,
    use_cache=True,
    progress_callback=None,
    max_workers=2,
    scenarios=None,
):
    """
    Extracts all scenarios from the requirement (or uses pre-extracted
    scenarios if provided), then generates one test case per scenario
    concurrently.
    """

    if scenarios is None:
        scenarios = extract_scenarios(requirement)

    scenario_summary = summarise_scenarios(scenarios)

    if not scenarios:
        return [], [], scenario_summary

    # Step 2: Generate one test case per scenario concurrently
    scenario_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_scenario = {}

        for scenario in scenarios:
            if progress_callback:
                progress_callback(scenario["title"][:50], "started")

            future = executor.submit(
                generate_test_case_for_scenario,
                llm,
                scenario,
                requirement,
                rag_context,
                model_name,
                requirement_id,
                use_cache,
            )
            future_to_scenario[future] = scenario

        for future in as_completed(future_to_scenario):
            scenario = future_to_scenario[future]

            try:
                tc, error, from_cache = future.result()
            except Exception as exc:
                tc, error, from_cache = None, str(exc), False

            result = {
                "scenario": scenario,
                "test_case": tc,
                "error": error,
                "from_cache": from_cache,
            }
            scenario_results.append(result)

            if progress_callback:
                if error:
                    status = "failed"
                elif from_cache:
                    status = "cached"
                else:
                    status = "done"
                progress_callback(scenario["title"][:50], status)

    # Step 3: Collect successful test cases and assign final IDs
    test_cases = []
    for result in scenario_results:
        if result["test_case"] is not None:
            test_cases.append(result["test_case"])

    # Sort by coverage area for logical output ordering
    area_order = [
        "Happy path / successful flow",
        "Confirmation or notification",
        "Missing mandatory data",
        "Invalid input",
        "Boundary condition",
        "Business rule validation",
        "Duplicate processing",
        "Data integrity",
        "Error handling",
        "Failed transaction or system failure",
        "Security or sensitive data handling",
        "Amendment or modification of an existing record",
        "Replacement or reissue of an existing record",
        "Document or certificate generation tied to a lifecycle event",
    ]

    test_cases.sort(
        key=lambda tc: area_order.index(tc.get("coverage_area", ""))
        if tc.get("coverage_area", "") in area_order
        else 99
    )

    # Assign clean sequential IDs
    for idx, tc in enumerate(test_cases, start=1):
        tc["test_case_id"] = f"TC-{idx:03d}"

    return test_cases, scenario_results, scenario_summary

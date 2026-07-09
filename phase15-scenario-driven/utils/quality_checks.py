"""
Some test case quality problems can't be reliably fixed by asking the
model nicely - we tried that (telling the model "preconditions must be
consistent with steps") and it still produced TC-D03 with preconditions
saying "a valid submission is made" while its own steps said "enter
invalid data". Small local models don't reliably hold a multi-clause
self-consistency instruction across a whole JSON object.

This module catches that class of problem deterministically, the same
way utils/coverage_analysis.py replaced LLM self-reported coverage with
checkable Python logic. These are heuristic keyword checks - cheap,
transparent, and good enough to flag a test case for human review. They
are NOT a guarantee of correctness, and deliberately err on the side of
flagging for a human to glance at, rather than silently dropping or
"fixing" a test case the model already wrote.
"""

import re


VALID_SIGNAL_WORDS = {"valid", "successfully", "successful", "active", "complete", "completed"}
INVALID_SIGNAL_WORDS = {"invalid", "missing", "empty", "expired", "incorrect", "unsupported", "duplicate"}

NEGATIVE_TEST_TYPES = {"Negative", "Boundary", "Error Handling", "Security"}
NEGATIVE_COVERAGE_AREAS = {
    "Invalid input",
    "Missing mandatory data",
    "Boundary condition",
    "Failed transaction or system failure",
    "Error handling",
}


def _tokenize_lower(text):
    return set(re.findall(r"[a-z]+", str(text).lower()))


def _contains_any(tokens, signal_words):
    return bool(tokens & signal_words)


def detect_quality_flags(test_case):
    """
    Returns a list of short, human-readable flag strings for a single
    test case. An empty list means no issues were detected (not the
    same as "guaranteed correct" - just "nothing the cheap checks caught").
    """

    flags = []

    preconditions_tokens = _tokenize_lower(test_case.get("preconditions", ""))
    steps_tokens = _tokenize_lower(" ".join(test_case.get("steps", [])))
    expected_result_tokens = _tokenize_lower(test_case.get("expected_result", ""))
    test_data_tokens = _tokenize_lower(test_case.get("test_data", ""))

    preconditions_text = str(test_case.get("preconditions", "")).lower()

    # "All fields completed EXCEPT exporter reference" is the standard,
    # correct way to write a missing-mandatory-field test case - the
    # word "except" (and similar) signals the precondition is already
    # describing a deliberately partial/incomplete state, not asserting
    # full validity. Without this check, that normal pattern got flagged
    # as a false contradiction in testing (TC-B02).
    precondition_signals_partial_state = any(
        phrase in preconditions_text
        for phrase in ("except", "excluding", "other than", "apart from", "leaving out")
    )

    preconditions_says_valid = (
        _contains_any(preconditions_tokens, VALID_SIGNAL_WORDS)
        and not precondition_signals_partial_state
    )
    steps_say_invalid = _contains_any(steps_tokens, INVALID_SIGNAL_WORDS)

    if preconditions_says_valid and steps_say_invalid:
        flags.append(
            "Preconditions describe a valid/successful state, but the steps "
            "enter invalid/missing data - these contradict each other."
        )

    # Hedged expected_result: a test case that doesn't commit to a single
    # outcome (saw this exact pattern in TC-B01: "validates the country
    # and displays an error message IF it does not match").
    expected_result_text = str(test_case.get("expected_result", "")).lower()

    if re.search(r"\bif (it|the) (does not|doesn't|isn't|is not)\b", expected_result_text):
        flags.append(
            "Expected result is conditional/hedged (\"...if it does not match\") "
            "instead of committing to one definite outcome."
        )

    # A negative/boundary/security test case whose test_data still reads
    # as valid data (e.g. testing Australia, a real supported country,
    # under an "invalid destination country" scenario).
    coverage_area = test_case.get("coverage_area", "")
    test_type = test_case.get("test_type", "")

    is_negative_context = (
        coverage_area in NEGATIVE_COVERAGE_AREAS or test_type in NEGATIVE_TEST_TYPES
    )

    test_data_says_invalid = _contains_any(test_data_tokens, INVALID_SIGNAL_WORDS)

    if is_negative_context and not test_data_says_invalid and test_type == "Positive":
        flags.append(
            f"Coverage area '{coverage_area}' suggests a negative/boundary "
            f"scenario, but test_type is 'Positive' and test_data does not "
            f"clearly describe an invalid/missing/boundary value."
        )

    # Steps that are suspiciously short for the apparent scope of the
    # title (a thin, possibly-truncated test case) - 1 step with no
    # submission/confirmation step at all.
    steps = test_case.get("steps", [])

    if len(steps) <= 1 and len(_tokenize_lower(test_case.get("title", ""))) > 4:
        flags.append(
            "Only 1 step for a test case with a substantial title - may be "
            "underspecified or cut short."
        )

    return flags


def annotate_quality_flags(test_cases):
    """
    Runs detect_quality_flags over every test case and attaches the
    result as test_case['quality_flags']. Returns the count of test
    cases that had at least one flag, for a quick summary count in the UI.
    """

    flagged_count = 0

    for test_case in test_cases:

        flags = detect_quality_flags(test_case)
        test_case["quality_flags"] = flags

        if flags:
            flagged_count += 1

    return flagged_count

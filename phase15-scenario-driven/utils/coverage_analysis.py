"""
Phase 13: Coverage & Gap Analysis.

Phase 11/12's Quality Scoring Agent asked the LLM to self-report coverage
("Coverage Score: 95%", "Missing Scenarios: None") - which we saw in
practice is unreliable: the model said "Business Rule Coverage: Not
applicable" in a run that had 3 business-rule test cases, and "Missing
Scenarios: None" when a real acceptance criterion (confirmation email)
had no matching test case.

This module replaces LLM self-reporting with DETERMINISTIC Python
checks computed directly from the structured fields Phase 12 already
produces (coverage_area, test_data, preconditions, expected_result,
source_traceability). The LLM is only still used for things that
genuinely need judgment (the Quality Scoring Agent's free-text
recommendations) - everything checkable in code is checked in code.

No new dependencies, no embeddings - matching is done with simple
keyword/word-overlap scoring, which is transparent, fast, and easy to
explain to a non-technical stakeholder ("this test case matched because
it shares these 4 words with the requirement line").
"""

import os
import re


ALL_COVERAGE_AREAS = [
    "Happy path / successful flow",
    "Invalid input",
    "Missing mandatory data",
    "Business rule validation",
    "Boundary condition",
    "Duplicate processing",
    "Failed transaction or system failure",
    "Confirmation or notification",
    "Receipt or reference generation",
    "Security or sensitive data handling",
    "Data integrity",
    "Error handling",
    # Added after the NEXDOC run exposed a real structural gap: domains
    # with lifecycle/workflow operations (amend, replace, reissue a
    # certificate) had ZERO test cases for them, because none of the
    # original 12 areas covers a record being modified after creation.
    # Phrased generically (not "amend a REX lodgement") so this applies
    # to any domain with an edit/replace/reissue-style workflow, not
    # just NEXDOC.
    "Amendment or modification of an existing record",
    "Replacement or reissue of an existing record",
    "Document or certificate generation tied to a lifecycle event",
]

# Coverage areas considered higher-risk by nature - used for the risk
# coverage summary. This is a simple, transparent rule rather than an
# LLM judgment call, and can be edited by a QA lead without touching code.
HIGH_RISK_AREAS = {
    "Security or sensitive data handling",
    "Failed transaction or system failure",
    "Business rule validation",
    "Data integrity",
    "Replacement or reissue of an existing record",
}

MEDIUM_RISK_AREAS = {
    "Duplicate processing",
    "Error handling",
    "Boundary condition",
    "Amendment or modification of an existing record",
    "Document or certificate generation tied to a lifecycle event",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "for", "on",
    "in", "with", "must", "should", "can", "will", "be", "as", "that",
    "this", "their", "if", "it", "by", "before", "after", "from", "at",
    "customer", "system", "valid", "page",
}


def _tokenize(text):

    words = re.findall(r"[a-z0-9]+", str(text).lower())

    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


def _test_case_text_blob(test_case):
    """
    Concatenates every text field of a test case into one blob for
    keyword matching against requirement lines / business rules.
    """

    parts = [
        test_case.get("title", ""),
        test_case.get("preconditions", ""),
        test_case.get("test_data", ""),
        " ".join(test_case.get("steps", [])),
        test_case.get("expected_result", ""),
        test_case.get("coverage_area", ""),
    ]

    return " ".join(parts)


def _match_score(line_tokens, test_case_tokens, token_weights):
    """
    Weighted overlap score instead of plain word-overlap. Plain overlap
    badly over-matches in this domain: words like "payment", "account",
    "bill" appear in almost every test case, so a naive ratio says
    nearly every acceptance criterion is "covered" by all 8 test cases -
    technically true but useless for spotting real gaps.

    token_weights down-weights words that appear in many test cases
    (an inverse-document-frequency style weight) so a match only counts
    strongly when it shares a relatively distinctive word (e.g.
    "confirmation", "duplicate", "boundary") rather than a ubiquitous one.
    """

    if not line_tokens or not test_case_tokens:
        return 0.0

    overlap = line_tokens & test_case_tokens

    if not overlap:
        return 0.0

    overlap_weight = sum(token_weights.get(token, 1.0) for token in overlap)
    line_weight = sum(token_weights.get(token, 1.0) for token in line_tokens)

    if line_weight == 0:
        return 0.0

    return overlap_weight / line_weight


def _compute_token_weights(test_case_tokens_list, total_test_cases):
    """
    Simple inverse-document-frequency weight per token: a token that
    appears in every test case gets weight close to 0 (uninformative),
    a token that appears in only one test case gets weight close to 1
    (highly distinctive, a strong signal for matching).
    """

    document_frequency = {}

    for _, tokens in test_case_tokens_list:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    weights = {}

    for token, df in document_frequency.items():
        weights[token] = 1.0 - (df / total_test_cases) if total_test_cases else 1.0
        # Never let a token's weight hit exactly 0, so a rare-but-not-
        # unique word still contributes a little.
        weights[token] = max(weights[token], 0.1)

    return weights


def compute_coverage_matrix(test_cases):
    """
    Returns a list of dicts, one per known coverage area, with the
    count of test cases tagged with that area and the list of their
    test_case_ids. Areas with zero matches are real gaps - flagged
    explicitly rather than silently absent from a table.
    """

    matrix = {area: [] for area in ALL_COVERAGE_AREAS}
    unrecognized = []

    for test_case in test_cases:

        area = test_case.get("coverage_area", "Unspecified")

        if area in matrix:
            matrix[area].append(test_case.get("test_case_id", ""))
        else:
            unrecognized.append(test_case.get("test_case_id", ""))

    rows = []

    for area in ALL_COVERAGE_AREAS:

        test_case_ids = matrix[area]

        rows.append({
            "coverage_area": area,
            "count": len(test_case_ids),
            "test_case_ids": test_case_ids,
            "status": "Covered" if test_case_ids else "MISSING",
        })

    return rows, unrecognized


def extract_acceptance_criteria(requirement_text):
    """
    Pulls bullet-point lines out of a requirement document. Looks for a
    line starting with "-" or "*" anywhere in the text - this matches
    the format of the Sydney Water sample requirement (and most
    user-story-style requirement docs with an "Acceptance Criteria:"
    section).
    """

    criteria = []

    for line in requirement_text.splitlines():

        stripped = line.strip()

        if stripped.startswith("-") or stripped.startswith("*"):
            cleaned = stripped.lstrip("-*").strip()
            if cleaned:
                criteria.append(cleaned)

    return criteria


def parse_business_rules(knowledge_folder="knowledge_base/business_rules"):
    """
    Pulls bullet-point rule lines out of every .txt file in the
    business_rules knowledge category. Returns a list of
    (rule_text, source_file) tuples.
    """

    rules = []

    if not os.path.exists(knowledge_folder):
        return rules

    for file_name in os.listdir(knowledge_folder):

        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(knowledge_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        for line in content.splitlines():

            stripped = line.strip()

            if stripped.startswith("-") or stripped.startswith("*"):
                cleaned = stripped.lstrip("-*").strip()
                if cleaned:
                    rules.append((cleaned, file_name))

    return rules


def match_lines_to_test_cases(lines, test_cases, threshold=0.30):
    """
    For each input line (an acceptance criterion or a business rule),
    finds test cases whose combined text shares enough DISTINCTIVE
    keywords with the line to count as a match (see _compute_token_weights
    for why this isn't plain word-overlap). `lines` can be a list of
    strings or a list of (text, source) tuples.

    Returns a list of dicts: {text, source, matched_test_case_ids, covered}
    """

    test_case_tokens_list = [
        (test_case.get("test_case_id", ""), _tokenize(_test_case_text_blob(test_case)))
        for test_case in test_cases
    ]

    token_weights = _compute_token_weights(test_case_tokens_list, len(test_cases))

    results = []

    for line in lines:

        if isinstance(line, tuple):
            text, source = line
        else:
            text, source = line, None

        line_tokens = _tokenize(text)

        matched_ids = [
            test_case_id
            for test_case_id, tc_tokens in test_case_tokens_list
            if _match_score(line_tokens, tc_tokens, token_weights) >= threshold
        ]

        results.append({
            "text": text,
            "source": source,
            "matched_test_case_ids": matched_ids,
            "covered": len(matched_ids) > 0,
        })

    return results


def compute_risk_coverage(test_cases):
    """
    Risk-weighted coverage summary: counts test cases by risk tier based
    on their coverage_area, rather than treating every test case as
    equally important. Deterministic, editable by a QA lead via
    HIGH_RISK_AREAS / MEDIUM_RISK_AREAS above - no LLM judgment call.
    """

    tiers = {"High": [], "Medium": [], "Low": []}

    for test_case in test_cases:

        area = test_case.get("coverage_area", "")
        test_case_id = test_case.get("test_case_id", "")

        if area in HIGH_RISK_AREAS:
            tiers["High"].append(test_case_id)
        elif area in MEDIUM_RISK_AREAS:
            tiers["Medium"].append(test_case_id)
        else:
            tiers["Low"].append(test_case_id)

    return tiers


def build_gap_analysis_report(
    requirement_text,
    test_cases,
    knowledge_folder="knowledge_base",
):
    """
    Single entry point - builds the full deterministic coverage & gap
    analysis report from a requirement and the final generated test
    cases. Returns a dict ready to render in Streamlit or export.
    """

    coverage_matrix, unrecognized_areas = compute_coverage_matrix(test_cases)

    missing_coverage_areas = [
        row["coverage_area"] for row in coverage_matrix if row["status"] == "MISSING"
    ]

    acceptance_criteria = extract_acceptance_criteria(requirement_text)
    criteria_traceability = match_lines_to_test_cases(acceptance_criteria, test_cases)

    business_rules = parse_business_rules(
        os.path.join(knowledge_folder, "business_rules")
    )
    business_rule_traceability = match_lines_to_test_cases(business_rules, test_cases)

    risk_coverage = compute_risk_coverage(test_cases)

    uncovered_criteria = [
        row for row in criteria_traceability if not row["covered"]
    ]

    uncovered_rules = [
        row for row in business_rule_traceability if not row["covered"]
    ]

    return {
        "coverage_matrix": coverage_matrix,
        "missing_coverage_areas": missing_coverage_areas,
        "unrecognized_areas": unrecognized_areas,
        "criteria_traceability": criteria_traceability,
        "uncovered_criteria": uncovered_criteria,
        "business_rule_traceability": business_rule_traceability,
        "uncovered_rules": uncovered_rules,
        "risk_coverage": risk_coverage,
    }


def format_facts_for_quality_prompt(report):
    """
    Renders the gap analysis report as a short, plain-text "ground
    truth" summary to inject into the Quality Scoring Agent's prompt.

    This exists because the LLM-based Quality Scoring report kept
    self-contradicting reality in real runs - e.g. saying "Business Rule
    Coverage: Not applicable" on a run that had 3 business-rule test
    cases, because the model was GUESSING at coverage instead of
    actually counting. Feeding it the already-computed facts (and
    instructing it not to contradict them) turns the unreliable part of
    the quality report from "estimate everything" into "summarize 3
    pre-computed facts, exercise judgment only on the rest."
    """

    lines = []

    coverage_area_count = len(report["coverage_matrix"])
    hit_count = coverage_area_count - len(report["missing_coverage_areas"])

    lines.append(f"- {hit_count} of {coverage_area_count} coverage areas have at least one test case.")

    if report["missing_coverage_areas"]:
        lines.append(
            "- These coverage areas have ZERO test cases: "
            + ", ".join(report["missing_coverage_areas"])
        )
    else:
        lines.append("- No coverage area is missing test cases.")

    business_rule_test_case_count = sum(
        1 for row in report["coverage_matrix"]
        if row["coverage_area"] == "Business rule validation"
        for _ in row["test_case_ids"]
    )

    if business_rule_test_case_count:
        lines.append(
            f"- {business_rule_test_case_count} test case(s) exist under "
            f"'Business rule validation' - do NOT say business rule "
            f"coverage is not applicable."
        )
    else:
        lines.append("- No test cases exist under 'Business rule validation'.")

    if report["uncovered_rules"]:
        lines.append(
            f"- {len(report['uncovered_rules'])} business rule(s) from the "
            f"knowledge base have no matching test case."
        )

    return "\n".join(lines)


def format_report_as_markdown(report, requirement_id="REQ001"):
    """
    Renders the gap analysis report dict as a Markdown document, for
    download / inclusion alongside the Excel and Word exports.
    """

    lines = [f"# Coverage & Gap Analysis - {requirement_id}", ""]

    lines.append("## Coverage Matrix")
    lines.append("")
    lines.append("| Coverage Area | Test Cases | Status |")
    lines.append("|---|---|---|")

    for row in report["coverage_matrix"]:
        ids = ", ".join(row["test_case_ids"]) if row["test_case_ids"] else "-"
        lines.append(f"| {row['coverage_area']} | {ids} | {row['status']} |")

    lines.append("")
    lines.append("## Acceptance Criteria Traceability")
    lines.append("")
    lines.append("| Acceptance Criterion | Matched Test Cases | Status |")
    lines.append("|---|---|---|")

    for row in report["criteria_traceability"]:
        ids = ", ".join(row["matched_test_case_ids"]) if row["matched_test_case_ids"] else "-"
        status = "Covered" if row["covered"] else "MISSING"
        lines.append(f"| {row['text']} | {ids} | {status} |")

    lines.append("")
    lines.append("## Business Rule Coverage")
    lines.append("")
    lines.append("| Business Rule | Source | Matched Test Cases | Status |")
    lines.append("|---|---|---|---|")

    for row in report["business_rule_traceability"]:
        ids = ", ".join(row["matched_test_case_ids"]) if row["matched_test_case_ids"] else "-"
        status = "Covered" if row["covered"] else "MISSING"
        lines.append(f"| {row['text']} | {row['source'] or '-'} | {ids} | {status} |")

    lines.append("")
    lines.append("## Risk Coverage")
    lines.append("")
    lines.append("| Risk Tier | Test Case Count | Test Case IDs |")
    lines.append("|---|---|---|")

    for tier, ids in report["risk_coverage"].items():
        ids_text = ", ".join(ids) if ids else "-"
        lines.append(f"| {tier} | {len(ids)} | {ids_text} |")

    return "\n".join(lines)

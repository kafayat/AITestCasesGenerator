"""
Phase 15 - Scenario Extraction Engine

The fundamental shift from Phase 14:

Phase 14 asked the model to do TWO jobs in one call:
  1. Figure out what scenarios to test (unreliable - model kept producing 1-3)
  2. Write the test case for each scenario (actually OK when given a clear scenario)

Phase 15 separates these:
  Job 1 -> Python code extracts scenarios deterministically from the requirement
  Job 2 -> Model writes ONE test case per scenario (tiny, fast, reliable call)

Result: number of test cases = number of scenarios extracted, every run,
with any model including 3B. Quality depends on the model; quantity is
now guaranteed by the extractor.

Scenario types extracted:
  - Happy path per action (Create/Update/Delete/Submit etc.)
  - Missing mandatory field per field
  - Conditional display rule per rule
  - Acceptance criteria bullet points
  - Security/access rules
  - Workflow/SLA rules
  - Error handling (system failure, invalid input)
"""

import re


# ---------------------------------------------------------------------------
# Coverage area mapping for each scenario type
# ---------------------------------------------------------------------------

SCENARIO_COVERAGE_MAP = {
    "happy_path":           "Happy path / successful flow",
    "confirmation":         "Confirmation or notification",
    "missing_field":        "Missing mandatory data",
    "invalid_input":        "Invalid input",
    "conditional_rule":     "Business rule validation",
    "duplicate":            "Duplicate processing",
    "data_integrity":       "Data integrity",
    "system_failure":       "Failed transaction or system failure",
    "security":             "Security or sensitive data handling",
    "error_handling":       "Error handling",
    "amendment":            "Amendment or modification of an existing record",
    "lifecycle":            "Document or certificate generation tied to a lifecycle event",
    "acceptance_criteria":  "Business rule validation",
}

SCENARIO_PRIORITY_MAP = {
    "happy_path":           "High",
    "confirmation":         "Medium",
    "missing_field":        "High",
    "invalid_input":        "High",
    "conditional_rule":     "High",
    "duplicate":            "Medium",
    "data_integrity":       "High",
    "system_failure":       "High",
    "security":             "High",
    "error_handling":       "High",
    "amendment":            "High",
    "lifecycle":            "Medium",
    "acceptance_criteria":  "Medium",
}

SCENARIO_TEST_TYPE_MAP = {
    "happy_path":           "Positive",
    "confirmation":         "Positive",
    "missing_field":        "Negative",
    "invalid_input":        "Negative",
    "conditional_rule":     "Business Rule",
    "duplicate":            "Business Rule",
    "data_integrity":       "Business Rule",
    "system_failure":       "Error Handling",
    "security":             "Security",
    "error_handling":       "Error Handling",
    "amendment":            "Positive",
    "lifecycle":            "Positive",
    "acceptance_criteria":  "Business Rule",
}


# ---------------------------------------------------------------------------
# Keyword lists for heuristic detection
# ---------------------------------------------------------------------------

ACTION_KEYWORDS = [
    "create", "add", "new", "insert",
    "update", "edit", "modify", "change", "amend",
    "delete", "remove", "cancel",
    "submit", "lodge", "request", "send",
    "search", "find", "view", "display",
    "approve", "reject", "review",
    "upload", "download", "export", "import",
]

MANDATORY_KEYWORDS = [
    "mandatory", "required", "must", "compulsory", "obligatory",
    "cannot be empty", "cannot be blank", "not optional",
]

CONDITIONAL_KEYWORDS = [
    "only when", "only if", "display when", "shown when", "shown if",
    "visible when", "visible if", "appears when", "appears if",
    "hide when", "hidden when", "conditional",
]

SECURITY_KEYWORDS = [
    "access", "authoris", "authenticat", "permission", "role",
    "user criteria", "available for", "not available for",
    "login", "log in", "unauthoris", "restrict",
]

SYSTEM_FAILURE_KEYWORDS = [
    "system failure", "service unavailable", "timeout", "error",
    "exception", "down", "unavailable", "outage",
]

DUPLICATE_KEYWORDS = [
    "duplicate", "already exists", "unique", "prevent duplicate",
    "same reference", "existing record",
]

SLA_KEYWORDS = [
    "sla", "service level", "business day", "working day",
    "within", "days", "hours", "turnaround",
]


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _clean(text):
    return str(text).strip() if text else ""


def _lower(text):
    return _clean(text).lower()


def _sentences(text):
    """Split text into sentences/lines for scanning."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", _lower(text)))


def _contains_any(text, keywords):
    low = _lower(text)
    return any(kw in low for kw in keywords)


# ---------------------------------------------------------------------------
# Scenario dataclass (plain dict for simplicity)
# ---------------------------------------------------------------------------

def _scenario(scenario_type, title, context="", field_name="", action=""):
    return {
        "scenario_type":    scenario_type,
        "title":            title,
        "context":          context,
        "field_name":       field_name,
        "action":           action,
        "coverage_area":    SCENARIO_COVERAGE_MAP.get(scenario_type, "Business rule validation"),
        "priority":         SCENARIO_PRIORITY_MAP.get(scenario_type, "Medium"),
        "test_type":        SCENARIO_TEST_TYPE_MAP.get(scenario_type, "Positive"),
    }


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_actions(requirement_text):
    """
    Finds distinct user actions described in the requirement.
    Handles the Excel-extracted format where each action is on its own line:
      Option: Create
      Option: Update
      Option: Delete
    """

    found = []

    # Primary: "Option: X" lines from Excel form field extraction
    option_matches = re.findall(
        r"^\s*[Oo]ption[:]?\s+([A-Z][A-Za-z\s]{1,20})$",
        requirement_text,
        re.MULTILINE,
    )
    for match in option_matches:
        action = _clean(match)
        if (
            action
            and 2 < len(action) < 25
            and action not in found
            and _contains_any(action.lower(), ACTION_KEYWORDS)
        ):
            found.append(action)

    # Secondary: inline option list e.g. "Options: Create, Update, Delete"
    if not found:
        inline_matches = re.findall(
            r"options?\s*[:\-]\s*([A-Za-z][A-Za-z\s,/]+?)(?:\n|$)",
            requirement_text,
            re.IGNORECASE,
        )
        for match in inline_matches:
            parts = re.split(r"[,/]", match)
            for part in parts:
                action = _clean(part)
                if (
                    action
                    and 2 < len(action) < 25
                    and action not in found
                    and _contains_any(action.lower(), ACTION_KEYWORDS)
                ):
                    found.append(action)

    # Tertiary: keyword scan
    if not found:
        low = _lower(requirement_text)
        for keyword in ACTION_KEYWORDS:
            if re.search(rf"\b{keyword}\b", low):
                capitalised = keyword.capitalize()
                if capitalised not in found:
                    found.append(capitalised)

    return found[:10]


def extract_mandatory_fields(requirement_text):
    """
    Finds mandatory field names from the requirement.
    Works for:
      - Excel-extracted text: "- [Single line text] Name | Mandatory: Yes"
      - Bullet lists: "- Name (mandatory)"
      - Prose: "Name is a required field"
    """

    fields = []

    # INFERRED ACCEPTANCE CRITERIA section is not real field definitions —
    # skip it to avoid false mandatory field matches from sentences like
    # "User must be able to submit a Create request successfully."
    # Split on that section and only process the part before it.
    split_marker = "INFERRED ACCEPTANCE CRITERIA"
    if split_marker in requirement_text:
        form_text = requirement_text[:requirement_text.index(split_marker)]
    else:
        form_text = requirement_text

    lines = _sentences(form_text)

    for line in lines:

        # Excel-style form field line with Mandatory: Yes
        if "mandatory: yes" in _lower(line):
            # Extract field label from between ] and |
            match = re.search(r"\]\s*([^|]+?)\s*\|", line)
            if match:
                field = _clean(match.group(1))
                # Filter out noise: field names should be short and not
                # contain sentence-like phrases
                if field and 2 < len(field) < 60 and field not in fields:
                    fields.append(field)
            continue

        # Bullet/prose: "Field name (mandatory)" or "Field name is required"
        if _contains_any(line, MANDATORY_KEYWORDS):
            clean_line = re.sub(r"^[-*•]\s*", "", line).strip()
            # Skip very long lines (prose paragraphs) and lines with
            # sentence-like content (starts with "User", "System", etc.)
            if len(clean_line) < 60 and not re.match(
                r"^(user|system|the system|customer|submitter)", clean_line, re.IGNORECASE
            ):
                field = re.sub(
                    r"\s*[\(\[]?(mandatory|required|must|compulsory)[\)\]]?\s*",
                    "",
                    clean_line,
                    flags=re.IGNORECASE,
                ).strip().rstrip(".:,")
                if field and len(field) > 2 and field not in fields:
                    fields.append(field)

    return fields


def extract_conditional_rules(requirement_text):
    """
    Finds conditional display / show-hide rules.
    Returns list of (rule_description, trigger, field_name) tuples.
    """

    rules = []
    lines = _sentences(requirement_text)

    for line in lines:
        if _contains_any(line, CONDITIONAL_KEYWORDS):
            # Try to extract field and trigger from structured form-field line
            # e.g. "- [Single line text] Phone number | Mandatory: Yes | Display when: contact = Phone"
            display_match = re.search(
                r"display when[:\s]+(.+?)$", line, re.IGNORECASE
            )
            field_match = re.search(r"\]\s*([^|]+?)\s*\|", line)

            field_name = _clean(field_match.group(1)) if field_match else ""
            condition = _clean(display_match.group(1)) if display_match else _clean(line)

            if len(condition) > 5 and len(condition) < 300:
                # Be specific about what the rule means — "shown/hidden" is
                # ambiguous and causes the model to guess wrong direction.
                # If the condition mentions "default" it is an auto-population
                # rule, not a hide rule.
                if field_name:
                    if any(kw in condition.lower() for kw in ["default", "auto", "populated", "logged in"]):
                        desc = f"{field_name} is auto-populated based on: {condition}"
                    else:
                        desc = f"{field_name} is conditionally shown or hidden based on: {condition}"
                else:
                    desc = condition
                rules.append((desc, condition, field_name))

    return rules


def extract_acceptance_criteria(requirement_text):
    """
    Extracts bullet-point acceptance criteria.
    Returns a list of criterion strings.
    """

    criteria = []
    lines = _sentences(requirement_text)
    in_criteria_section = False

    for line in lines:
        low = _lower(line)

        if any(kw in low for kw in ["acceptance criteria", "inferred acceptance", "must be able", "coverage areas to test"]):
            in_criteria_section = True
            continue

        if in_criteria_section:
            if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                criterion = line.lstrip("-*•").strip()
                if criterion and len(criterion) > 10:
                    criteria.append(criterion)
            elif line and not line.startswith(" ") and len(line) > 30:
                # New section - stop collecting
                in_criteria_section = False

    return criteria


def extract_security_rules(requirement_text):
    """Finds access control / security constraints."""

    rules = []
    lines = _sentences(requirement_text)

    for line in lines:
        if _contains_any(line, SECURITY_KEYWORDS) and len(line) < 200:
            clean = line.lstrip("-*•").strip()
            if clean and clean not in rules:
                rules.append(clean)

    return rules[:5]  # cap to avoid over-generating


def extract_sla_rules(requirement_text):
    """Finds SLA / turnaround time rules."""

    rules = []
    lines = _sentences(requirement_text)

    for line in lines:
        if _contains_any(line, SLA_KEYWORDS) and len(line) < 200:
            clean = line.lstrip("-*•").strip()
            if clean and clean not in rules:
                rules.append(clean)

    return rules[:3]


# ---------------------------------------------------------------------------
# Master scenario builder
# ---------------------------------------------------------------------------

def extract_scenarios(requirement_text):
    """
    Master function: extracts ALL testable scenarios from any requirement
    text and returns a structured list. This list is then passed to the
    model one scenario at a time (Phase 15's core approach).

    Returns: list of scenario dicts, each with:
      - scenario_type, title, context, field_name, action
      - coverage_area, priority, test_type (pre-mapped)
    """

    scenarios = []

    # ── 1. Happy path per action ─────────────────────────────────────────────
    actions = extract_actions(requirement_text)
    for action in actions:
        scenarios.append(_scenario(
            "happy_path",
            f"User successfully submits a {action} request with all valid details",
            context=requirement_text[:1000],
            action=action,
        ))

    # ── 2. Confirmation ──────────────────────────────────────────────────────
    if actions:
        scenarios.append(_scenario(
            "confirmation",
            "User receives confirmation after successful request submission",
            context=requirement_text[:500],
        ))

    # ── 3. Missing mandatory field per field ─────────────────────────────────
    mandatory_fields = extract_mandatory_fields(requirement_text)
    for field in mandatory_fields:
        scenarios.append(_scenario(
            "missing_field",
            f"Request cannot be submitted when {field} is left empty",
            context=requirement_text[:1000],
            field_name=field,
        ))

    # ── 4. Conditional display rules ─────────────────────────────────────────
    conditional_rules = extract_conditional_rules(requirement_text)
    for desc, condition, field_name in conditional_rules:

        # Skip noise — acceptance criteria sentences picked up as rules
        if not field_name:
            continue
        if len(field_name) < 3:
            continue
        if _contains_any(field_name.lower(), ["conditional field", "mandatory field", "display condition"]):
            continue
        # Skip generic Excel column headers picked up as field names
        if _contains_any(field_name.lower(), ["questions and options", "instruction text", "order", "type"]):
            continue

        scenarios.append(_scenario(
            "conditional_rule",
            f"Conditional display rule: {field_name}",
            context=desc,
            field_name=field_name,
        ))

    # ── 5. Acceptance criteria ───────────────────────────────────────────────
    criteria = extract_acceptance_criteria(requirement_text)
    for criterion in criteria:

        # Skip criteria already covered by missing_field or error_handling
        # scenarios — these cause the model to generate duplicate "Name field
        # empty" test cases because it sees "mandatory fields must be validated"
        # and defaults to its most-used template.
        skip_keywords = [
            "mandatory", "must be validated", "must not allow submission",
            "cannot be submitted", "must be able to submit",
            "successfully submits", "missing mandatory",
        ]
        if _contains_any(criterion.lower(), skip_keywords):
            continue

        scenarios.append(_scenario(
            "acceptance_criteria",
            f"Verify: {criterion[:80]}",
            context=criterion,
        ))

    # ── 6. Security ──────────────────────────────────────────────────────────
    security_rules = extract_security_rules(requirement_text)
    if security_rules:
        scenarios.append(_scenario(
            "security",
            "Only authorised users can access and use this form",
            context="\n".join(security_rules),
        ))

    # ── 7. Error handling ────────────────────────────────────────────────────
    scenarios.append(_scenario(
        "error_handling",
        "All mandatory field validation errors are displayed when form is submitted with empty fields",
        context=requirement_text[:500],
    ))

    if _contains_any(requirement_text, SYSTEM_FAILURE_KEYWORDS):
        scenarios.append(_scenario(
            "system_failure",
            "User sees a clear error when the system is unavailable during submission",
            context=requirement_text[:500],
        ))

    # ── 8. Workflow / SLA ────────────────────────────────────────────────────
    # Skip raw table rows (contain | pipe characters) — not real scenarios
    sla_rules = extract_sla_rules(requirement_text)
    for rule in sla_rules:
        if "|" in rule or len(rule) < 10:
            continue
        scenarios.append(_scenario(
            "lifecycle",
            f"Workflow rule verified: {rule[:80]}",
            context=rule,
        ))

    # ── 9. Amendment (if update/edit action exists) ──────────────────────────
    update_actions = [a for a in actions if _contains_any(a, ["update", "edit", "amend", "modify", "change"])]
    if update_actions:
        scenarios.append(_scenario(
            "amendment",
            f"Existing record is successfully modified using the {update_actions[0]} action",
            context=requirement_text[:500],
            action=update_actions[0],
        ))

    # ── 10. Delete happy path ────────────────────────────────────────────────
    delete_actions = [a for a in actions if _contains_any(a, ["delete", "remove", "cancel"])]
    if delete_actions:
        scenarios.append(_scenario(
            "happy_path",
            f"User successfully submits a {delete_actions[0]} request with confirmation checkbox ticked",
            context=requirement_text[:1000],
            action=delete_actions[0],
        ))

    # Remove duplicates — use exact field_name matching for missing_field
    # scenarios (so "Email empty" and "Name empty" are always distinct),
    # and title similarity at 0.75 for all other scenario types.
    seen_titles = []
    seen_field_scenarios = set()
    unique_scenarios = []

    for s in scenarios:
        title_low = _lower(s["title"])

        if s["scenario_type"] == "missing_field" and s.get("field_name"):
            key = f"missing_field::{s['field_name'].lower()}"
            if key in seen_field_scenarios:
                continue
            seen_field_scenarios.add(key)
            unique_scenarios.append(s)
            seen_titles.append(title_low)
        else:
            if not any(_title_overlap(title_low, seen) > 0.75 for seen in seen_titles):
                unique_scenarios.append(s)
                seen_titles.append(title_low)

    return unique_scenarios


def _title_overlap(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def summarise_scenarios(scenarios):
    """Returns a human-readable summary of extracted scenarios."""

    lines = [f"Extracted {len(scenarios)} scenarios:"]
    by_type = {}
    for s in scenarios:
        t = s["scenario_type"]
        by_type.setdefault(t, []).append(s["title"])

    for t, titles in by_type.items():
        lines.append(f"\n  {t.upper()} ({len(titles)}):")
        for title in titles:
            lines.append(f"    - {title}")

    return "\n".join(lines)

def classify_knowledge_file(file_name, content):

    file_name = file_name.lower()
    content = content.lower()

    scores = {
        "requirements": 0,
        "business_rules": 0,
        "sample_test_cases": 0,
        "defects": 0,
        "standards": 0,
        "api_docs": 0
    }

    requirement_keywords = [
        "acceptance criteria",
        "as a ",
        "i want",
        "so that",
        "requirement",
        "user story",
        "feature"
    ]

    for keyword in requirement_keywords:
        if keyword in content:
            scores["requirements"] += 5

    business_rule_keywords = [
        "business rule",
        "billing and payment rules",
        "must",
        "should",
        "only after",
        "prevent duplicate",
        "receipt number",
        "confirmation email",
        "payment must",
        "failed payment"
    ]

    for keyword in business_rule_keywords:
        if keyword in content:
            scores["business_rules"] += 4

    sample_test_case_keywords = [
        "test case id",
        "test case",
        "expected result",
        "preconditions",
        "test data",
        "steps:",
        "priority:",
        "type:"
    ]

    for keyword in sample_test_case_keywords:
        if keyword in content:
            scores["sample_test_cases"] += 6

    defect_keywords = [
        "defect",
        "bug",
        "incident",
        "root cause",
        "failed in production",
        "known issue",
        "error reported"
    ]

    for keyword in defect_keywords:
        if keyword in content:
            scores["defects"] += 6

    standard_keywords = [
        "testing standards",
        "qa standards",
        "standard",
        "guideline",
        "best practice",
        "must include",
        "test cases must include"
    ]

    for keyword in standard_keywords:
        if keyword in content:
            scores["standards"] += 5

    api_keywords = [
        "api",
        "endpoint",
        "request",
        "response",
        "swagger",
        "openapi",
        "status code",
        "payload",
        "json body"
    ]

    for keyword in api_keywords:
        if keyword in content:
            scores["api_docs"] += 6

    if "requirement" in file_name or "story" in file_name:
        scores["requirements"] += 10

    if "rule" in file_name or "business" in file_name:
        scores["business_rules"] += 10

    if "test" in file_name or "case" in file_name:
        scores["sample_test_cases"] += 10

    if "defect" in file_name or "bug" in file_name:
        scores["defects"] += 10

    if "standard" in file_name or "qa" in file_name:
        scores["standards"] += 10

    if "api" in file_name or "swagger" in file_name:
        scores["api_docs"] += 10

    detected_category = max(
        scores,
        key=scores.get
    )

    confidence_score = scores[detected_category]

    if confidence_score == 0:
        detected_category = "standards"
        confidence_score = 1

    return detected_category, confidence_score, scores
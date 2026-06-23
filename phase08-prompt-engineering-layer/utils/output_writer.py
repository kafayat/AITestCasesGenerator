import json
import pandas as pd


def test_cases_to_rows(test_cases):
    rows = []

    for test_case in test_cases:
        rows.append({
            "Test Case ID": test_case.test_case_id,
            "Requirement ID": test_case.requirement_id,
            "Requirement Reference": test_case.requirement_reference,
            "Business Rule Reference": test_case.business_rule_reference,
            "Module": test_case.module,
            "Feature": test_case.feature,
            "Risk Level": test_case.risk_level,
            "Test Objective": test_case.test_objective,
            "Title": test_case.title,
            "Priority": test_case.priority,
            "Type": test_case.test_type,
            "Precondition": test_case.preconditions,
            "Test Data": test_case.test_data,
            "Steps": "\n".join(test_case.steps),
            "Expected Result": test_case.expected_result,
            "Post Condition": test_case.post_condition,
            "Automation Candidate": test_case.automation_candidate,
            "Traceability": test_case.traceability
        })

    return rows


def save_outputs(test_cases, output_folder="outputs"):
    rows = test_cases_to_rows(test_cases)

    df = pd.DataFrame(rows)

    excel_path = f"{output_folder}/test_cases.xlsx"
    csv_path = f"{output_folder}/test_cases.csv"
    markdown_path = f"{output_folder}/test_cases.md"
    html_path = f"{output_folder}/test_cases.html"

    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False)
    df.to_markdown(markdown_path, index=False)

    html_content = df.to_html(index=False)

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    return {
        "excel": excel_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "html": html_path
    }
def generate_test_cases(llm, requirement, analysis, rag_context):
    prompt = f"""
You are an Enterprise QA Test Case Generator.

Use the requirement, analysis, and retrieved enterprise knowledge.

Requirement:
{requirement}

Requirement Analysis:
{analysis}

Retrieved Enterprise Knowledge:
{rag_context}

Generate exactly 4 test cases:
- 1 Positive
- 1 Negative
- 1 Boundary
- 1 Security

Return VALID JSON ONLY.
Do not return markdown.
Do not return explanation.
The response must start with {{ and end with }}.

JSON format:
{{
  "test_cases": [
    {{
      "test_case_id": "TC001",
      "title": "Title here",
      "priority": "High",
      "test_type": "Positive",
      "preconditions": "Precondition here",
      "test_data": "Test data here",
      "steps": [
        "Step 1",
        "Step 2",
        "Step 3"
      ],
      "expected_result": "Expected result here",
      "requirement_id": "REQ001"
    }}
  ]
}}
"""
    return llm.invoke(prompt)
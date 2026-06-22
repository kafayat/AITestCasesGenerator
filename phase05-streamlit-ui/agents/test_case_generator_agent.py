# Test Case Generator Agent
def generate_test_cases(llm, requirement, analysis):

    # Prompt asks AI to generate structured JSON only
    prompt = f"""
Generate valid JSON only.

Requirement:
{requirement}

Analysis:
{analysis}

Create exactly 2 test cases:
1 positive
1 negative

Use this exact JSON format:

{{
  "test_cases": [
    {{
      "test_case_id": "TC001",
      "title": "Valid login",
      "priority": "High",
      "test_type": "Positive",
      "preconditions": "User exists",
      "test_data": "valid username and password",
      "steps": [
        "Open login page",
        "Enter valid username",
        "Enter valid password",
        "Click login"
      ],
      "expected_result": "User logs in successfully",
      "requirement_id": "REQ001"
    }},
    {{
      "test_case_id": "TC002",
      "title": "Invalid login",
      "priority": "High",
      "test_type": "Negative",
      "preconditions": "User is on login page",
      "test_data": "invalid password",
      "steps": [
        "Open login page",
        "Enter valid username",
        "Enter invalid password",
        "Click login"
      ],
      "expected_result": "Error message is displayed",
      "requirement_id": "REQ001"
    }}
  ]
}}

Rules:
- JSON only
- No markdown
- No explanation
- No extra text
"""

    # Return AI-generated JSON response
    return llm.invoke(prompt)
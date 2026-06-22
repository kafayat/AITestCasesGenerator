# Test Case Generator Agent with RAG support
# This agent uses requirement + analysis + company knowledge
def generate_test_cases(llm, requirement, analysis, rag_context):

    # Prompt includes RAG knowledge from knowledge_base files
    prompt = f"""
Generate valid JSON only.

Requirement:
{requirement}

Requirement Analysis:
{analysis}

Company Knowledge:
{rag_context}

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
      "preconditions": "User exists and is active",
      "test_data": "valid username and valid password",
      "steps": [
        "Open login page",
        "Enter valid username",
        "Enter valid password",
        "Click login"
      ],
      "expected_result": "User logs in successfully and is redirected to dashboard",
      "requirement_id": "REQ001"
    }},
    {{
      "test_case_id": "TC002",
      "title": "Invalid login with incorrect password",
      "priority": "High",
      "test_type": "Negative",
      "preconditions": "User exists and is active",
      "test_data": "valid username and invalid password",
      "steps": [
        "Open login page",
        "Enter valid username",
        "Enter invalid password",
        "Click login"
      ],
      "expected_result": "System displays a generic login error without revealing which field is incorrect",
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

    # Return generated JSON
    return llm.invoke(prompt)
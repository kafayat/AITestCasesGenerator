def generate_test_cases(llm, requirement, analysis):
    prompt = f"""
You are a Senior QA Test Case Generator Agent.

Using the requirement and analysis below, generate test cases.

Requirement:
{requirement}

Requirement Analysis:
{analysis}

Generate:
- Positive test cases
- Negative test cases
- Boundary test cases
- Security test cases

Format each test case with:
Test Case ID
Title
Priority
Type
Preconditions
Test Data
Steps
Expected Result
"""
    return llm.invoke(prompt)
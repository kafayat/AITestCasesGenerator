def analyze_requirement(llm, requirement):
    prompt = f"""
Analyze this requirement.

Requirement:
{requirement}

Return:
Feature:
Actor:
Action:
Acceptance Criteria:
Validations:
Missing Information:
Risk Areas:
"""
    return llm.invoke(prompt)
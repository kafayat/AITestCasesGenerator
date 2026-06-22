# Requirement Analyzer Agent
def analyze_requirement(llm, requirement):

    # Short prompt for faster response
    prompt = f"""
Analyze this requirement in short form.

Requirement:
{requirement}

Return only:
Feature:
Actor:
Action:
Validation:
Missing:
"""

    # Send prompt to AI model
    return llm.invoke(prompt)
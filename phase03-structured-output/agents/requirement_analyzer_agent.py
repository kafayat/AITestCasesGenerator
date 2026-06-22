# This agent reads the requirement and gives a short analysis
def analyze_requirement(llm, requirement):

    # Keep prompt short so model responds faster
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

    # Send prompt to LLM
    return llm.invoke(prompt)
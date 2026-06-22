# This function is the Requirement Analyzer Agent
# Its job is to read the requirement and summarize key information
def analyze_requirement(llm, requirement):

    # Keep the prompt short so the local model runs faster
    # The model will extract only key requirement details
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

    # Send the prompt to the LLM model and return the AI response
    return llm.invoke(prompt)
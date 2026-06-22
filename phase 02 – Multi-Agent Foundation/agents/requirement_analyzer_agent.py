def analyze_requirement(llm, requirement):
    prompt = f"""
You are a Requirement Analyzer Agent.

Analyze this requirement and identify:
- Feature
- Actor/User
- Action
- Business Rules
- Validations
- Missing Information

Requirement:
{requirement}
"""
    return llm.invoke(prompt)
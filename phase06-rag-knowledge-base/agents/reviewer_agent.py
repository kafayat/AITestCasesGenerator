# Reviewer Agent
# This agent checks quality of generated test cases
def review_test_cases(llm, test_cases_json):

    # Limit text size for faster review
    short_test_cases = test_cases_json[:1200]

    # Reviewer prompt
    prompt = f"""
Review briefly in 3 bullet points only.

Check:
- missing scenarios
- duplicate test cases
- unclear expected results
- alignment with QA standards

Test cases:
{short_test_cases}
"""

    # Return review response
    return llm.invoke(prompt)
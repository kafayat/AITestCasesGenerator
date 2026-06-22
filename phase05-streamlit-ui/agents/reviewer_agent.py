# Reviewer Agent
def review_test_cases(llm, test_cases_json):

    # Limit test case size for faster review
    short_test_cases = test_cases_json[:1200]

    # Short reviewer prompt
    prompt = f"""
Review briefly in 3 bullet points only.

Check:
- missing cases
- duplicate cases
- unclear expected results

Test cases:
{short_test_cases}
"""

    # Return reviewer feedback
    return llm.invoke(prompt)
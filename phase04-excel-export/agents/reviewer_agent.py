# This agent reviews test cases quickly
def review_test_cases(llm, test_cases_json):

    # Limit input size for faster review
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

    return llm.invoke(prompt)
def review_test_cases(llm, test_cases):
    prompt = f"""
You are a QA Reviewer.

Review the test cases in maximum 5 bullet points only.

Check:
1. Missing scenarios
2. Duplicate cases
3. Unclear expected results

Test Cases:
{test_cases[:2000]}
"""
    return llm.invoke(prompt)
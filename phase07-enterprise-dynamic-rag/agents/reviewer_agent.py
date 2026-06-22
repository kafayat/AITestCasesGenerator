def review_test_cases(llm, test_cases_json, rag_context):
    short_test_cases = test_cases_json[:2000]
    short_context = rag_context[:1200]

    prompt = f"""
You are an Enterprise QA Reviewer.

Review the generated test cases against enterprise knowledge.

Enterprise Knowledge:
{short_context}

Generated Test Cases:
{short_test_cases}

Check:
- Coverage
- Negative scenarios
- Boundary scenarios
- Security scenarios
- Traceability
- Missing business rules

Return maximum 5 bullet points.
"""
    return llm.invoke(prompt)
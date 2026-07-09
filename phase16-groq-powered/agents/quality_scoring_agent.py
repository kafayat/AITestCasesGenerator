from utils.prompt_loader import load_prompt


def score_test_case_quality(
    llm,
    requirement,
    analysis,
    rag_context,
    test_cases_json
):

    prompt_template = load_prompt(
        "prompts/quality_scoring_prompt.txt"
    )

    prompt = (
        prompt_template
        .replace("{requirement}", requirement[:1200])
        .replace("{analysis}", analysis[:800])
        .replace("{rag_context}", rag_context[:1000])
        .replace("{test_cases}", test_cases_json[:2500])
    )

    return llm.invoke(prompt)
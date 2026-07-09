from utils.prompt_loader import load_prompt


def review_test_cases(
    llm,
    requirement,
    analysis,
    rag_context,
    test_cases
):

    prompt_template = load_prompt(
        "prompts/reviewer_prompt.txt"
    )

    prompt = (
        prompt_template
        .replace("{requirement}", requirement[:1000])
        .replace("{analysis}", analysis[:800])
        .replace("{rag_context}", rag_context[:1000])
        .replace("{test_cases}", test_cases[:1500])
    )

    return llm.invoke(prompt)
from utils.prompt_loader import load_prompt


def generate_test_cases(
    llm,
    requirement,
    analysis,
    rag_context
):

    prompt_template = load_prompt(
        "prompts/test_case_generator_prompt.txt"
    )

    prompt = (
        prompt_template
        .replace("{requirement}", requirement)
        .replace("{analysis}", analysis)
        .replace("{rag_context}", rag_context)
    )

    return llm.invoke(prompt)
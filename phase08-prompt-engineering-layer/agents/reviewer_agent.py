# Import prompt loader utility
from utils.prompt_loader import load_prompt


# Reviewer Agent
def review_test_cases(
    llm,
    requirement,
    analysis,
    rag_context,
    test_cases
):

    # Load reviewer prompt template
    prompt_template = load_prompt(
        "prompts/reviewer_prompt.txt"
    )

    # Keep content short for local model speed
    prompt = prompt_template.format(
        requirement=requirement[:1500],
        analysis=analysis[:1200],
        rag_context=rag_context[:2000],
        test_cases=test_cases[:2500]
    )

    # Send prompt to LLM
    return llm.invoke(prompt)
# Import prompt loader utility
from utils.prompt_loader import load_prompt


# Enterprise Test Case Generator Agent
def generate_test_cases(
    llm,
    requirement,
    analysis,
    rag_context
):

    # Load test case generator prompt template
    prompt_template = load_prompt(
        "prompts/test_case_generator_prompt.txt"
    )

    # Inject dynamic values into prompt
    prompt = prompt_template.format(
        requirement=requirement,
        analysis=analysis,
        rag_context=rag_context
    )

    # Send prompt to LLM
    return llm.invoke(prompt)
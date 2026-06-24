from utils.prompt_loader import load_prompt


def analyze_requirement(llm, requirement):

    prompt_template = load_prompt(
        "prompts/requirement_analyzer_prompt.txt"
    )

    prompt = prompt_template.replace(
        "{requirement}",
        requirement
    )

    return llm.invoke(prompt)
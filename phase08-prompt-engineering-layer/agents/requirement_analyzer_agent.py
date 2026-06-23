# Import prompt loader utility
from utils.prompt_loader import load_prompt


# Requirement Analyzer Agent
def analyze_requirement(llm, requirement):

    # Load prompt template from external file
    prompt_template = load_prompt(
        "prompts/requirement_analyzer_prompt.txt"
    )

    # Inject requirement into prompt template
    prompt = prompt_template.format(
        requirement=requirement
    )

    # Send prompt to LLM
    return llm.invoke(prompt)
from utils.prompt_loader import load_prompt


def infer_requirement_from_knowledge(
    llm,
    rag_context
):

    prompt_template = load_prompt(
        "prompts/requirement_inference_prompt.txt"
    )

    prompt = prompt_template.replace(
        "{rag_context}",
        rag_context[:2500]
    )

    return llm.invoke(prompt)
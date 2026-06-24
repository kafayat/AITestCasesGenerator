from langchain_ollama import OllamaLLM


def get_llm(provider, model_name):

    if provider == "Ollama":
        return OllamaLLM(
            model=model_name
        )

    raise ValueError(
        f"Unsupported model provider: {provider}"
    )
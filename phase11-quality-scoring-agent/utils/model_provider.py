from langchain_ollama import OllamaLLM


def get_llm(provider, model_name):

    if provider == "Ollama":

        return OllamaLLM(
            model=model_name,
            temperature=0.1,
            num_predict=6000
        )

    raise ValueError(
        f"Unsupported model provider: {provider}"
    )
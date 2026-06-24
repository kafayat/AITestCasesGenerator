"""
Model provider with task-based routing.

Phase 11 used ONE model instance for every agent (generation, analysis,
scoring, review) with a fixed num_predict=6000. That meant the cheapest
tasks (e.g. requirement analysis) paid the same generation ceiling as the
heaviest one (test case generation), and there was no way to use a smaller
/ faster model for high-frequency calls while reserving a bigger model for
calls that actually need deeper reasoning.

Phase 12 keeps Ollama but routes by task:
- "generation"  -> small, fast model, low num_predict (called many times,
                   once per coverage group, concurrently)
- "analysis"    -> small model, very low num_predict
- "scoring"     -> can use a larger model, called once per run
- "review"      -> can use a larger model, called once per run (optional)
- "repair"      -> small model, low num_predict (only used as last resort,
                   after pure-python JSON repair has already been tried)
"""

from langchain_ollama import OllamaLLM


# Conservative output ceilings per task. These are deliberately smaller
# than Phase 11's flat 6000, because each generation call now only has to
# cover 2-3 coverage areas instead of all 12 in one shot.
NUM_PREDICT_BY_TASK = {
    "generation": 1400,
    "analysis": 500,
    "scoring": 900,
    "review": 700,
    "inference": 900,
    "repair": 1200,
}

_llm_cache = {}


def get_llm(provider, model_name, task="generation"):

    if provider != "Ollama":
        raise ValueError(f"Unsupported model provider: {provider}")

    num_predict = NUM_PREDICT_BY_TASK.get(task, 1200)

    cache_key = (provider, model_name, num_predict)

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
        num_predict=num_predict,
    )

    _llm_cache[cache_key] = llm

    return llm

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

Phase 14 update: temperature is now also routed by task, not a flat 0.1
for everything. Real-world testing showed the "lifecycle_operations"
generation group copying its own few-shot worked examples almost
verbatim (3 NEXDOC test cases that were word-for-word the prompt's
example content, with zero domain-specific wording at all). A
temperature of 0.1 combined with a concrete, well-written example in
context makes rote copying the path of least resistance for a small
model - it's literally the most probable continuation. Raising
generation temperature gives the model room to diverge from the example
while the JSON-structure instructions (and the post-generation
"copied the example" filter in test_case_generator_agent.py) keep the
output usable. Tasks that need precise, structured, low-variance output
(scoring, repair) stay at low temperature.
"""

from langchain_ollama import OllamaLLM


# Conservative output ceilings per task. These are deliberately smaller
# than Phase 11's flat 6000, because each generation call now only has to
# cover 2-3 coverage areas instead of all 12 in one shot.
NUM_PREDICT_BY_TASK = {
    "generation": 1200,
    "analysis": 500,
    "scoring": 900,
    "review": 700,
    "inference": 900,
    "repair": 1000,
}

TEMPERATURE_BY_TASK = {
    # 0.1 for generation works well for 7B models which follow instructions
    # reliably without needing high temperature to avoid example-copying.
    # The _filter_copied_examples() check handles any residual copying.
    "generation": 0.1,
    "analysis": 0.1,
    "scoring": 0.1,
    "review": 0.15,
    "inference": 0.15,
    "repair": 0.1,
}

_llm_cache = {}


def get_llm(provider, model_name, task="generation"):

    if provider != "Ollama":
        raise ValueError(f"Unsupported model provider: {provider}")

    num_predict = NUM_PREDICT_BY_TASK.get(task, 1200)
    temperature = TEMPERATURE_BY_TASK.get(task, 0.1)

    cache_key = (provider, model_name, num_predict, temperature)

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
    )

    _llm_cache[cache_key] = llm

    return llm

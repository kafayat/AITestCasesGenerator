"""
Phase 16 — Multi-provider model support.

Providers supported:
  1. Groq       — free, extremely fast (500-800 t/s), no credit card needed
                  Best models: llama-3.1-70b-versatile, mixtral-8x7b-32768
                  Sign up: console.groq.com — get API key in 2 minutes
                  28 scenarios: under 60 seconds

  2. Azure AI   — GPT-4o-mini, reliable, ~$0.002 per 28-scenario run
  Foundry         $200 free credit on new Azure accounts

  3. Ollama     — fully local, free, no internet needed
                  Slow on CPU (~90s per call), good for offline use

All three expose the same .invoke() and .stream() interface so the
rest of the codebase needs zero changes regardless of which provider
is selected.
"""

from langchain_ollama import OllamaLLM


NUM_PREDICT_BY_TASK = {
    "generation": 2000,
    "analysis":    600,
    "scoring":    1000,
    "review":      800,
    "inference":   900,
    "repair":     1000,
}

TEMPERATURE_BY_TASK = {
    "generation": 0.1,
    "analysis":   0.1,
    "scoring":    0.1,
    "review":     0.15,
    "inference":  0.15,
    "repair":     0.1,
}

_llm_cache = {}


class GroqLLM:
    """
    Wraps the Groq client with the same .invoke()/.stream() interface
    as OllamaLLM so the rest of the codebase needs zero changes.

    Groq runs open-source models (Llama 3.1, Mixtral, Gemma 2) on
    custom LPU hardware — typically 500-800 tokens/second, which makes
    28 scenario calls complete in under 60 seconds total.

    Recommended models:
      llama-3.1-70b-versatile  — best quality, still very fast
      llama-3.1-8b-instant     — fastest, good for simple scenarios
      mixtral-8x7b-32768       — good at structured JSON output
      gemma2-9b-it             — lightweight, reliable
    """

    def __init__(self, api_key, model_name,
                 temperature=0.1, max_tokens=1200):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureFoundryLLM:
    """
    Wraps the Azure OpenAI client with the same interface.
    """

    def __init__(self, endpoint, deployment_name, api_key,
                 temperature=0.1, max_tokens=1200):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version="2024-08-01-preview",
            api_key=api_key,
        )
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_llm(provider, model_name, task="generation",
            groq_api_key=None,
            azure_endpoint=None, azure_api_key=None):

    temperature = TEMPERATURE_BY_TASK.get(task, 0.1)
    max_tokens = NUM_PREDICT_BY_TASK.get(task, 1200)

    if provider == "Groq":
        if not groq_api_key:
            raise ValueError(
                "Groq requires an API key. "
                "Sign up free at console.groq.com and paste your key in the sidebar."
            )
        cache_key = ("groq", model_name, task)
        if cache_key not in _llm_cache:
            _llm_cache[cache_key] = GroqLLM(
                api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _llm_cache[cache_key]

    if provider == "Azure AI Foundry":
        if not azure_endpoint or not azure_api_key:
            raise ValueError(
                "Azure AI Foundry requires an endpoint URL and API key."
            )
        cache_key = ("azure", model_name, task)
        if cache_key not in _llm_cache:
            _llm_cache[cache_key] = AzureFoundryLLM(
                endpoint=azure_endpoint,
                deployment_name=model_name,
                api_key=azure_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _llm_cache[cache_key]

    if provider == "Ollama":
        cache_key = ("ollama", model_name, max_tokens, temperature)
        if cache_key not in _llm_cache:
            _llm_cache[cache_key] = OllamaLLM(
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
            )
        return _llm_cache[cache_key]

    raise ValueError(f"Unsupported provider: {provider}")

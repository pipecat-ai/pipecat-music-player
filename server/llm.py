"""LLM service factory for the voice + UI worker LLMs.

Selects a pipecat ``LLMService`` based on the ``LLM_PROVIDER`` env var so
the demo can be retargeted at a different provider without touching the
worker code. Each branch reads its own ``<PROVIDER>_API_KEY`` and optional
``<PROVIDER>_MODEL``, and imports the provider's SDK lazily — only the
provider you select needs to be installed (add the corresponding
``pipecat-ai`` extra to ``pyproject.toml``).

Note: ``descriptions.py`` (LLM-generated catalog blurbs and Q&A) calls
the OpenAI SDK directly and needs ``OPENAI_API_KEY`` regardless of
``LLM_PROVIDER``.
"""

import os

from pipecat.services.llm_service import LLMService


def create_llm_service(*, system_prompt: str) -> LLMService:
    """Build the configured LLM service with the given system prompt.

    Reads ``LLM_PROVIDER`` (default ``"openai"``). Adding a provider is a
    new ``elif`` branch with the same shape: lazy-import the service + its
    settings, read its API-key + model env vars, return the service.

    Args:
        system_prompt: The system instruction the worker's LLM should use.

    Returns:
        A pipecat ``LLMService`` ready to drop into a pipeline or a worker.

    Raises:
        KeyError: If the provider's API-key env var is missing.
        ValueError: If ``LLM_PROVIDER`` is not a known provider.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        from pipecat.services.openai.base_llm import OpenAILLMSettings
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                system_instruction=system_prompt,
                model=os.getenv("OPENAI_MODEL"),
            ),
        )

    if provider == "cerebras":
        from pipecat.services.cerebras.llm import CerebrasLLMService, CerebrasLLMSettings

        return CerebrasLLMService(
            api_key=os.environ["CEREBRAS_API_KEY"],
            settings=CerebrasLLMSettings(
                system_instruction=system_prompt,
                model=os.getenv("CEREBRAS_MODEL"),
            ),
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Expected one of: openai, cerebras.")

"""Model name validators for each provider."""

from .model_catalog import get_known_models


VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in ("ollama", "ollama_cloud", "openrouter", "deepinfra", "mimo", "custom_openai", "lm-studio", "llama-cpp")
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For ollama, ollama_cloud, openrouter, deepinfra, custom_openai, lm-studio, llama-cpp - any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in ("ollama", "ollama_cloud", "openrouter", "deepinfra", "mimo", "custom_openai", "lm-studio", "llama-cpp"):
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]

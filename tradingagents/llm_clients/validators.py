"""Model name validators for each provider."""

from .model_catalog import get_known_models


_OPEN_CATALOG_PROVIDERS = (
    "ollama", "ollama_cloud", "openrouter", "deepinfra", "mimo", "custom_openai", "lmstudio", "lm-studio", "llama-cpp", "tencent"
)

VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in _OPEN_CATALOG_PROVIDERS
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For open catalog and Tencent providers - any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in _OPEN_CATALOG_PROVIDERS:
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]

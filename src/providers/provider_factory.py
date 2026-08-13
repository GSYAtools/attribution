from src.providers.base import LLMProvider
from src.providers.mock_provider import MockProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.openai_provider import OpenAIProvider

def create_provider(provider_name: str) -> LLMProvider:
    """
    Create an LLM provider from its configured name.

    Real providers will be added here only after their
    implementations have been tested independently.
    """

    providers = {
        "mock": MockProvider,
	"openai": OpenAIProvider,
	"gemini": GeminiProvider
    }

    if provider_name not in providers:
        available = ", ".join(sorted(providers))

        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {available}"
        )

    return providers[provider_name]()

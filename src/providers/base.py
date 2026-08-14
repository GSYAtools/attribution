from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    """Normalized response returned by an LLM provider."""

    text: str
    model_id: str
    provider: str
    usage: dict[str, Any] | None = None
    raw_response: Any | None = None


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    provider_name: str

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        generation_config: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> ModelResponse:
        """
        Generate one model response.

        Implementations must return a normalized ModelResponse.

        output_schema contains the JSON Schema that the provider
        must enforce for the generated response.
        """
        raise NotImplementedError

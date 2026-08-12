from typing import Any

from src.providers.base import LLMProvider, ModelResponse


class FailingProvider(LLMProvider):
    provider_name = "failing"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        generation_config: dict[str, Any],
    ) -> ModelResponse:
        raise RuntimeError("Simulated provider failure")


def test_provider_failure_is_raised():
    provider = FailingProvider()

    try:
        provider.generate(
            system_prompt="system",
            user_prompt="user",
            model_id="test-model",
            generation_config={},
        )
    except RuntimeError as exc:
        assert str(exc) == "Simulated provider failure"
    else:
        raise AssertionError(
            "Expected provider failure was not raised."
        )

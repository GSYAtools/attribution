from typing import Any

from src.providers.base import LLMProvider, ModelResponse


class MockProvider(LLMProvider):
    """Deterministic provider used for local pipeline testing."""

    provider_name = "mock"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        generation_config: dict[str, Any],
    ) -> ModelResponse:

        response = (
            '{"attribution":"insufficient_basis",'
            '"confidence":50,'
            '"justification":"Mock response for pipeline testing.",'
            '"evidence_used":["Mock evidence."],'
            '"limitations":["Mock limitation."],'
            '"action":"abstain"}'
        )

        return ModelResponse(
            text=response,
            model_id=model_id,
            provider=self.provider_name,
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            raw_response=None,
        )

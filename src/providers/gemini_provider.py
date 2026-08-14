from typing import Any

from google import genai
from google.genai import types

from src.providers.base import LLMProvider, ModelResponse
from src.providers.errors import (
    ProviderAuthenticationError,
    ProviderInvalidRequestError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnknownError,
)
from src.utils import get_env


class GeminiProvider(LLMProvider):
    """Google Gemini implementation of the common provider interface."""

    provider_name = "gemini"

    def __init__(self) -> None:
        self.client = genai.Client(
            api_key=get_env("GEMINI_API_KEY")
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        generation_config: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> ModelResponse:

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": output_schema,
            "thinking_config": types.ThinkingConfig(
                thinking_level="low"
            ),
        }

        max_output_tokens = generation_config.get(
            "max_output_tokens"
        )

        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = (
                max_output_tokens
            )

        try:
            response = self.client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    **config_kwargs
                ),
            )

        except Exception as exc:
            error_text = str(exc)
            error_lower = error_text.lower()

            if any(
                marker in error_lower
                for marker in (
                    "api key",
                    "authentication",
                    "unauthorized",
                )
            ):
                raise ProviderAuthenticationError(
                    provider=self.provider_name,
                    category="authentication",
                    message=error_text,
                    retryable=False,
                    original_exception=exc,
                ) from exc

            if any(
                marker in error_lower
                for marker in (
                    "quota",
                    "resource exhausted",
                    "insufficient",
                )
            ):
                raise ProviderQuotaError(
                    provider=self.provider_name,
                    category="insufficient_quota",
                    message=error_text,
                    retryable=False,
                    original_exception=exc,
                ) from exc

            if any(
                marker in error_lower
                for marker in (
                    "429",
                    "rate limit",
                    "too many requests",
                )
            ):
                raise ProviderRateLimitError(
                    provider=self.provider_name,
                    category="rate_limit",
                    message=error_text,
                    retryable=True,
                    original_exception=exc,
                ) from exc

            if any(
                marker in error_lower
                for marker in (
                    "timeout",
                    "timed out",
                )
            ):
                raise ProviderTimeoutError(
                    provider=self.provider_name,
                    category="timeout",
                    message=error_text,
                    retryable=True,
                    original_exception=exc,
                ) from exc

            if any(
                marker in error_lower
                for marker in (
                    "invalid argument",
                    "invalid request",
                    "bad request",
                )
            ):
                raise ProviderInvalidRequestError(
                    provider=self.provider_name,
                    category="invalid_request",
                    message=error_text,
                    retryable=False,
                    original_exception=exc,
                ) from exc

            raise ProviderUnknownError(
                provider=self.provider_name,
                category="api_error",
                message=error_text,
                retryable=False,
                original_exception=exc,
            ) from exc

        usage = None

        if getattr(
            response,
            "usage_metadata",
            None,
        ) is not None:

            usage_metadata = response.usage_metadata

            usage = {
                "input_tokens": getattr(
                    usage_metadata,
                    "prompt_token_count",
                    None,
                ),
                "output_tokens": getattr(
                    usage_metadata,
                    "candidates_token_count",
                    None,
                ),
                "total_tokens": getattr(
                    usage_metadata,
                    "total_token_count",
                    None,
                ),
            }

        return ModelResponse(
            text=response.text,
            model_id=model_id,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
        )

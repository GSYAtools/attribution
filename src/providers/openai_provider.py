from typing import Any

import openai
from openai import OpenAI

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


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the common LLM provider interface."""

    provider_name = "openai"

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=get_env("OPENAI_API_KEY")
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        generation_config: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> ModelResponse:

        request: dict[str, Any] = {
            "model": model_id,
            "instructions": system_prompt,
            "input": user_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "sc5_output",
                    "schema": output_schema,
                    "strict": True,
                }
            },
        }

        max_output_tokens = generation_config.get(
            "max_output_tokens"
        )

        if max_output_tokens is not None:
            request["max_output_tokens"] = (
                max_output_tokens
            )

        try:
            response = self.client.responses.create(
                **request
            )

        except openai.AuthenticationError as exc:
            raise ProviderAuthenticationError(
                provider=self.provider_name,
                category="authentication",
                message=str(exc),
                retryable=False,
                original_exception=exc,
            ) from exc

        except openai.RateLimitError as exc:
            message = str(exc)

            if (
                "insufficient_quota" in message
                or "credit_balance_exhausted" in message
                or "no credits remaining"
                in message.lower()
            ):
                raise ProviderQuotaError(
                    provider=self.provider_name,
                    category="insufficient_quota",
                    message=message,
                    retryable=False,
                    original_exception=exc,
                ) from exc

            raise ProviderRateLimitError(
                provider=self.provider_name,
                category="rate_limit",
                message=message,
                retryable=True,
                original_exception=exc,
            ) from exc

        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(
                provider=self.provider_name,
                category="timeout",
                message=str(exc),
                retryable=True,
                original_exception=exc,
            ) from exc

        except openai.BadRequestError as exc:
            raise ProviderInvalidRequestError(
                provider=self.provider_name,
                category="invalid_request",
                message=str(exc),
                retryable=False,
                original_exception=exc,
            ) from exc

        except openai.APIError as exc:
            raise ProviderUnknownError(
                provider=self.provider_name,
                category="api_error",
                message=str(exc),
                retryable=False,
                original_exception=exc,
            ) from exc

        usage = None

        if response.usage is not None:
            usage = {
                "input_tokens": getattr(
                    response.usage,
                    "input_tokens",
                    None,
                ),
                "output_tokens": getattr(
                    response.usage,
                    "output_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    response.usage,
                    "total_tokens",
                    None,
                ),
            }

        return ModelResponse(
            text=response.output_text,
            model_id=model_id,
            provider=self.provider_name,
            usage=usage,
            raw_response=response,
        )

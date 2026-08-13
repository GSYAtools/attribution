from dataclasses import dataclass


@dataclass
class ProviderError(Exception):
    """Normalized error raised by an LLM provider."""

    provider: str
    category: str
    message: str
    retryable: bool
    original_exception: Exception | None = None

    def __str__(self) -> str:
        return (
            f"{self.provider}: {self.category}: "
            f"{self.message}"
        )


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderQuotaError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderInvalidRequestError(ProviderError):
    pass


class ProviderUnknownError(ProviderError):
    pass

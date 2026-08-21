from enum import StrEnum


class ProviderErrorKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    NOT_SUPPORTED = "not_supported"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(
        self,
        *,
        provider: str,
        message: str,
        code: str,
        kind: ProviderErrorKind,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.message = message
        self.code = code
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(message)

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AppError


@dataclass(frozen=True)
class PriceEntry:
    input_per_million_usd: float | None
    output_per_million_usd: float | None
    cached_input_per_million_usd: float | None = None


class PricingCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._prices = self._load(path)

    @staticmethod
    def _load(path: Path) -> dict[tuple[str, str], PriceEntry]:
        if not path.exists():
            raise AppError(
                message=f"Pricing catalog not found: {path}",
                code="PRICING_CATALOG_MISSING",
                status_code=500,
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        prices: dict[tuple[str, str], PriceEntry] = {}

        for provider, models in payload["providers"].items():
            for model, values in models.items():
                prices[(provider, model)] = PriceEntry(
                    input_per_million_usd=values.get("input_per_million_usd"),
                    output_per_million_usd=values.get("output_per_million_usd"),
                    cached_input_per_million_usd=values.get("cached_input_per_million_usd"),
                )
        return prices

    def get(self, provider: str, model: str) -> PriceEntry | None:
        return self._prices.get((provider, model))

    def estimate_cost(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float | None:
        entry = self.get(provider, model)
        if entry is None or entry.input_per_million_usd is None:
            return None

        uncached_input = max(input_tokens - cached_input_tokens, 0)
        input_cost = uncached_input / 1_000_000 * entry.input_per_million_usd

        cached_rate = entry.cached_input_per_million_usd
        if cached_rate is None:
            cached_rate = entry.input_per_million_usd
        cached_cost = cached_input_tokens / 1_000_000 * cached_rate

        output_cost = 0.0
        if entry.output_per_million_usd is not None:
            output_cost = output_tokens / 1_000_000 * entry.output_per_million_usd

        return round(input_cost + cached_cost + output_cost, 10)

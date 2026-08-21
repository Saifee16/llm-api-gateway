from pathlib import Path

from app.core.pricing import PricingCatalog


def test_pricing_catalog_estimates_known_model_cost() -> None:
    catalog = PricingCatalog(Path("config/pricing.json"))
    cost = catalog.estimate_cost(
        provider="gemini",
        model="gemini-3.5-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == 10.5


def test_pricing_catalog_returns_none_for_unknown_price() -> None:
    catalog = PricingCatalog(Path("config/pricing.json"))
    assert (
        catalog.estimate_cost(
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=100,
            output_tokens=0,
        )
        is None
    )

"""The model-provider switch: the registry, the clients it builds, and `GET /models`.

No model call is made. What is testable without one is exactly what breaks the picker if
it regresses -- which ids are on offer, which one a request that names none resolves to,
and the Azure model-name/deployment-name split, which is the single thing in this feature
that fails as an opaque 404 rather than as an error naming the cause.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents import providers
from app.config import get_settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    """Real settings with both providers credentialed, so the catalogue is exercised in
    the state a developer runs in rather than in a half-configured one."""
    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(s, "azure_openai_api_key", "test-azure-key")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://example.openai.azure.com/")
    monkeypatch.setattr(s, "master_model", None)
    monkeypatch.setattr(s, "specialist_model", None)
    return s


def test_catalog_lists_both_providers(settings) -> None:
    ids = [p.id for p in providers.catalog()]
    assert ids == [providers.GEMINI, providers.AZURE_OPENAI]


def test_unconfigured_provider_is_listed_with_a_reason(settings, monkeypatch) -> None:
    """Hidden options are worse than disabled ones: the rep was told they had the key."""
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    azure = next(p for p in providers.catalog() if p.id == providers.AZURE_OPENAI)
    assert azure.configured is False
    assert "AZURE_OPENAI_API_KEY" in azure.unconfigured_reason
    # Still offered in the catalogue, so the dialog can explain rather than omit.
    assert azure.models


def test_resolve_defaults_to_the_configured_provider(settings, monkeypatch) -> None:
    monkeypatch.setattr(settings, "model_provider", providers.AZURE_OPENAI)
    selection = providers.resolve()
    assert selection.provider == providers.AZURE_OPENAI
    assert selection.master_model == settings.azure_openai_model


def test_default_provider_falls_forward_when_the_configured_one_has_no_key(
    settings, monkeypatch
) -> None:
    """A stale MODEL_PROVIDER must not 503 every endpoint while a good key sits in .env."""
    monkeypatch.setattr(settings, "model_provider", providers.GEMINI)
    monkeypatch.setattr(settings, "gemini_api_key", None)
    assert providers.default_provider() == providers.AZURE_OPENAI


def test_unknown_provider_and_model_raise(settings) -> None:
    with pytest.raises(providers.UnknownModel):
        providers.resolve("anthropic")
    with pytest.raises(providers.UnknownModel):
        providers.resolve(providers.AZURE_OPENAI, "gpt-4o")


def test_tier_overrides_apply_only_on_the_provider_that_offers_them(settings, monkeypatch) -> None:
    """A MASTER_MODEL left pointing at Gemini must not follow the rep to Azure and 404."""
    monkeypatch.setattr(settings, "master_model", "gemini-3.5-flash-lite")

    gemini = providers.resolve(providers.GEMINI)
    assert gemini.master_model == "gemini-3.5-flash-lite"

    azure = providers.resolve(providers.AZURE_OPENAI)
    assert azure.master_model == settings.azure_openai_model


def test_an_explicit_choice_beats_the_tier_override(settings, monkeypatch) -> None:
    """The env vars are defaults. A default that overrode a request would make the picker
    silently inert."""
    monkeypatch.setattr(settings, "master_model", "gemini-3.5-flash-lite")
    selection = providers.resolve(providers.GEMINI, "gemini-3.5-flash-lite")
    assert selection.master_model == selection.specialist_model == "gemini-3.5-flash-lite"


def test_azure_client_is_built_on_the_deployment_not_the_model_name(settings) -> None:
    """The one failure mode here that surfaces as an opaque 404 rather than an error
    naming the cause."""
    model_id = "gpt-5.4-nano"
    selection = providers.resolve(providers.AZURE_OPENAI, model_id)
    client = providers.build_chat_model(selection, model_id)

    assert client.deployment_name == settings.azure_openai_deployments[model_id]
    assert client.deployment_name != model_id
    # `max_tokens` is serialized as `max_completion_tokens`, which is what the GPT-5
    # family requires; `max_tokens` is rejected outright.
    assert client.max_tokens == providers.MAX_OUTPUT_TOKENS[providers.AZURE_OPENAI]
    # Unset on purpose: these deployments accept only the default temperature.
    assert client.temperature is None


def test_rate_limiters_are_shared_per_provider_and_not_across_them(settings) -> None:
    """Shared within a provider because the quota is per project+model; separate across
    providers because the two caps differ by 5x."""
    gemini = providers.build_chat_model(
        providers.resolve(providers.GEMINI), "gemini-3.5-flash-lite"
    )
    gemini_again = providers.build_chat_model(
        providers.resolve(providers.GEMINI), "gemini-3.5-flash-lite"
    )
    azure = providers.build_chat_model(
        providers.resolve(providers.AZURE_OPENAI, "gpt-5.4-mini"), "gpt-5.4-mini"
    )

    assert gemini.rate_limiter is gemini_again.rate_limiter
    assert azure.rate_limiter is not gemini.rate_limiter


def test_build_refuses_a_provider_with_no_credentials(settings, monkeypatch) -> None:
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    selection = providers.resolve(providers.AZURE_OPENAI, "gpt-5.4-mini")
    with pytest.raises(providers.ProviderNotConfigured):
        providers.build_chat_model(selection, "gpt-5.4-mini")


# ------------------------------------------------------------------------- HTTP


def test_models_endpoint_serves_the_picker(client: TestClient, settings) -> None:
    body = client.get("/models").json()

    assert body["default_provider"] in {providers.GEMINI, providers.AZURE_OPENAI}
    assert body["default_model"]

    azure = next(p for p in body["providers"] if p["id"] == providers.AZURE_OPENAI)
    ids = [m["id"] for m in azure["models"]]
    assert ids == list(settings.azure_openai_deployments)
    # The deployment is a subtitle, never the selectable id -- exposing it invites someone
    # to paste it back in as a model name.
    assert all(m["description"].startswith("deployment ") for m in azure["models"])


def test_health_reports_each_provider_separately(client: TestClient, settings) -> None:
    """Either provider on its own is enough to run, so a single flag would have the
    sidebar warning about a failure the rep would not experience."""
    body = client.get("/health").json()
    assert body["model_providers_configured"] == {
        providers.GEMINI: True,
        providers.AZURE_OPENAI: True,
    }
    assert body["any_model_provider_configured"] is True


def test_campaign_run_rejects_an_unoffered_model_as_a_client_error(
    client: TestClient, settings
) -> None:
    """400, not 503: a stale localStorage value in the browser is the caller's problem to
    fix, and collapsing it into 503 made it look like a broken backend."""
    response = client.post(
        "/campaign/run",
        json={"query": "hi", "provider": providers.AZURE_OPENAI, "model": "gpt-4o"},
    )
    assert response.status_code == 400
    assert "gpt-4o" in response.json()["detail"]


def test_campaign_run_reports_missing_credentials_as_unavailable(
    client: TestClient, settings, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    response = client.post(
        "/campaign/run",
        json={"query": "hi", "provider": providers.AZURE_OPENAI, "model": "gpt-5.4-mini"},
    )
    assert response.status_code == 503
    assert "AZURE_OPENAI_API_KEY" in response.json()["detail"]

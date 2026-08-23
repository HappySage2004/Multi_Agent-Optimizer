"""`GET /models` — the catalogue behind the settings dialog's model picker.

One read-only endpoint. It exists because the frontend must not hold its own copy of
which providers are wired up: a hard-coded list goes stale the moment a key is added or
a deployment is renamed, and the rep would be offered an option that 503s.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agents import providers
from app.api.schemas import ModelOptionOut, ModelsOut, ProviderOut

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsOut)
def list_models() -> ModelsOut:
    """Providers, their models, and which one a request that names neither will use."""
    default_provider = providers.default_provider()
    try:
        default_model = providers.resolve(default_provider).master_model
    except providers.UnknownModel:
        # A provider configured with no deployments. Report it honestly rather than 500 --
        # the per-provider `unconfigured_reason` below says what is missing.
        default_model = ""
    return ModelsOut(
        default_provider=default_provider,
        default_model=default_model,
        providers=[
            ProviderOut(
                id=info.id,
                label=info.label,
                models=[
                    ModelOptionOut(id=m.id, label=m.label, description=m.description)
                    for m in info.models
                ],
                default_model=info.default_model,
                configured=info.configured,
                unconfigured_reason=info.unconfigured_reason,
                requests_per_minute=info.requests_per_minute,
            )
            for info in providers.catalog()
        ],
    )

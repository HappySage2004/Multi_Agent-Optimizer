"""Which model providers the agent layer can run on, and how to build a client for one.

Two providers, one switch. The hackathon supplied both a Gemini key and a set of Azure
OpenAI deployments, and they behave differently enough that the choice belongs to the rep
rather than to a deploy-time env var: Gemini's free tier allows ~20 requests/day/model and
one full orchestration costs 15-20 of them, so a demo runs out of Gemini before it runs
out of questions.

The selection therefore rides on the **request** (`CampaignQuery.provider` / `.model`),
the same way pricing levers ride on the run rather than on a delegation message. Nothing
here is global state: `resolve()` turns whatever the UI sent (or nothing) into a
`ModelSelection`, and `build_chat_model` is the only place a client is constructed.

Three things this module owns, and they are here rather than in `master.py` so the
`/models` endpoint can serve the same catalogue the agent builder consumes:

1. **The catalogue** — what the UI may offer, and whether each provider has credentials.
2. **The Azure model-name/deployment-name split.** They are different strings on Azure and
   sending the model name gets a 404, so the mapping is applied in exactly one place.
3. **The rate limiter, one per provider.** It must be shared across the Master and both
   specialists, because the quota is per project+model — three separate limiters would
   each assume the whole budget. It must *not* be shared across providers, because the
   two caps are an order of magnitude apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.config import get_settings
from app.logging_utils import debug

GEMINI = "gemini"
AZURE_OPENAI = "azure_openai"

PROVIDER_IDS = (GEMINI, AZURE_OPENAI)

PROVIDER_LABELS = {
    GEMINI: "Google Gemini",
    AZURE_OPENAI: "Azure OpenAI",
}

# Reasoning tokens are billed against the completion cap on the GPT-5 family, so a cap
# sized for Gemini's visible output starves the answer. Per provider rather than one
# constant, because the failure is silent — a truncated tool call, not an error.
MAX_OUTPUT_TOKENS = {
    GEMINI: 8_192,
    AZURE_OPENAI: 16_384,
}


class UnknownModel(ValueError):
    """The caller named a provider or model this build does not offer."""


class ProviderNotConfigured(RuntimeError):
    """The selected provider has no credentials. Distinct from a bad model id: the fix is
    an env var, not a different request."""


@dataclass(frozen=True)
class ModelOption:
    """One entry in the model picker."""

    id: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    """One provider as the settings dialog needs to render it."""

    id: str
    label: str
    models: list[ModelOption] = field(default_factory=list)
    default_model: str = ""
    configured: bool = False
    #: Empty when `configured`. Otherwise names the env var that is missing, so the rep is
    #: told what to fix rather than just seeing the option greyed out.
    unconfigured_reason: str = ""
    requests_per_minute: float = 0.0


@dataclass(frozen=True)
class ModelSelection:
    """A resolved, credential-checked choice of provider and per-tier models."""

    provider: str
    master_model: str
    specialist_model: str

    @property
    def label(self) -> str:
        if self.master_model == self.specialist_model:
            return f"{PROVIDER_LABELS[self.provider]}/{self.master_model}"
        return f"{PROVIDER_LABELS[self.provider]}/{self.master_model}+{self.specialist_model}"


# ------------------------------------------------------------------ the catalogue


def _gemini_models() -> list[ModelOption]:
    settings = get_settings()
    # The configured model is always offered even if it is missing from the list, so a
    # GEMINI_MODEL override never produces a picker that cannot select it.
    ids = list(dict.fromkeys([settings.gemini_model, *settings.gemini_models]))
    return [ModelOption(id=i, label=i) for i in ids]


def _azure_models() -> list[ModelOption]:
    settings = get_settings()
    return [
        # The deployment name is deliberately not the option id. It is a per-tenant opaque
        # string that would be meaningless in the UI, and exposing it invites someone to
        # paste it back in as a model name.
        ModelOption(id=name, label=name, description=f"deployment {deployment}")
        for name, deployment in settings.azure_openai_deployments.items()
    ]


def _models(provider: str) -> list[ModelOption]:
    return _gemini_models() if provider == GEMINI else _azure_models()


def _default_model(provider: str) -> str:
    settings = get_settings()
    if provider == GEMINI:
        return settings.gemini_model
    options = _azure_models()
    ids = {o.id for o in options}
    if settings.azure_openai_model in ids:
        return settings.azure_openai_model
    return options[0].id if options else ""


def requests_per_minute(provider: str) -> float:
    settings = get_settings()
    return (
        settings.model_requests_per_minute
        if provider == GEMINI
        else settings.azure_requests_per_minute
    )


def _missing_credentials(provider: str) -> str:
    """Empty string when the provider can be used, else what is missing."""
    settings = get_settings()
    if provider == GEMINI:
        if not settings.gemini_api_key:
            return "GEMINI_API_KEY is not set in the repo-root .env (or backend/.env)."
        return ""
    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
            ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
        )
        if not value
    ]
    if missing:
        return f"{' and '.join(missing)} not set in the repo-root .env (or backend/.env)."
    if not settings.azure_openai_deployments:
        return "AZURE_OPENAI_DEPLOYMENTS is empty — no deployment to call."
    return ""


def is_configured(provider: str) -> bool:
    return _missing_credentials(provider) == ""


def catalog() -> list[ProviderInfo]:
    """Every provider, whether it has credentials, and what it can run.

    Unconfigured providers are still listed. The dialog shows them disabled with the
    reason attached, which is more useful than hiding a capability the rep was told they
    had.
    """
    return [
        ProviderInfo(
            id=pid,
            label=PROVIDER_LABELS[pid],
            models=_models(pid),
            default_model=_default_model(pid),
            configured=is_configured(pid),
            unconfigured_reason=_missing_credentials(pid),
            requests_per_minute=requests_per_minute(pid),
        )
        for pid in PROVIDER_IDS
    ]


def default_provider() -> str:
    """The provider a request that names none will run on.

    Falls forward to a configured provider when the configured default has no
    credentials — otherwise a stale `MODEL_PROVIDER` leaves every endpoint returning 503
    while a perfectly good key sits in the same .env.
    """
    configured = get_settings().model_provider
    if configured in PROVIDER_IDS and is_configured(configured):
        return configured
    for pid in PROVIDER_IDS:
        if is_configured(pid):
            return pid
    # Nothing is configured. Report the declared default so the error names what the
    # operator actually asked for.
    return configured if configured in PROVIDER_IDS else GEMINI


# --------------------------------------------------------------------- resolution


def resolve(provider: str | None = None, model: str | None = None) -> ModelSelection:
    """Turn an optional UI choice into a concrete, validated selection.

    Raises `UnknownModel` for a provider or model this build does not offer. Credentials
    are **not** checked here — `require_credentials` is a separate call, because a bad
    model id is a 400 (the caller's fault) and missing credentials is a 503 (ours).
    """
    pid = provider or default_provider()
    if pid not in PROVIDER_IDS:
        raise UnknownModel(f"Unknown model provider '{pid}'. Expected one of {list(PROVIDER_IDS)}.")

    ids = {o.id for o in _models(pid)}
    if model:
        if model not in ids:
            raise UnknownModel(
                f"'{model}' is not a model this build offers on {PROVIDER_LABELS[pid]}. "
                f"Available: {sorted(ids)}."
            )
        # An explicit choice from the UI is the whole selection: the per-tier env
        # overrides are defaults, and a default must not override a request.
        return ModelSelection(provider=pid, master_model=model, specialist_model=model)

    settings = get_settings()
    base = _default_model(pid)
    if not base:
        raise UnknownModel(f"{PROVIDER_LABELS[pid]} has no models configured.")
    return ModelSelection(
        provider=pid,
        master_model=settings.master_model if settings.master_model in ids else base,
        specialist_model=(settings.specialist_model if settings.specialist_model in ids else base),
    )


def require_credentials(selection: ModelSelection) -> None:
    reason = _missing_credentials(selection.provider)
    if reason:
        raise ProviderNotConfigured(
            f"{PROVIDER_LABELS[selection.provider]} is unavailable. {reason}"
        )


def credentials_hint(provider: str) -> str:
    """Which env var to check when the provider rejects our credentials."""
    return "GEMINI_API_KEY" if provider == GEMINI else "AZURE_OPENAI_API_KEY"


# ------------------------------------------------------------------- the clients


@lru_cache(maxsize=len(PROVIDER_IDS))
def _rate_limiter(provider: str) -> InMemoryRateLimiter:
    """One limiter per provider, shared by the Master and all specialists on it.

    Shared because the cap is per project+model: separate limiters would each think they
    had the full budget and together exceed it. Not shared *across* providers, because
    Gemini's free tier is 15/min and Azure's is nowhere near that tight.
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute(provider) / 60.0,
        check_every_n_seconds=0.1,
        # Allow a small burst so short exchanges are not needlessly slowed.
        max_bucket_size=3,
    )


def build_chat_model(selection: ModelSelection, model_id: str) -> BaseChatModel:
    """A chat client for one tier of a resolved selection.

    Imports are local: neither provider SDK should be a hard import cost for a process
    that only ever runs the deterministic pipeline (the test suite, for one).
    """
    require_credentials(selection)
    settings = get_settings()
    provider = selection.provider
    max_tokens = MAX_OUTPUT_TOKENS[provider]

    debug(
        f"chat model {provider}/{model_id} (max_output_tokens={max_tokens}, "
        f"rate_limit={requests_per_minute(provider):g}/min shared, "
        f"timeout={settings.request_timeout_seconds:g}s, retries={settings.model_max_retries})"
    )

    if provider == GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_id,
            api_key=settings.gemini_api_key,
            max_output_tokens=max_tokens,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.model_max_retries,
            rate_limiter=_rate_limiter(provider),
            # Thinking is left at the model default: the model reasons on its own, and
            # pinning `thinking_level` on this client version stalls the request.
        )

    from langchain_openai import AzureChatOpenAI

    deployment = settings.azure_openai_deployments[model_id]
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        # The deployment goes in the URL; `model` is the human-facing name and is what
        # token accounting and tracing report.
        azure_deployment=deployment,
        model=model_id,
        # AzureChatOpenAI renames this to `max_completion_tokens` on the wire, which is
        # what the GPT-5 family requires — `max_tokens` is rejected outright.
        max_tokens=max_tokens,
        # `temperature` is deliberately unset: the GPT-5 deployments accept only the
        # default, and sending 0.7 (langchain's historical default) is a 400.
        timeout=settings.request_timeout_seconds,
        max_retries=settings.model_max_retries,
        rate_limiter=_rate_limiter(provider),
    )

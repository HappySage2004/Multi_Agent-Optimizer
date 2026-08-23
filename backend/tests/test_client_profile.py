"""Client negotiation profile tests.

The properties that matter are about restraint rather than arithmetic. This model informs a
live sales conversation, so it must never move a price on its own, must never state a
confidence its sample cannot carry, and must never resolve an ambiguous account by guessing.
"""

from __future__ import annotations

import pytest

from app.ml.client_profile import (
    CONFIDENCE_SE_THRESHOLD,
    MIN_LINE_ITEMS,
    SUGGESTION_RANGE,
    ClientProfileModel,
    get_client_profile_model,
)
from app.ml.levers import COMMERCIAL_MULTIPLIER_RANGE
from app.tools import master_tools


@pytest.fixture(scope="module")
def model() -> ClientProfileModel:
    return get_client_profile_model()


@pytest.fixture(scope="module")
def profiles(model):
    return [model.profile(cid) for cid in model._frame.index]


# --- the restraint properties -------------------------------------------------


def test_a_suggestion_never_escapes_its_own_bound(profiles):
    """The suggestion is a nudge, capped tighter than the lever it feeds. An LLM will read
    this number out to a salesperson, so the bound has to hold in code."""
    lo, hi = SUGGESTION_RANGE
    for p in profiles:
        assert lo <= p.suggested_commercial_multiplier <= hi, p.client_id
        # And it must be applicable through the lever without being clamped again.
        assert (
            COMMERCIAL_MULTIPLIER_RANGE[0]
            <= p.suggested_commercial_multiplier
            <= COMMERCIAL_MULTIPLIER_RANGE[1]
        )


def test_a_thin_or_noisy_history_suggests_no_change_at_all(profiles):
    """The core restraint. Within-client spread is about as wide as between-client spread,
    so a departure from 1.0 has to survive its own standard error before it is voiced."""
    for p in profiles:
        if p.confidence in {"none", "weak"}:
            assert p.suggested_commercial_multiplier == 1.0, (
                f"{p.client_id}: {p.confidence} confidence still suggested "
                f"x{p.suggested_commercial_multiplier}"
            )


def test_every_suggestion_is_backed_by_a_measured_departure(profiles):
    suggesting = [p for p in profiles if p.suggested_commercial_multiplier != 1.0]
    assert suggesting, "no client got a suggestion; the test proved nothing"

    for p in suggesting:
        assert p.confidence in {"moderate", "strong"}
        assert p.line_items >= MIN_LINE_ITEMS
        assert p.realized_price_index is not None
        assert p.price_index_standard_error is not None
        # The departure must clear the threshold that justified voicing it.
        distance = abs(p.realized_price_index - 1.0) / p.price_index_standard_error
        assert distance >= CONFIDENCE_SE_THRESHOLD


def test_confidence_is_about_the_departure_not_just_the_sample_size(profiles):
    """A dead-neutral index on 500 line items is precisely measured and says nothing. If
    confidence tracked sample size alone, it would be called 'strong' and mislead a rep."""
    precise_but_neutral = [
        p
        for p in profiles
        if p.line_items >= 200
        and p.realized_price_index is not None
        and abs(p.realized_price_index - 1.0) < 0.005
    ]
    for p in precise_but_neutral:
        assert p.confidence == "weak"
        assert p.suggested_commercial_multiplier == 1.0


def test_talking_points_cite_real_figures(profiles):
    """Same rule the relevance engine follows: no generic sales copy. Every profile has to
    say something a rep could check."""
    for p in profiles:
        assert p.talking_points, p.client_id
        assert any(any(ch.isdigit() for ch in point) for point in p.talking_points), p.client_id


def test_a_price_objection_is_always_surfaced(profiles):
    """The single most useful thing here, and it must never be silently dropped."""
    with_losses = [p for p in profiles if p.price_driven_losses]
    assert with_losses, "no client has a price-driven loss; the test proved nothing"

    for p in with_losses:
        assert any("walked away over price" in point for point in p.talking_points)


def test_an_unrecorded_discount_is_named_rather_than_shown_as_zero(profiles):
    """`price_gap_pct` is not populated on every price-driven lead. A missing gap must read
    as unknown — "asking for 0% off" would tell a rep the exact opposite of the truth."""
    unknown_gap = [p for p in profiles if p.price_driven_losses and p.avg_price_gap_asked is None]
    for p in unknown_gap:
        point = next(x for x in p.talking_points if "walked away over price" in x)
        assert "not recorded" in point
        assert "0% off" not in point


def test_the_leverage_tier_is_never_presented_as_a_forecast(profiles):
    """Per client the tier ordering does not hold — the label tracks account size. Anything
    that reads it must be told so, or a rep will act on a number that inverts."""
    mentions = [
        point for p in profiles for point in p.talking_points if "negotiation_leverage" in point
    ]
    assert mentions
    for point in mentions:
        assert "context only" in point

    for p in profiles:
        assert "NOT a forecast" in p.as_context()["price_behaviour"]["tier_caveat"]


def test_the_context_payload_declares_itself_advisory(profiles):
    for p in profiles[:50]:
        guidance = p.as_context()["guidance"]
        assert "ADVISORY ONLY" in guidance["how_to_use"]
        assert "set_pricing_levers" in guidance["how_to_use"]


# --- lookup -------------------------------------------------------------------


def test_lookup_by_id_and_by_exact_name(model):
    cid = model._frame.index[0]
    name = model.profile(cid).company_name
    assert model.resolve(cid) == [cid]
    assert model.resolve(name) == [cid]
    assert model.resolve(name.lower()) == [cid]


def test_an_unknown_client_resolves_to_nothing_rather_than_a_guess(model):
    assert model.resolve("Definitely Not A Real Advertiser Ltd") == []


def test_profile_on_an_unknown_id_raises(model):
    with pytest.raises(KeyError):
        model.profile("CLI-999999")


# --- the tool surface ---------------------------------------------------------


def test_tool_returns_a_profile_for_a_known_client(model):
    cid = next(c for c in model._frame.index if model.profile(c).line_items > 100)
    out = master_tools.get_client_negotiation_profile.invoke({"client": cid})
    assert out["status"] == "ok"
    assert out["client_id"] == cid
    assert out["price_behaviour"]["realized_price_index"] is not None
    assert "ADVISORY ONLY" in out["guidance"]["how_to_use"]


def test_tool_reports_not_found_rather_than_inventing_a_prospect():
    out = master_tools.get_client_negotiation_profile.invoke({"client": "Nonexistent Brand X"})
    assert out["status"] == "not_found"
    assert "no history" in out["detail"]


def test_tool_asks_rather_than_guessing_when_a_name_is_ambiguous(model):
    """A partial name that matches several accounts must come back as a question. Picking
    one would put the wrong client's negotiation history in a rep's mouth."""
    from collections import Counter

    # Find a substring shared by more than one company name.
    first_words = Counter(
        str(model.profile(c).company_name).split()[0].lower() for c in model._frame.index
    )
    shared = next((w for w, n in first_words.items() if n > 1), None)
    if shared is None:
        pytest.skip("no company-name prefix is shared by two clients in this dataset")

    out = master_tools.get_client_negotiation_profile.invoke({"client": shared})
    assert out["status"] == "ambiguous"
    assert len(out["matches"]) > 1
    assert all("client_id" in m and "company_name" in m for m in out["matches"])


def test_the_profile_tool_creates_no_run_and_touches_no_package():
    """It is a read-only reference lookup, like describe_relevance_model. If it ever starts
    creating a run, `PIPELINE_ENTRY_TOOL` in the frontend would mis-detect a rebuild."""
    from app.services import local_db

    before = len(local_db.list_records(local_db.RUNS))
    master_tools.get_client_negotiation_profile.invoke({"client": "Nonexistent Brand X"})
    assert len(local_db.list_records(local_db.RUNS)) == before


# --- population sanity --------------------------------------------------------


def test_the_repeat_client_share_is_what_the_design_assumes(model):
    """The whole module is justified by 'almost every client is repeat business'. If that
    ever stops being true, the value proposition changes and this should fail."""
    profiles = [model.profile(cid) for cid in model._frame.index]
    repeat = sum(1 for p in profiles if p.is_repeat)
    assert repeat / len(profiles) > 0.90, f"only {repeat}/{len(profiles)} clients are repeat"

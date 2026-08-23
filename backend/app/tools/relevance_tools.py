"""AUDIENCE RELEVANCE ENGINE + the tools the Master Agent calls to run it.

This is stage 2 of the pipeline (SOLUTION.md sections 4-5): campaign spec in, ranked
`ScreenCandidate[]` artifact out. It is **deterministic** — there is no LLM in this stage
at all. The Master Agent decides *when* to run it and passes the campaign's structured
inputs; every number below is computed in Python and SQL.

There is deliberately no Data Intelligence subagent. An LLM shell around a deterministic
engine bought nothing but latency and a chance to paraphrase numbers wrongly, so the
engine is a Master-owned tool instead (SOLUTION.md section 31.2: "LLMs reason; tools
calculate"). The pipeline is Master + two specialists (ML, OR).

Engine and tools live in one file by design decision — this is the whole audience
capability in one readable place, rather than a thin wrapper over a package.

================================ WHAT IT COMPUTES ================================
Feature layer (DuckDB, `app/data/db.py`):
    v_screen_profile          demographics + POI context + pool_key, one row per screen
    v_screen_demand_history   avg daily riders per screen x time block x day type

Audience scores, normalized 0-1 across the WHOLE inventory at build time so they stay
comparable between campaigns. Every weight set sums to 1.0 over bounded inputs, so each
score is inside the contract's 0-1 range by construction:
    income_score               min-max of zone income_index
    young_adult_score          min-max of pct_age_18_34
    middle_age_score           min-max of pct_age_35_54
    professional_score         0.6 income + 0.4 occupation_professional
    young_professionals_score  0.4 young_adult + 0.3 income + 0.3 occupation_professional
    student_score              0.4 young_adult + 0.3 university_nearby
                             + 0.3 occupation_student
    family_score               0.6 middle_age + 0.2 young_adult
                             + 0.2 occupation_family
    high_income_score          = income_score
    commuter_score             peak-block impressions / total impressions

`dominant_occupation` is graded rather than treated as a white-collar/not-white-collar
flag: it has five values and `mixed` is the most common, so a binary flag scored `mixed`
identically to `student`.

Relevance is a transparent weighted sum of five components (SOLUTION.md section 5):
    0.40 audience_similarity + 0.20 geographic_fit + 0.15 context_fit
  + 0.15 time_of_day_fit     + 0.10 historical_performance

`transit_score` is reported alongside but is NOT in that sum: it is the screen's audience
volume as a percentile of the eligible pool. Volume is the optimizer's objective quantity,
not a fit score — mixing it into relevance would rank a busy screen as a better *match*
than a quiet one in exactly the right zone.

============================== TWO UNITS TO GET RIGHT ==============================
A ROUTE'S RIDERS ARE SHARED BETWEEN ITS STOPS, NOT MULTIPLIED BY THEM. Each stop takes
`stop_share` of its route's ridership (`v_route_stop_weight`), so summing the stops along a
corridor comes to exactly 1.00x that corridor's ridership. Crediting every stop with the
whole route — which is what this did originally — made it 20.4x (median), and a rider
cannot be 20 different people.

A POOL IS A SITE, NOT A LOCATION ROW. `pool_key` groups on (city, name, serving corridors):
910 stop-mounted `location_id`s are 878 physical sites, because one station is modelled as
several location rows and screens on two platforms see the same crowd. 972 pools in total —
878 sites + 94 corridors.

================================ KNOWN LIMITATIONS ================================
Stated here because they change how the output should be read:

1.  BLOCK 1 IS A MODELLING GAP, NOT A FINDING. Volume is schedule-derived only, with no
    ambient/pedestrian term, and no scheduled service starts between 00:00 and 04:00 — so
    measured block-1 volume is exactly zero for all 11,163 screens. But 8,544 of 191,110
    real bookings (4.5%) sit in block 1, so the inventory demonstrably sells. Zero means
    "not modelled", never "nobody there". `impressions_block_1_estimated` publishes an
    8%-of-block-6 ASSUMPTION separately, and it is excluded from every total, from
    off-peak, and from `commuter_score`, so no validated figure moves with it.

2.  FIXED AND MOBILE SCREENS ARE ~2.7x APART on median daily volume (12,775 vs 4,789), and
    `metro_station` is ~59x `bus_stop` (14,873 vs 250). The gap was 40.9x fixed:mobile
    before stop shares. The mobile figure is one vehicle's share of its corridor (corridor
    block total / vehicles on corridor) — a stated modelling judgement, since
    `route_schedules` carries no vehicle_id. The divisor is published as
    `pool_partition_count` so the optimizer can recover the corridor's whole crowd for the
    reach ceiling. Note the corridor side is built from SCHEDULED estimates while the stop
    side is built from OBSERVED actuals; `corridor_pool_sanity()` checks the two paths stay
    consistent.

3.  MOBILE SCREENS HAVE NO DEMOGRAPHICS. Zone is undefined for a vehicle by construction,
    so `income_score`, `professional_score`, `young_professionals_score`, `student_score`,
    `family_score` and `high_income_score` are 0 for all 2,615 of them and only
    `commuter_score` carries signal. A structural floor, not a measurement. Averaging the
    zones a corridor touches would fix this and is the obvious next improvement — it is
    deliberately not smuggled into the occupation map.

4.  `historical_performance` IS A COMPLETION RATE, not campaign performance. It measures
    whether past bookings on this screen in this vertical were fulfilled, and falls back to
    0.5 for screens with no history in the vertical.

5.  A NARROW SUB-SCORE DECIDES LITTLE. `commuter_score` spans a narrow range, so on a
    commuter-only brief the 0.40-weighted audience term is close to constant and the
    ranking is effectively decided by the other four components. The extreme case — a
    sub-score identical across the whole pool — is reported as `constant_subscores`,
    because that state used to look exactly like success.

6.  THE POOL IS A TRUNCATION, AND TRUNCATION CAN BE CATEGORICAL. `top_n` keeps the best 250
    on relevance. A whole screen type can sit below another type's floor: `bus_stop`
    averages 0.5891 against `metro_station`'s 0.6066 — 2.9% — and still landed 0 of 250, so
    a 3% scoring difference produced a 100%/0% outcome and permitting a type via
    `allowed_screen_types` changed nothing. `CampaignSpec.screen_type_mix` stratifies the
    cut per requested type; without it the cut stays global, and the achieved composition is
    reported either way.

7.  NO HELD-OUT ACCURACY METRIC. The model is an aggregation of observed ridership, not a
    fitted predictor, so it has no error bar and no stage emits a per-screen confidence.

Port note: the feature and scoring logic began as a port of the audience relevance
notebook, verified output-identical at the time on all 12 impression columns, POI footfall
and POI counts across 11,163 screens. It has since diverged deliberately, and each
departure is flagged at its site: stop-share weighting, site-level pooling, graded
occupation affinity, mobile POI non-applicability, size/footfall tie-breaks, the separate
block-1 estimate, and stratified truncation. `family_score` no longer needs a
renormalization constant — its weights sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from app.config import get_settings
from app.data.db import has_ridership_actuals, query_df
from app.data.reference import corridor_zones, eligible_screen_ids, screen_facts
from app.logging_utils import debug, error, info
from app.models.campaign import (
    AUDIENCE_TERMS,
    INDUSTRY_VERTICALS,
    SCREEN_TYPES,
    CampaignSpec,
)
from app.models.screens import ScreenCandidate
from app.services import run_state
from app.services.artifact_store import write_records

ARTIFACT_KIND = "screen_candidates"

# A time block is a 4-hour window of the day. Within it all 6 rotation slots cycle
# continuously (1->2->...->6->1), and the same structure repeats every day of the flight.
# Slot POSITION is meaningless: holding k slots means appearing on k of every 6 loop passes,
# so exposures are LINEAR in k. Turning that share of voice into an exposure count is
# `app/optimize/exposure.py`'s job, not this stage's — everything here is PEOPLE PASSING.
SLOTS_PER_BLOCK = 6

ALL_TIME_BLOCKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
ALL_DAY_TYPES: tuple[str, ...] = ("weekday", "weekend")

# Blocks 2 (04:00-08:00) and 5 (16:00-20:00) are the commute peaks.
PEAK_BLOCKS: tuple[int, ...] = (2, 5)

# Vehicle-mounted screen types. They have no fixed location, hence no POI context and no
# zone demographics — both structural, neither a data-quality problem.
MOBILE_SCREEN_TYPES: tuple[str, ...] = ("bus", "metro_rail_coach")

# Physical canvas size, used ONLY to break relevance ties (see `score`). Not a relevance
# component and deliberately not in WEIGHTS: size is a property of the asset, not a measure
# of how well its audience matches the brief.
SCREEN_SIZE_RANK: dict[str, float] = {"S": 0.0, "M": 0.5, "L": 1.0}

# Time block 1 is 00:00-04:00 and no scheduled service starts in it, so the volume model
# reports exactly zero for all 11,163 screens — while `bookings` holds 8,544 real block-1
# bookings. This is the ASSUMED share of block 6 (20:00-24:00) volume that block 1 probably
# carries, used only to publish a separate estimate. It is a judgement with no measurement
# behind it, so it stays OUT of total/peak/offpeak impressions and out of commuter_score:
# nothing the validator checks may move with an assumed constant (CLAUDE.md 31), and
# commuter_score's denominator is total_impressions.
BLOCK_1_ESTIMATE_SHARE_OF_BLOCK_6 = 0.08
ESTIMATED_BLOCK = 1
REFERENCE_BLOCK_FOR_ESTIMATE = 6

# `zone_demographics.dominant_occupation` has exactly five values across the 30 zones:
# mixed (14), white_collar (7), blue_collar (3), retail_service (3), student (3). These
# grade each one's affinity with an audience instead of the old binary white-collar flag,
# which treated 'mixed' — the most common value by far — as identical to 'student'.
#
# All three maps cover all five values, so a NaN out of them means one of exactly two
# things and they must not be conflated: a NEW category nobody mapped (schema drift, warn
# loudly), or a mobile screen with no zone at all (structural, expected for all 2,615 of
# them — see KNOWN LIMITATIONS #3).
OCCUPATION_PROFESSIONAL_AFFINITY: dict[str, float] = {
    "white_collar": 1.0,
    "mixed": 0.5,
    "retail_service": 0.3,
    "blue_collar": 0.2,
    "student": 0.0,
}
OCCUPATION_STUDENT_AFFINITY: dict[str, float] = {
    "student": 1.0,
    "mixed": 0.5,
    "retail_service": 0.3,
    "white_collar": 0.0,
    "blue_collar": 0.0,
}
OCCUPATION_FAMILY_AFFINITY: dict[str, float] = {
    "blue_collar": 0.7,
    "retail_service": 0.6,
    "white_collar": 0.5,
    "mixed": 0.5,
    "student": 0.0,
}

WEIGHTS: dict[str, float] = {
    "audience_similarity": 0.40,
    "geographic_fit": 0.20,
    "context_fit": 0.15,
    "time_of_day_fit": 0.15,
    "historical_performance": 0.10,
}

AUDIENCE_TERM_TO_SCORE_COLUMN: dict[str, list[str]] = {
    # Each term now resolves to the column that actually measures it. Two were wrong:
    # `young_professionals` averaged professional_score with STUDENT_SCORE, and
    # `high_income` was scored on professional_score, which is 40% occupation.
    "young_professionals": ["young_professionals_score"],
    "professionals": ["professional_score"],
    "students": ["student_score"],
    "families": ["family_score"],
    "high_income": ["high_income_score"],
    "commuters": ["commuter_score"],
}

# Which time blocks each audience is actually out and about in. Published on the artifact
# so the pricing stage prices the blocks the campaign wants, rather than each stage
# reimplementing this table and drifting apart.
AUDIENCE_TO_PREFERRED_BLOCKS: dict[str, list[int]] = {
    "young_professionals": [2, 5],
    "professionals": [2, 5],
    "commuters": [2, 5],
    "students": [2, 3, 4],
    "families": [4, 5],
    "high_income": [2, 5],
}

INDUSTRY_TO_POI_CONTEXT: dict[str, list[str]] = {
    "entertainment": ["stadium_arena", "entertainment_district", "museum", "tourist_landmark"],
    "healthcare": ["hospital"],
    "finance": ["office_park", "corporate_campus"],
    "auto": ["shopping_mall", "corporate_campus"],
    "telecom": ["office_park", "corporate_campus", "shopping_mall"],
    "real_estate": ["office_park", "residential_tower", "corporate_campus"],
    "government": ["government_building"],
    "education": ["university"],
    "hospitality": ["hotel_convention", "tourist_landmark", "museum"],
    "cpg": ["shopping_mall", "grocery_anchor"],
    "retail": ["shopping_mall", "grocery_anchor"],
    "nonprofit": ["government_building", "university"],
    "technology": ["office_park", "corporate_campus"],
}

NEUTRAL = 0.5
"""Score used when an input is missing. Always recorded in `defaults_applied`."""

KNOWN_LIMITATIONS: tuple[str, ...] = (
    (
        "Volume is schedule-derived only; there is no ambient/pedestrian term. Block 1 "
        "(00:00-04:00) has no scheduled service, so its MEASURED volume is zero for every "
        "screen — while 8,544 of 191,110 real bookings (4.5%) sit in block 1, so the "
        "inventory demonstrably sells. This is a modelling GAP, not a finding about "
        "block 1. `impressions_block_1_estimated` publishes an 8%-of-block-6 assumption "
        "separately; it is excluded from every total and from commuter_score."
    ),
    (
        "Fixed screens carry ~2.7x the median daily volume of mobile ones (12,775 vs "
        "4,789), and metro_station ~59x bus_stop. Volume-per-dollar rankings still lean "
        "fixed, though far less than before stop shares were applied — the gap was 40.9x "
        "fixed:mobile when each route's WHOLE ridership was credited at every one of its "
        "stops."
    ),
    (
        "Mobile screens have no zone demographics. Zone is undefined for a vehicle, so "
        "income, professional, young-professional, student and family scores are all 0 for "
        "all 2,615 of them and only commuter_score carries signal. A structural floor, not "
        "a measurement. Averaging the zones a corridor touches would fix it."
    ),
    "historical_performance is a booking completion rate, not campaign performance.",
    (
        "commuter_score spans a narrow range, so on a commuter-only brief the "
        "0.40-weighted audience term is close to constant and the ranking is effectively "
        "decided by the other four components. `constant_subscores` on the artifact "
        "summary reports the extreme case, where a sub-score is identical pool-wide."
    ),
    (
        "No held-out accuracy metric: the audience model is an aggregation of observed "
        "ridership rather than a fitted predictor, so it has no error bar and no stage "
        "emits a per-screen confidence."
    ),
    (
        "The candidate pool is truncated to top_n on relevance. Because a screen type can "
        "sit categorically below another (bus_stop averages 0.5891 against "
        "metro_station's 0.6066 and still landed 0 of 250), a single global cut can return "
        "one screen type only. `screen_type_mix` on the spec stratifies the cut per "
        "requested type; without it the cut stays global."
    ),
    (
        "nearby_ambient_footfall is a POI proxy that correlates only weakly with transit "
        "ridership (~0.12-0.26) and can disagree ~20x at one location. It is reported and "
        "used ONLY as a tie-break — never added into any impressions, reach or price."
    ),
)
"""Surfaced by `describe_relevance_model` so the Master Agent states limits it can cite."""


def block_column(block: int, day_type: str) -> str:
    return f"impressions_block_{block}_{day_type}"


BLOCK_COLUMNS: tuple[str, ...] = tuple(
    block_column(b, dt) for b in ALL_TIME_BLOCKS for dt in ALL_DAY_TYPES
)


# =============================================================================
# FEATURE LAYER
# =============================================================================


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max scale to 0-1. For scores where "biggest is best" is a fair comparison.

    Computed once over the whole inventory at build time, never per campaign, so a
    screen's audience score does not shift when the geography filter changes.
    """
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(NEUTRAL, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def _allocate_quotas(available: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` slots between screen types, capped at each type's real availability.

    Equal shares first, then repeatedly redistribute what a scarce type could not use. A
    type with 12 screens gets 12, not 125, and the 113 it cannot fill go to the types that
    can — so a mix request never wastes pool capacity on inventory that does not exist.
    """
    quotas = {t: 0 for t in available}
    remaining = min(total, sum(available.values()))
    open_types = [t for t in available if available[t] > 0]
    while remaining > 0 and open_types:
        share = max(remaining // len(open_types), 1)
        for screen_type in list(open_types):
            if remaining <= 0:
                break
            grant = min(share, available[screen_type] - quotas[screen_type], remaining)
            quotas[screen_type] += grant
            remaining -= grant
            if quotas[screen_type] >= available[screen_type]:
                open_types.remove(screen_type)
    return quotas


def _occupation_affinity(df: pd.DataFrame, affinity: dict[str, float]) -> pd.Series:
    """Map `dominant_occupation` onto a 0-1 affinity, distinguishing the two NaN causes.

    A zone whose `dominant_occupation` is a value NOBODY MAPPED is schema drift: someone
    changed the data and the map needs a decision. That warns, loudly, once per unmapped
    value.

    A NULL because the screen is VEHICLE-MOUNTED is not drift. Zone is undefined for a
    vehicle by construction, so all 2,615 mobile screens land here on every run. It is the
    documented mobile-demographics floor (KNOWN LIMITATIONS #3), reported per candidate row
    by `reasons_for`, and warning per row would bury the drift signal above in noise.

    Both resolve to 0.0. Averaging the zones a corridor touches is the real fix for the
    mobile case, and it is deliberately NOT done here — it is a change to what mobile
    demographics MEAN, not a detail of the occupation map.
    """
    occupation = df["dominant_occupation"]
    unmapped = sorted(
        {str(v) for v in occupation.dropna().unique() if str(v) not in affinity and str(v) != ""}
    )
    if unmapped:
        error(
            f"dominant_occupation value(s) {unmapped} are not in the affinity map — those "
            f"zones score 0 on this component. This is schema drift, not the mobile floor: "
            f"add them to the OCCUPATION_*_AFFINITY maps in relevance_tools.py."
        )
    return occupation.map(affinity).astype(float).fillna(0.0)


def _impressions_wide(demand: pd.DataFrame) -> pd.DataFrame:
    """Long screen x block x day_type demand -> one row per screen, 12 columns."""
    wide = demand.pivot_table(
        index="screen_id",
        columns=["time_block_id", "day_type"],
        values="daily_impressions",
        aggfunc="sum",
    )
    wide.columns = [block_column(int(b), str(dt)) for b, dt in wide.columns]
    return wide.reset_index()


def _build_profile() -> pd.DataFrame:
    """One row per screen: geography, demographics, POI context, 12 impression columns
    and the normalized audience scores."""
    profile = query_df("SELECT * FROM v_screen_profile")
    demand = query_df("SELECT * FROM v_screen_demand_history")

    df = profile.merge(_impressions_wide(demand), on="screen_id", how="left")

    # Guarantee all 12 block x day_type columns exist. A combination that never appears in
    # the demand view (block 1, always) must still be present and zero, so every downstream
    # sum has the same shape.
    for col in BLOCK_COLUMNS:
        df[col] = df[col].fillna(0.0) if col in df.columns else 0.0

    # POI types arrive as a DuckDB list; a set makes the context check a cheap intersection.
    df["poi_type_set"] = [
        set(v) if v is not None and len(v) else set() for v in df["nearby_poi_types"]
    ]
    df["has_university_nearby"] = df["poi_type_set"].map(lambda s: "university" in s)

    # Whether a POI context check MEANS anything for this screen. `v_screen_poi` joins POIs
    # on `anchor_location_id = location_id`, and a vehicle has no location_id — so every
    # mobile screen has an empty `poi_type_set` for an ARCHITECTURAL reason, not because
    # its surroundings were checked and found to be a poor match. Scoring them against the
    # POI set anyway handed all 2,615 the 0.2 mismatch penalty, which is a measurement the
    # data cannot support. Keyed on screen_type per the four real values; equivalent to
    # `location_id IS NULL` on this inventory, and stated in the terms a brief uses.
    df["poi_applicable"] = ~df["screen_type"].isin(MOBILE_SCREEN_TYPES)

    # --- audience scores -----------------------------------------------------
    # Every weight set below sums to 1.0 over inputs already bounded 0-1, so each score is
    # inside the ScreenCandidate contract's 0-1 range by construction rather than by a
    # renormalization constant. `family_score` used to need dividing by a 1.25 ceiling
    # (it measured 1.140 and broke the bound); it no longer does.
    df["income_score"] = _normalize(df["income_index"].fillna(0.0))
    df["young_adult_score"] = _normalize(df["pct_age_18_34"].fillna(0.0))
    df["middle_age_score"] = _normalize(df["pct_age_35_54"].fillna(0.0))

    occ_professional = _occupation_affinity(df, OCCUPATION_PROFESSIONAL_AFFINITY)
    occ_student = _occupation_affinity(df, OCCUPATION_STUDENT_AFFINITY)
    occ_family = _occupation_affinity(df, OCCUPATION_FAMILY_AFFINITY)
    university = df["has_university_nearby"].astype(float)

    df["professional_score"] = (0.6 * df["income_score"] + 0.4 * occ_professional).clip(0.0, 1.0)
    # Distinct from professional_score: a young-professionals brief wants the AGE as well
    # as the income and the occupation. It used to average professional_score with
    # student_score, which is a different audience wearing the same label.
    df["young_professionals_score"] = (
        0.4 * df["young_adult_score"] + 0.3 * df["income_score"] + 0.3 * occ_professional
    ).clip(0.0, 1.0)
    df["student_score"] = (
        0.4 * df["young_adult_score"] + 0.3 * university + 0.3 * occ_student
    ).clip(0.0, 1.0)
    df["family_score"] = (
        0.6 * df["middle_age_score"] + 0.2 * df["young_adult_score"] + 0.2 * occ_family
    ).clip(0.0, 1.0)
    # A distinct column rather than an alias of professional_score, so a high_income brief
    # is scored on income alone and not partly on occupation.
    df["high_income_score"] = df["income_score"]

    # --- impression rollups --------------------------------------------------
    # The 12 granular columns stay the source of truth; these are conveniences. Every
    # rollup here is over MEASURED columns only — the block-1 estimate below is excluded
    # from all of them on purpose.
    for dt in ALL_DAY_TYPES:
        df[f"impressions_{dt}"] = sum(df[block_column(b, dt)] for b in ALL_TIME_BLOCKS)

    offpeak_blocks = [b for b in ALL_TIME_BLOCKS if b not in PEAK_BLOCKS]
    df["peak_impressions"] = sum(
        df[block_column(b, dt)] for b in PEAK_BLOCKS for dt in ALL_DAY_TYPES
    )
    df["offpeak_impressions"] = sum(
        df[block_column(b, dt)] for b in offpeak_blocks for dt in ALL_DAY_TYPES
    )
    # Defined as peak + offpeak rather than as its own 12-column sum, so
    # `peak + offpeak == total` holds EXACTLY rather than to within floating-point
    # addition order. A test asserts it.
    df["total_impressions"] = df["peak_impressions"] + df["offpeak_impressions"]

    df["commuter_score"] = (
        df["peak_impressions"] / df["total_impressions"].replace(0.0, 1.0)
    ).clip(0.0, 1.0)

    # Block 1's SEPARATE estimate. Deliberately not written into
    # `impressions_block_1_{day_type}`, which stays at its measured zero: overwriting the
    # measurement would make an assumption indistinguishable from data everywhere
    # downstream. Per day type, because block 6 traffic differs between weekday and
    # weekend and a single scalar could not preserve that.
    for dt in ALL_DAY_TYPES:
        df[f"block_{ESTIMATED_BLOCK}_estimated_{dt}"] = (
            BLOCK_1_ESTIMATE_SHARE_OF_BLOCK_6 * df[block_column(REFERENCE_BLOCK_FOR_ESTIMATE, dt)]
        )

    return df


def _build_booking_history() -> dict[tuple[str, str], tuple[float, int]]:
    """(screen_id, industry) -> (completion rate, bookings observed).

    Precomputed once. Scanning 191,109 bookings per screen inside a scoring loop was the
    single most expensive thing in the original notebook.
    """
    df = query_df(
        """
        SELECT screen_id,
               industry_vertical,
               avg(CASE WHEN booking_status = 'completed' THEN 1.0 ELSE 0.0 END) AS rate,
               count(*)                                                          AS n
        FROM bookings
        GROUP BY 1, 2
        """
    )
    return {
        (r.screen_id, r.industry_vertical): (float(r.rate), int(r.n))
        for r in df.itertuples(index=False)
    }


@dataclass
class ScoringResult:
    """One scoring pass. A named struct rather than a tuple because the caller needs the
    composition diagnostics as much as the rows: a pool that is 100% one screen type is
    the most consequential fact about it, and a tuple made that easy to drop."""

    candidates: pd.DataFrame
    target_blocks: list[int]
    notes: list[str]
    eligible_by_screen_type: dict[str, int]
    pool_by_screen_type: dict[str, int]
    mix_unfilled: dict[str, str]


@dataclass
class RelevanceEngine:
    """Deterministic audience + relevance model. Process singleton; build costs ~15s."""

    profile: pd.DataFrame
    booking_history: dict[tuple[str, str], tuple[float, int]]
    demand_source: str
    pool_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def screens(self) -> int:
        return len(self.profile)

    def preferred_blocks(self, audience_terms: list[str]) -> list[int]:
        """Blocks this campaign's audience is active in — a campaign-level property.

        Returns the full set rather than a single winner: a professionals campaign
        genuinely wants both commute peaks, and choosing between them is a budget decision
        for the optimizer, not something this engine should decide silently.
        """
        blocks: set[int] = set()
        for term in audience_terms:
            blocks.update(AUDIENCE_TO_PREFERRED_BLOCKS.get(term, []))
        return sorted(blocks)

    # --- scoring ----------------------------------------------------------------

    def score(self, spec: CampaignSpec, eligible: set[str], top_n: int) -> ScoringResult:
        """Rank the eligible inventory.

        Vectorized: every sub-score is computed as a column over the filtered frame, not
        row by row.
        """
        df = self.profile[self.profile["screen_id"].isin(eligible)].copy()
        df = self._apply_exclusions(df, spec)
        if df.empty:
            return ScoringResult(
                candidates=df,
                target_blocks=self.preferred_blocks(spec.audience_terms),
                notes=[],
                eligible_by_screen_type={},
                pool_by_screen_type={},
                mix_unfilled={},
            )

        blocks = self.preferred_blocks(spec.audience_terms)
        day_types = (
            [spec.day_type_focus] if spec.day_type_focus in ALL_DAY_TYPES else list(ALL_DAY_TYPES)
        )

        audience, audience_note = self._audience_similarity(df, spec.audience_terms)
        geography = self._geographic_fit(df, spec)
        context, context_note = self._context_fit(df, spec.industry_vertical)
        time_of_day, tod_note = self._time_of_day_fit(df, blocks, day_types)
        history, history_note = self._historical_performance(df, spec.industry_vertical)

        df["audience_similarity"] = audience
        df["geographic_fit"] = geography
        df["context_fit"] = context
        df["time_of_day_fit"] = time_of_day
        df["historical_performance"] = history
        df["relevance_score"] = sum(df[name] * w for name, w in WEIGHTS.items())

        # Addressable volume: the blocks this campaign actually wants, not all 24 hours.
        cols = [block_column(b, dt) for b in (blocks or ALL_TIME_BLOCKS) for dt in day_types]
        df["addressable_impressions"] = df[cols].sum(axis=1)
        # Percentile rather than min-max: volume still spans ~59x between screen types
        # (metro_station vs bus_stop), and a min-max would crush all but the busiest sites
        # to ~0. It spanned ~380x before stop shares, which is why this was never min-max.
        df["transit_score"] = df["addressable_impressions"].rank(pct=True).fillna(0.0)

        notes = [n for n in (audience_note, context_note, tod_note, history_note) if n]
        notes.extend(self._constant_subscore_notes(df))

        # Tie-breaking. Screens at ONE stop genuinely tie on relevance — same zone, same
        # POIs, same traffic — so the ordering among them was decided by screen_id alone,
        # and on the canonical brief 4 screens sat on the identical 0.7493. Physical size
        # breaks it first (a larger canvas is a real advantage at equal fit), then ambient
        # footfall (Fix 7's quarantined signal, used ONLY here), then screen_id.
        #
        # `screen_id` MUST stay the final key: it is what makes the artifact reproducible.
        # Applied at SORT TIME only — neither term is in WEIGHTS, because neither is a
        # measure of audience fit. `position` is deliberately absent: nothing in this data
        # says any mounting position outperforms another.
        df["_size_rank"] = df["screen_size"].map(SCREEN_SIZE_RANK).astype(float).fillna(0.0)
        ranked = df.sort_values(
            ["relevance_score", "_size_rank", "weighted_nearby_footfall", "screen_id"],
            ascending=[False, False, False, True],
        )
        kept = self._stratified_head(ranked, spec, top_n)

        eligible_by_type = ranked["screen_type"].value_counts().to_dict()
        pool_by_type = kept["screen_type"].value_counts().to_dict()
        unfilled = {
            t: (
                f"requested in screen_type_mix but {eligible_by_type.get(t, 0)} eligible "
                f"screens of this type survived hard filtering, so the pool holds "
                f"{pool_by_type.get(t, 0)}"
            )
            for t in spec.screen_type_mix
            if not pool_by_type.get(t)
        }
        if unfilled:
            info(f"STAGE 2-3 requested screen types not represented in the pool: {unfilled}")

        return ScoringResult(
            candidates=kept.reset_index(drop=True),
            target_blocks=blocks,
            notes=notes,
            eligible_by_screen_type={str(k): int(v) for k, v in eligible_by_type.items()},
            pool_by_screen_type={str(k): int(v) for k, v in pool_by_type.items()},
            mix_unfilled=unfilled,
        )

    def _constant_subscore_notes(self, df: pd.DataFrame) -> list[str]:
        """Flag any sub-score that is identical across the WHOLE pool.

        A constant sub-score contributes nothing but its weight, so the effective model is
        smaller than the published one — and every path to that state looked like success.
        An unmatched `industry_vertical` used to pin `context_fit` AND
        `historical_performance` (0.25 of the weight) to 0.5 while the tool reported a
        normal-looking ranking. Loud beats correct-on-average here.
        """
        notes: list[str] = []
        for name, weight in WEIGHTS.items():
            values = df[name]
            if len(values) > 1 and values.nunique(dropna=False) == 1:
                notes.append(
                    f"{name} is CONSTANT at {float(values.iloc[0]):.3f} across all "
                    f"{len(values)} scored screens, so its {weight:.0%} weight does not "
                    f"affect the ranking — the effective model is "
                    f"{1.0 - weight:.0%} of the published one for this brief."
                )
        return notes

    def _stratified_head(
        self, ranked: pd.DataFrame, spec: CampaignSpec, top_n: int
    ) -> pd.DataFrame:
        """Truncate to `top_n`, allocating per requested screen type when a mix is asked for.

        THE BUG THIS FIXES. A brief wanting metro stations AND bus screens could not be
        served. `hard_constraints["allowed_screen_types"]` is a FILTER, not a mix:
        permitting both let 4,629 screens through, and a single global relevance cut kept
        250 — 100% of them metro_station, 0 bus. Zero bus screens reached pricing, let
        alone the optimizer.

        It is not that bus screens are bad. Exclusion is CATEGORICAL rather than marginal:
        bus's best score sits below metro's pool minimum, and in the sharpest case
        `bus_stop` averages 0.5891 against metro_station's 0.6066 — a 2.9% gap — and still
        landed 0 of 250. A 3% scoring difference produced a 100%/0% outcome.

        So a requested mix allocates the budget of `top_n` per named type, capped at what
        each type actually has, redistributing any shortfall to the other requested types.

        This is NOT a new per-screen recommendation — `head(top_n)` was already a decision
        about which screens are handed on, and this changes only how that existing
        truncation is made. No screen is picked for the optimizer and no "recommended type"
        column is added.

        With no mix requested the behaviour is unchanged: one global cut. Giving every
        available type a small floor unconditionally is the obvious extension and is
        deliberately NOT done here — it changes what a pool means on every brief ever run,
        which is a decision to take explicitly rather than as a side effect.
        """
        requested = [t for t in spec.screen_type_mix if t in set(ranked["screen_type"])]
        if not spec.screen_type_mix:
            return ranked.head(top_n)

        available = {t: int((ranked["screen_type"] == t).sum()) for t in requested}
        quotas = _allocate_quotas(available, top_n)

        kept = pd.concat(
            [ranked[ranked["screen_type"] == t].head(quotas[t]) for t in requested if quotas.get(t)]
        )
        # Any remainder that no requested type could absorb goes back to the global
        # ranking, so a mix request never returns FEWER candidates than a plain cut would.
        shortfall = top_n - len(kept)
        if shortfall > 0:
            rest = ranked[~ranked.index.isin(kept.index)].head(shortfall)
            kept = pd.concat([kept, rest])
        # Restore relevance order across the strata: the pool is documented as ranked
        # best-first and the optimizer's diagnostics assume it. Reindexing against
        # `ranked.index` rather than re-sorting keeps the full tie-break chain intact.
        keep = set(kept.index)
        return kept.reindex([i for i in ranked.index if i in keep])

    def _apply_exclusions(self, df: pd.DataFrame, spec: CampaignSpec) -> pd.DataFrame:
        """Hard cuts from `hard_constraints`, applied before scoring.

        An excluded screen must never appear in the pool, not merely rank lower.
        """
        hc = spec.hard_constraints
        if allowed := hc.get("allowed_screen_types"):
            df = df[df["screen_type"].isin(list(allowed))]
        if excluded := hc.get("excluded_screen_types"):
            df = df[~df["screen_type"].isin(list(excluded))]
        if excluded := hc.get("excluded_zone_ids"):
            df = df[~df["zone_id"].isin(list(excluded))]
        if excluded := hc.get("excluded_positions"):
            df = df[~df["position"].isin(list(excluded))]
        return df

    def _audience_similarity(
        self, df: pd.DataFrame, terms: list[str]
    ) -> tuple[pd.Series, str | None]:
        if not terms:
            return pd.Series(NEUTRAL, index=df.index), (
                "no audience_terms on the spec — audience_similarity defaulted to 0.5 for "
                "every screen"
            )
        columns = [c for t in terms for c in AUDIENCE_TERM_TO_SCORE_COLUMN.get(t, [])]
        if not columns:
            return pd.Series(NEUTRAL, index=df.index), (
                f"audience_terms {terms} map to no score column — audience_similarity "
                f"defaulted to 0.5"
            )
        return df[columns].mean(axis=1).clip(0.0, 1.0), None

    def _geographic_fit(self, df: pd.DataFrame, spec: CampaignSpec) -> pd.Series:
        """Graded fit inside the already-eligible pool.

        Every row here passed the hard geography filter, so this grades *how well* it
        matches, not whether it is allowed:

            1.0  exact match on a requested zone or corridor, or a city-wide brief
            0.8  mobile screen whose corridor passes through a requested zone — it serves
                 the area without being sited in it
            0.6  right city, but a finer geography was requested and this is not in it
            0.0  should be unreachable after the hard filter; kept as a guard
        """
        wanted_zones = set(spec.zone_ids)
        wanted_corridors = set(spec.corridor_ids)
        wanted_cities = set(spec.city_ids)
        czones = corridor_zones()

        in_city = df["city_id"].isin(wanted_cities)
        exact = df["zone_id"].isin(wanted_zones) | df["corridor_id"].isin(wanted_corridors)
        # corridor_id is NaN for fixed screens, and bool(nan) is True — test the type.
        touches = df["corridor_id"].map(
            lambda c: isinstance(c, str) and bool(czones.get(c, set()) & wanted_zones)
        )

        if not wanted_zones and not wanted_corridors:
            # A city-wide brief asked for the whole city; everything in it is an exact fit.
            return in_city.astype(float)

        score = pd.Series(0.0, index=df.index)
        score = score.where(~in_city, 0.6)
        score = score.where(~touches, 0.8)
        return score.where(~exact, 1.0)

    def _context_fit(self, df: pd.DataFrame, industry: str | None) -> tuple[pd.Series, str | None]:
        """Does this screen's POI surroundings suit the advertiser's industry?

        Mobile screens are excluded from the judgement entirely rather than scored 0.2 —
        see `poi_applicable` in `_build_profile`.
        """
        if not industry:
            return pd.Series(NEUTRAL, index=df.index), (
                "spec has no industry_vertical — context_fit defaulted to 0.5"
            )
        wanted = INDUSTRY_TO_POI_CONTEXT.get(industry)
        if wanted is None:
            # Unreachable while CampaignSpec validates industry_vertical against the closed
            # INDUSTRY_VERTICALS list and a test pins that list against this map's keys.
            # Kept as a guard because the alternative is a silent 0.5 across the pool.
            return pd.Series(NEUTRAL, index=df.index), (
                f"industry_vertical '{industry}' has no POI mapping — context_fit defaulted to 0.5"
            )
        target = set(wanted)
        scored = df["poi_type_set"].map(lambda s: 1.0 if s & target else 0.2).astype(float)
        return scored.where(df["poi_applicable"], NEUTRAL), None

    def _time_of_day_fit(
        self, df: pd.DataFrame, blocks: list[int], day_types: list[str]
    ) -> tuple[pd.Series, str | None]:
        """Fraction of a screen's traffic falling in the campaign's target blocks.

        Uses block IDs, not daypart names: 'night' maps to two non-adjacent blocks (1 and
        6), so the name is ambiguous and the ID is not.
        """
        if not blocks:
            return pd.Series(NEUTRAL, index=df.index), (
                "audience_terms imply no preferred time blocks — time_of_day_fit defaulted to 0.5"
            )
        wanted = [block_column(b, dt) for b in blocks for dt in day_types]
        every = [block_column(b, dt) for b in ALL_TIME_BLOCKS for dt in day_types]
        numerator = df[wanted].sum(axis=1)
        denominator = df[every].sum(axis=1).replace(0.0, 1.0)
        return (numerator / denominator).clip(0.0, 1.0), None

    def _historical_performance(
        self, df: pd.DataFrame, industry: str | None
    ) -> tuple[pd.Series, str | None]:
        if not industry:
            return pd.Series(NEUTRAL, index=df.index), (
                "spec has no industry_vertical — historical_performance defaulted to 0.5"
            )
        lookup = self.booking_history
        values = [lookup.get((sid, industry), (NEUTRAL, 0))[0] for sid in df["screen_id"]]
        return pd.Series(values, index=df.index).clip(0.0, 1.0), None

    # --- explanation -------------------------------------------------------------

    def reasons_for(
        self, row: pd.Series, spec: CampaignSpec, blocks: list[int], day_types: list[str]
    ) -> tuple[list[str], list[str]]:
        """Feature-citing reasons + the defaults that fired, for one candidate row.

        SOLUTION.md section 25 forbids generic text: every line names a real feature value.
        Only run for the rows actually kept, so the per-row cost is bounded by top_n.
        """
        reasons: list[str] = []
        defaults: list[str] = []

        zone = _or_none(row.get("zone_name")) or _or_none(row.get("city_zone"))
        if zone and pd.notna(row.get("pct_age_18_34")):
            reasons.append(
                f"Zone {zone}: {row['pct_age_18_34']:.1f}% aged 18-34, "
                f"{row['pct_age_35_54']:.1f}% aged 35-54, income index "
                f"{row['income_index']:.1f}, {row['dominant_occupation']} dominant"
            )
        elif row.get("inventory_class") == "mobile":
            defaults.append(
                f"Vehicle-mounted screen on corridor {row.get('corridor_id')}: no zone "
                f"demographics exist for mobile inventory, so its demographic sub-scores "
                f"are a floor of 0, not a measurement"
            )

        if not row.get("poi_applicable", True):
            defaults.append(
                "context_fit not applicable — mobile screen has no fixed location, so no "
                "POI context exists to match against the advertiser's industry. Scored a "
                "neutral 0.5 rather than the 0.2 mismatch penalty."
            )
        elif row.get("num_nearby_pois"):
            types = ", ".join(sorted(row["poi_type_set"])[:4])
            reasons.append(
                f"{int(row['num_nearby_pois'])} POI(s) anchored here ({types}); closest "
                f"{row['closest_poi_distance_km']:.2f} km, distance-weighted footfall "
                f"{row['weighted_nearby_footfall']:,.0f}/day"
            )

        if blocks:
            addressable = row["addressable_impressions"]
            reasons.append(
                f"{addressable:,.0f} riders/day in target block(s) "
                f"{blocks} ({', '.join(day_types)}) — {row['time_of_day_fit']:.0%} of this "
                f"screen's {row['total_impressions']:,.0f} daily traffic, "
                f"{row['transit_score']:.0%} percentile volume in the eligible pool"
            )

        geo = row["geographic_fit"]
        if geo == 1.0:
            reasons.append(
                f"Inside the requested geography ({row.get('zone_id') or row.get('corridor_id')})"
            )
        elif geo == 0.8:
            reasons.append(
                f"Corridor {row.get('corridor_id')} passes through the requested zone(s) "
                f"without being sited in one"
            )
        elif geo == 0.6:
            reasons.append(
                f"In requested city {row.get('city_id')} but outside the requested zone/corridor"
            )

        if spec.industry_vertical:
            entry = self.booking_history.get((row["screen_id"], spec.industry_vertical))
            if entry is not None:
                rate, n = entry
                reasons.append(
                    f"{rate:.0%} of {n} past {spec.industry_vertical} booking(s) on this "
                    f"screen completed"
                )
            else:
                defaults.append(
                    f"No {spec.industry_vertical} booking history on this screen — "
                    f"historical_performance defaulted to 0.5"
                )

        pool_size = self.pool_sizes.get(row.get("pool_key"), 1)
        if pool_size > 1:
            # Names the site, not the synthetic pool_key — the key is an internal handle
            # and the agent prompt forbids putting it in an answer.
            site = _or_none(row.get("location_name")) or _or_none(row.get("corridor_id"))
            reasons.append(
                f"Shares its audience with {pool_size - 1} other screen(s) at "
                f"{site or 'the same site'} — their impressions must be deduplicated, "
                f"not summed"
            )

        return reasons, defaults


@lru_cache(maxsize=1)
def corridor_pool_sanity() -> dict[str, Any]:
    """Assert a corridor's whole crowd is never smaller than one station's crowd on it.

    A corridor's pool population is reconstructed downstream as one vehicle's share x
    `pool_partition_count` (`app/optimize/contract.py`), a round trip across two modules
    that this engine divided in `v_corridor_block_demand`. Nothing checked that the result
    stays consistent with the stop-mounted figures built from the SAME schedules.

    The invariant: for every (corridor, block, day type), the corridor's reconstructed pool
    population is at least the largest audience any single station on it draws FROM THAT
    CORRIDOR'S OWN ROUTES.

    That last restriction is the whole subtlety, and getting it wrong makes the check cry
    wolf. A station's total audience sums every route serving it, including routes on other
    corridors — so comparing a station's TOTAL against ONE corridor's total flags 40 cells
    that are not violations at all, just busy interchanges. Measured: corridor ACS-RT-B008
    carries 10 riders in block 2 on a weekday (2 departures), while a stop on it measures
    206.6 because a metro corridor also passes through. Both figures are correct.

    Flags rather than raises. A violation is real information about the data, and failing
    the whole stage over one corridor would withhold 11,162 correct screens.
    """
    df = query_df(
        """
        WITH corridor_total AS (
            SELECT cb.corridor_id, cb.time_block_id, cb.day_type,
                   cb.avg_daily_ridership * coalesce(v.n_vehicles, 1) AS pool_population
            FROM v_corridor_block_demand cb
            LEFT JOIN v_corridor_vehicle_count v USING (corridor_id)
        ),
        -- Each stop's audience from THIS corridor's routes only: the same
        -- ridership x stop_share product `v_screen_demand_history` uses, restricted to the
        -- routes that belong to the corridor.
        station AS (
            SELECT corridor_id, time_block_id, day_type,
                   max(riders) AS largest_station
            FROM (
                SELECT rs.corridor_id, w.location_id, rb.time_block_id, rb.day_type,
                       sum(rb.avg_daily_ridership * w.stop_share) AS riders
                FROM (SELECT DISTINCT route_id, corridor_id FROM route_stops) rs
                JOIN v_route_stop_weight w   ON w.route_id = rs.route_id
                JOIN v_route_block_demand rb ON rb.route_id = rs.route_id
                GROUP BY 1, 2, 3, 4
            )
            GROUP BY 1, 2, 3
        )
        SELECT c.corridor_id, c.time_block_id, c.day_type,
               c.pool_population, s.largest_station,
               s.largest_station / nullif(c.pool_population, 0) AS ratio
        FROM corridor_total c
        JOIN station s USING (corridor_id, time_block_id, day_type)
        WHERE s.largest_station > c.pool_population * 1.000001
        ORDER BY ratio DESC
        """
    )
    if not df.empty:
        error(
            f"corridor pool sanity: {len(df)} (corridor, block, day type) cell(s) where the "
            f"largest single station out-measures its whole corridor. The vehicle-division "
            f"round trip and the stop-share path disagree — investigate before trusting "
            f"mobile reach ceilings."
        )
    return {
        "invariant": (
            "corridor pool population >= the largest single-station audience drawn from "
            "that corridor's own routes. A station's riders on a corridor are a subset of "
            "that corridor's riders."
        ),
        "violations": len(df),
        "worst": (
            df.head(5).round(2).to_dict("records") if not df.empty else "none — the paths agree"
        ),
        "caveat": (
            "The two sides come from different sources and the check tolerates that: a "
            "corridor's pool population is built from route_schedules.estimated_ridership "
            "(scheduled) while a stop's audience comes from ridership_actuals (observed). "
            "Corridor totals reconcile to 1.000x within each source and to ~0.97x across "
            "them, so mobile pool ceilings are scheduled-based and fixed ones are "
            "observation-based. Not corrected here — it would change the mobile unit."
        ),
    }


@lru_cache(maxsize=1)
def get_relevance_engine() -> RelevanceEngine:
    """Process-wide singleton. Build costs ~15s; never construct it per request."""
    info("building audience relevance engine...")
    profile = _build_profile()
    engine = RelevanceEngine(
        profile=profile,
        booking_history=_build_booking_history(),
        demand_source=(
            "ridership_actuals (observed)"
            if has_ridership_actuals()
            else (
                "route_schedules.estimated_ridership (scheduled — ridership_actuals "
                "not provisioned)"
            )
        ),
        pool_sizes=profile["pool_key"].value_counts().to_dict(),
    )
    info(
        f"relevance engine ready: {engine.screens:,} screens, "
        f"{len(engine.pool_sizes):,} audience pools, demand from {engine.demand_source}"
    )
    return engine


# =============================================================================
# TOOLS — the Master Agent's stage-2 surface
# =============================================================================


@tool
def describe_inventory(run_id: str) -> dict:
    """Count the real inventory inside a run's requested geography.

    Reference lookup against the screens/locations/vehicles tables. Use this to confirm
    the campaign geography resolves to real screens before scoring them.

    Args:
        run_id: Handle for the campaign run, from create_campaign_spec.
    """
    if not run_state.exists(run_id):
        error(f"STAGE 2-3 describe_inventory called with unknown run_id={run_id!r}")
        return run_state.unknown_run(run_id, tool="describe_inventory")

    spec = run_state.get_spec(run_id)
    eligible = eligible_screen_ids(spec.city_ids, spec.zone_ids, spec.corridor_ids)
    facts = screen_facts()

    by_type: dict[str, int] = {}
    by_class: dict[str, int] = {}
    zones: set[str] = set()
    for sid in eligible:
        f = facts[sid]
        by_type[f.screen_type] = by_type.get(f.screen_type, 0) + 1
        by_class[f.inventory_class] = by_class.get(f.inventory_class, 0) + 1
        if f.zone_id:
            zones.add(f.zone_id)

    debug(
        f"STAGE 2-3 inventory in cities{spec.city_ids}/zones{spec.zone_ids}/"
        f"corridors{spec.corridor_ids}: {len(eligible)} eligible screen(s), "
        f"{len(zones)} zone(s), by_class={by_class}"
    )
    if not eligible:
        info(f"STAGE 2-3 the requested geography resolves to zero screens (run_id={run_id})")

    return {
        "run_id": run_id,
        "requested_geography": {
            "city_ids": spec.city_ids,
            "zone_ids": spec.zone_ids,
            "corridor_ids": spec.corridor_ids,
        },
        "eligible_screens": len(eligible),
        "by_screen_type": by_type,
        "by_inventory_class": by_class,
        "distinct_zones_covered": len(zones),
        "source": "reference lookup (real data)",
        # A brief asking for "digital screens only" is asking for something that cannot be
        # filtered here, because there is nothing to filter against. Stated rather than
        # left for an agent to infer a flag that does not exist.
        "screen_attributes_available": [
            "screen_type",
            "position",
            "screen_size",
            "inventory_class",
            "zone_id",
        ],
        "no_digital_flag": (
            "screens.csv records no digital/static attribute at all — its only descriptive "
            "columns are screen_type, position and screen_size (S/M/L). The inventory model "
            "itself implies digital: 6 ad slots rotating continuously through a 4-hour "
            "block is not something a static poster does. So a brief specifying 'digital "
            "screens only' cannot be filtered and is already satisfied by every screen. Say "
            "that plainly rather than implying a filter was applied."
        ),
    }


@tool
def build_screen_candidates(run_id: str, top_n: int | None = None) -> dict:
    """Score the eligible inventory and persist the ranked candidate pool.

    Runs the deterministic audience relevance engine: hard-filters the inventory to the
    campaign's geography and screen-type constraints, then ranks what survives on a
    weighted blend of audience match, geography, POI context, time-of-day fit and
    historical booking performance. Produces the `screen_candidates` artifact that the ML
    and OR stages consume.

    All inputs come from the run's campaign spec — audience terms, geography, industry,
    day-type focus and hard constraints. Nothing is passed in and nothing is inferred
    here, so the same spec always yields the same pool.

    Returns an artifact reference plus aggregates, never the candidate rows.

    Args:
        run_id: Handle for the campaign run, from create_campaign_spec.
        top_n: Candidate pool size. Defaults to the configured pool size (250).
    """
    if not run_state.exists(run_id):
        error(f"STAGE 2-3 build_screen_candidates called with unknown run_id={run_id!r}")
        return run_state.unknown_run(run_id, tool="build_screen_candidates")

    spec = run_state.get_spec(run_id)
    limit = top_n or get_settings().candidate_pool_size

    eligible = eligible_screen_ids(spec.city_ids, spec.zone_ids, spec.corridor_ids)
    if not eligible:
        error(f"STAGE 2-3 no eligible screens for run_id={run_id} — geography unsatisfiable")
        return {
            "status": "no_candidates",
            "run_id": run_id,
            "detail": (
                "The requested geography resolves to zero screens. The campaign spec "
                "cannot be satisfied as written — report this instead of proceeding."
            ),
        }

    engine = get_relevance_engine()
    debug(
        f"STAGE 2-3 scoring {len(eligible)} eligible screens, keeping top {limit}, "
        f"audience_terms={spec.audience_terms or '[]'}, day_type_focus={spec.day_type_focus}, "
        f"screen_type_mix={spec.screen_type_mix or '[]'}"
    )
    result = engine.score(spec, eligible, limit)
    ranked, blocks, notes = result.candidates, result.target_blocks, result.notes

    if ranked.empty:
        error(f"STAGE 2-3 hard constraints eliminated every eligible screen, run_id={run_id}")
        return {
            "status": "no_candidates",
            "run_id": run_id,
            "detail": (
                f"All {len(eligible)} screens in the requested geography were removed by "
                f"the spec's hard constraints ({spec.hard_constraints}). Report this and "
                f"offer to relax a constraint — do not relax one unilaterally."
            ),
        }

    debug(
        f"STAGE 2-3 scored {len(ranked)} screen(s) survived hard filtering, target blocks {blocks}"
    )
    for note in notes:
        debug(f"STAGE 2-3 engine note: {note}")

    day_types = (
        [spec.day_type_focus] if spec.day_type_focus in ALL_DAY_TYPES else list(ALL_DAY_TYPES)
    )
    candidates = _to_candidates(engine, ranked, spec, blocks, day_types)
    summary = _summarize(engine, result, candidates, spec, len(eligible))

    ref = write_records(ARTIFACT_KIND, candidates, provenance="computed", summary=summary)
    run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
    if summary["defaults_applied"]:
        # A sub-score that fell back to a neutral default is a data gap, not a ranking.
        info(f"STAGE 2-3 neutral defaults applied: {summary['defaults_applied']}")
    info(f"STAGE 2-3 pool composition by screen type: {summary['pool_composition_by_screen_type']}")
    info(
        f"STAGE 2-3 candidates ready: {len(candidates)} of {len(eligible)} eligible, "
        f"relevance {summary['relevance_min']}-{summary['relevance_max']}, "
        f"{summary['distinct_audience_pools']} audience pools, "
        f"daily reach {summary['pooled_daily_audience']:,.0f} "
        f"(naive {summary['naive_daily_audience']:,.0f}), artifact={ref.artifact_id}"
    )

    return {
        "status": "ok",
        "artifact": ref.as_context(),
        "eligible_screens": len(eligible),
        "candidates_selected": len(candidates),
        "relevance_score_range": [summary["relevance_min"], summary["relevance_max"]],
        "target_time_blocks": blocks,
        "day_type_focus": spec.day_type_focus,
        "audience": {
            "distinct_audience_pools": summary["distinct_audience_pools"],
            "pooled_daily_audience": summary["pooled_daily_audience"],
            "naive_daily_audience": summary["naive_daily_audience"],
            "note": (
                "pooled_daily_audience is the deduplicated figure. Screens sharing a "
                "pool_key see the same people; the naive sum over-counts them."
            ),
        },
        "demand_source": engine.demand_source,
        # Composition is reported unconditionally, mix requested or not. The Master cannot
        # state what a package is made of if it was never told, and a pool that is 100% one
        # screen type is the single most consequential thing about it.
        "pool_composition_by_screen_type": summary["pool_composition_by_screen_type"],
        "eligible_by_screen_type": summary["eligible_by_screen_type"],
        "screen_type_mix_requested": spec.screen_type_mix,
        "screen_type_mix_unfilled": summary["screen_type_mix_unfilled"],
        "constant_subscores": summary["constant_subscores"],
        "top_screens_preview": [
            {
                "screen_id": c.screen_id,
                "relevance_score": c.relevance_score,
                "screen_type": c.screen_type,
                "zone_id": c.zone_id,
            }
            for c in candidates[:5]
        ],
        "defaults_applied": summary["defaults_applied"],
    }


@tool
def describe_relevance_model(run_id: str) -> dict:
    """Report how relevance and audience volume are computed, and what they exclude.

    Reference lookup against the fitted engine — no campaign data involved. Use this to
    justify *why* a candidate ranks where it does, and to state the model's limits
    accurately rather than guessing at them.

    Args:
        run_id: Handle for the campaign run. Accepted for consistency; unused.
    """
    engine = get_relevance_engine()
    debug(
        f"STAGE 2-3 relevance model report requested (run_id={run_id}): "
        f"{engine.screens:,} screens, demand from {engine.demand_source}"
    )
    return {
        "relevance_score": {
            "form": "transparent weighted sum of five 0-1 components",
            "weights": WEIGHTS,
            "note": (
                "transit_score is reported but NOT in this sum — audience volume is the "
                "optimizer's objective quantity, not a measure of fit."
            ),
        },
        "audience_volume": {
            "unit": "PEOPLE PASSING — no viewability discount is applied in this stage",
            "form": "average daily riders per screen x time block x day type",
            "source": engine.demand_source,
            "fixed_screens": "sum of average daily ridership over every route serving the stop",
            "mobile_screens": (
                "corridor block total / vehicles on the corridor — one vehicle's share. A "
                "modelling judgement: route_schedules carries no vehicle_id. The divisor is "
                "published as pool_partition_count so the optimizer can recover the "
                "corridor's whole crowd for the reach ceiling."
            ),
            "slots_per_block": SLOTS_PER_BLOCK,
            "slot_semantics": (
                "A block is a 4-hour window in which all 6 rotation slots cycle "
                "continuously, so slot position is meaningless and holding k slots means "
                "appearing on k of every 6 loop passes. Exposures are LINEAR in slot count; "
                "the conversion from people passing to viewed exposures belongs to "
                "app/optimize/exposure.py, not here."
            ),
            "reach_unit": "pool_key — screens sharing one see the same people",
        },
        "screens_profiled": engine.screens,
        "audience_pools": len(engine.pool_sizes),
        "pool_definition": (
            "A SITE for stop-mounted screens — (city, name, serving corridors) — not a raw "
            "location_id, because one physical station is several location rows. 878 sites "
            "plus 94 corridors."
        ),
        "audience_terms_vocabulary": list(AUDIENCE_TERMS),
        "industry_verticals_vocabulary": list(INDUSTRY_VERTICALS),
        "screen_types_vocabulary": list(SCREEN_TYPES),
        "corridor_pool_sanity": corridor_pool_sanity(),
        "known_limitations": KNOWN_LIMITATIONS,
    }


# =============================================================================
# CONTRACT MAPPING
# =============================================================================


def _to_candidates(
    engine: RelevanceEngine,
    ranked: pd.DataFrame,
    spec: CampaignSpec,
    blocks: list[int],
    day_types: list[str],
) -> list[ScreenCandidate]:
    """Ranked feature rows -> ScreenCandidate contract. Mapping only; no recomputation."""
    candidates: list[ScreenCandidate] = []
    for _, record in ranked.iterrows():
        reasons, defaults = engine.reasons_for(record, spec, blocks, day_types)
        candidates.append(
            ScreenCandidate(
                screen_id=record["screen_id"],
                relevance_score=round(float(record["relevance_score"]), 4),
                audience_match_score=round(float(record["audience_similarity"]), 4),
                geography_score=round(float(record["geographic_fit"]), 4),
                contextual_score=round(float(record["context_fit"]), 4),
                transit_score=round(float(record["transit_score"]), 4),
                time_of_day_score=round(float(record["time_of_day_fit"]), 4),
                historical_performance_score=round(float(record["historical_performance"]), 4),
                reasons=reasons,
                defaults_applied=defaults,
                hard_constraints_passed=True,
                pool_key=_or_none(record.get("pool_key")),
                pool_partition_count=int(record.get("pool_partition_count") or 1),
                impressions_by_block={
                    f"{b}_{dt}": round(float(record[block_column(b, dt)]), 1)
                    for b in ALL_TIME_BLOCKS
                    for dt in ALL_DAY_TYPES
                },
                impressions_weekday=round(float(record["impressions_weekday"]), 1),
                impressions_weekend=round(float(record["impressions_weekend"]), 1),
                impressions_block_1_estimated={
                    dt: round(float(record[f"block_{ESTIMATED_BLOCK}_estimated_{dt}"]), 1)
                    for dt in ALL_DAY_TYPES
                },
                nearby_ambient_footfall=round(float(record["weighted_nearby_footfall"]), 1),
                city_id=_or_none(record.get("city_id")),
                location_name=_or_none(record.get("location_name")),
                zone_id=_or_none(record.get("zone_id")),
                zone_name=_or_none(record.get("zone_name")),
                corridor_id=_or_none(record.get("corridor_id")),
                screen_type=_or_none(record.get("screen_type")),
            )
        )
    return candidates


def _or_none(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def _summarize(
    engine: RelevanceEngine,
    result: ScoringResult,
    candidates: list[ScreenCandidate],
    spec: CampaignSpec,
    eligible: int,
) -> dict[str, Any]:
    """Artifact summary — aggregates only, safe to render into a prompt.

    `preferred_time_blocks` is published here so the pricing stage prices the blocks this
    campaign's audience is actually in, reading one authoritative copy rather than
    reimplementing the mapping.
    """
    ranked = result.candidates
    scores = [c.relevance_score for c in candidates]
    # Deduplicated daily audience: one figure per physical pool, not per screen.
    pooled = ranked.groupby("pool_key")["addressable_impressions"].max().sum()
    constant = [n for n in result.notes if "is CONSTANT at" in n]
    return {
        "eligible_screens": eligible,
        "candidates": len(candidates),
        "relevance_min": round(min(scores), 4),
        "relevance_mean": round(sum(scores) / len(scores), 4),
        "relevance_max": round(max(scores), 4),
        "preferred_time_blocks": [str(b) for b in result.target_blocks],
        "day_type_focus": spec.day_type_focus,
        "audience_terms": spec.audience_terms,
        "distinct_audience_pools": int(ranked["pool_key"].nunique()),
        "pooled_daily_audience": round(float(pooled), 1),
        "naive_daily_audience": round(float(ranked["addressable_impressions"].sum()), 1),
        "demand_source": engine.demand_source,
        "defaults_applied": result.notes,
        # Composition, always. See `build_screen_candidates`'s return payload.
        "pool_composition_by_screen_type": result.pool_by_screen_type,
        "eligible_by_screen_type": result.eligible_by_screen_type,
        "screen_type_mix_requested": spec.screen_type_mix,
        "screen_type_mix_unfilled": result.mix_unfilled,
        "constant_subscores": constant,
    }


TOOLS = [describe_inventory, build_screen_candidates, describe_relevance_model]

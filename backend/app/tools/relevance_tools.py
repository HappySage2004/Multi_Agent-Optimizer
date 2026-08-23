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
comparable between campaigns:
    income_score / high_income_score
                        min-max of zone income_index
    professional_score  0.6 x income_score + 0.4 x occ_prof_affinity
    young_professionals_score
                        0.4 x young_adult + 0.3 x income + 0.3 x occ_prof
    young_adult_score   min-max of pct_age_18_34, scaled 0.85
    student_score       0.4 x young_adult + 0.3 x university_nearby + 0.3 x occ_student
    family_score        0.6 x middle_age + 0.2 x young_adult + 0.2 x occ_family
    commuter_score      peak-block impressions / total impressions

Relevance is a transparent weighted sum of five components (SOLUTION.md section 5):
    0.40 audience_similarity + 0.20 geographic_fit + 0.15 context_fit
  + 0.15 time_of_day_fit     + 0.10 historical_performance

`transit_score` is reported alongside but is NOT in that sum: it is the screen's audience
volume as a percentile of the eligible pool. Volume is the optimizer's objective quantity,
not a fit score — mixing it into relevance would rank a busy screen as a better *match*
than a quiet one in exactly the right zone.

================================ KNOWN LIMITATIONS ================================
Stated here because they change how the output should be read:

1.  IMPRESSION VOLUME IS SCHEDULE-DERIVED ONLY. There is no ambient/pedestrian term. Any
    block with no scheduled service reports exactly zero, so block 1 (00:00-04:00) is zero
    across all 11,163 screens even though `bookings` holds 8,544 real block-1 bookings.
    Treat zero as "not modelled", not "nobody there".

2.  FIXED AND MOBILE SCREENS ARE ~380x APART. `metro_station` median daily volume is
    227,981 against `bus` at 598. The mobile figure is one vehicle's share of its corridor
    (corridor block total / vehicles on corridor) — a stated modelling judgement, since
    `route_schedules` carries no vehicle_id. The divisor is published as
    `pool_partition_count` so the optimizer can recover the corridor's whole crowd for the
    reach ceiling. The gap is directionally real (a station concourse is not one bus) but
    large enough that any volume-per-dollar ranking picks fixed inventory almost
    exclusively. Note the optimizer does NOT rank that way — it maximizes pooled reach,
    which saturates, so cheap distinct pools compete with expensive busy ones: the canonical
    brief's package is 16 `metro_station` + 9 `bus_stop`. That routes around the gap; it
    does not fix it.

3.  MOBILE SCREENS HAVE NO DEMOGRAPHICS. Zone is undefined for a vehicle by construction,
    so `income_score`, `professional_score`, `student_score` and `family_score` are 0 for
    all 2,615 of them and only `commuter_score` carries signal. Their demographic
    sub-score is a floor, not a measurement. Averaging the zones a corridor touches would
    fix this and is the obvious next improvement.

4.  `historical_performance` IS A COMPLETION RATE, not campaign performance. It measures
    whether past bookings on this screen in this vertical were fulfilled. It falls back to
    0.5 for screens with no history in the vertical (4,413 of 11,163 on a `finance` brief).

5.  `commuter_score` SPANS ONLY 0.34-0.51. On a commuter brief the 0.40-weighted audience
    term is close to constant, so ranking is effectively decided by the other four
    components. Not wrong, but do not read a commuter brief's ordering as commuter-driven.

Port note: the feature and scoring logic is a faithful port of the audience relevance
notebook, verified output-identical on all 12 impression columns, POI footfall, POI counts
and all 1,004 pool keys across 11,163 screens. Three deliberate changes, each flagged at
its site: `family_score` is renormalized to 0-1 (it reached 1.140 and broke the
`ScreenCandidate` contract bound), geography is a hard filter before scoring rather than a
soft penalty, and `geographic_fit` handles ID lists and corridor-touches-zone.
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
from app.logging_utils import debug, error, info, warning
from app.models.campaign import AUDIENCE_TERMS, CampaignSpec
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

WEIGHTS: dict[str, float] = {
    "audience_similarity": 0.40,
    "geographic_fit": 0.20,
    "context_fit": 0.15,
    "time_of_day_fit": 0.15,
    "historical_performance": 0.10,
}

AUDIENCE_TERM_TO_SCORE_COLUMN: dict[str, list[str]] = {
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
    "young_professionals": [2, 5, 6],
    "professionals": [2, 5],
    "commuters": [2, 5],
    "students": [2, 4, 6],
    "families": [4, 5],
    "high_income": [2, 3, 5],
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

VEHICLE_MOUNTED_TYPES: tuple[str, ...] = ("bus", "metro_rail_coach")

CONTEXT_FIT_MOBILE_NOTE = "context_fit not applicable -- mobile screen has no fixed location"

EXPECTED_OCCUPATIONS: frozenset[str] = frozenset(
    {"white_collar", "mixed", "retail_service", "blue_collar", "student"}
)

OCC_PROF_MAP: dict[str, float] = {
    "white_collar": 1.0,
    "mixed": 0.5,
    "retail_service": 0.3,
    "blue_collar": 0.2,
    "student": 0.0,
}

OCC_STUDENT_MAP: dict[str, float] = {
    "student": 1.0,
    "mixed": 0.5,
    "retail_service": 0.3,
    "white_collar": 0.0,
    "blue_collar": 0.0,
}

OCC_FAMILY_MAP: dict[str, float] = {
    "blue_collar": 0.7,
    "retail_service": 0.6,
    "white_collar": 0.5,
    "mixed": 0.5,
    "student": 0.0,
}

RIDERSHIP_COLUMNS: tuple[str, ...] = (
    "avg_daily_ridership",
    "daily_ridership",
    "estimated_ridership",
    "route_daily_ridership",
)

KNOWN_LIMITATIONS: tuple[str, ...] = (
    (
        "Volume is schedule-derived only; there is no ambient/pedestrian term. Block 1 "
        "(00:00-04:00) is zero for every screen despite 8,544 real block-1 bookings."
    ),
    (
        "metro_station median daily volume is ~380x bus: any impressions-per-dollar "
        "ranking will favour fixed inventory almost exclusively."
    ),
    (
        "Mobile screens have no zone demographics, so their demographic sub-scores are a "
        "floor of 0 rather than a measurement."
    ),
    "historical_performance is a booking completion rate, not campaign performance.",
    (
        "commuter_score spans only 0.34-0.51, so a commuter brief's ordering is "
        "effectively decided by the other four components."
    ),
    (
        "No held-out accuracy metric: the audience model has never been scored against a "
        "baseline, so no per-screen confidence is emitted."
    ),
)
"""Surfaced by `describe_relevance_model` so the Master Agent states limits it can cite."""


def block_column(block: int, day_type: str) -> str:
    return f"impressions_block_{block}_{day_type}"


BLOCK_COLUMNS: tuple[str, ...] = tuple(
    block_column(b, dt) for b in ALL_TIME_BLOCKS for dt in ALL_DAY_TYPES
)

RANK_OUTPUT_COLUMNS: tuple[str, ...] = (
    "screen_id",
    "pool_key",
    "relevance_score",
    "reason_audience_similarity",
    "reason_geographic_fit",
    "reason_context_fit",
    "reason_time_of_day_fit",
    "reason_historical_performance",
    "total_impressions",
    "impressions_weekday",
    "impressions_weekend",
    "campaign_preferred_blocks",
    "campaign_day_type_focus",
    "impressions_preferred_blocks",
    "defaults_applied",
) + BLOCK_COLUMNS


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


def _as_bool_mask(series: pd.Series) -> pd.Series:
    """Coerce string/boolean terminal flags without raising on mixed dtypes."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin(("true", "1", "yes", "t"))


def compute_stop_weights(route_stops: pd.DataFrame) -> pd.DataFrame:
    """Allocate each route's ridership across its stops instead of copying 100% to each.

    Terminals (first or last) get weight 2.0, intermediates 1.0. If the terminal columns
    are missing, every stop is weight 1.0. `stop_share` sums to 1.0 per route.
    """
    out = route_stops.copy()
    has_first = "is_first_stop" in out.columns
    has_last = "is_last_stop" in out.columns
    if has_first or has_last:
        first = _as_bool_mask(out["is_first_stop"]) if has_first else False
        last = _as_bool_mask(out["is_last_stop"]) if has_last else False
        terminal = first | last
        out["stop_weight"] = pd.Series(1.0, index=out.index)
        out.loc[terminal, "stop_weight"] = 2.0
    else:
        out["stop_weight"] = 1.0

    route_total = out.groupby("route_id")["stop_weight"].transform("sum")
    out["stop_share"] = out["stop_weight"] / route_total.replace(0.0, pd.NA)
    out["stop_share"] = out["stop_share"].fillna(0.0)

    ridership_col = next((c for c in RIDERSHIP_COLUMNS if c in out.columns), None)
    if ridership_col is not None:
        out["allocated_ridership"] = out[ridership_col].astype(float) * out["stop_share"]
    return out


def build_occupation_affinity(occupation: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Map dominant_occupation onto professional / student / family affinities."""
    series = (
        occupation["dominant_occupation"] if isinstance(occupation, pd.DataFrame) else occupation
    )
    series = pd.Series(series)
    present = series.dropna().astype(str)
    unknown = sorted(set(present) - EXPECTED_OCCUPATIONS)
    if unknown:
        warning(
            f"dominant_occupation contains unexpected categor(y/ies) {unknown}; "
            f"expected one of {sorted(EXPECTED_OCCUPATIONS)}. Affinities default to 0.0."
        )

    def _map(mapping: dict[str, float]) -> pd.Series:
        return series.map(lambda v: mapping.get(v, 0.0) if pd.notna(v) else 0.0).astype(float)

    return pd.DataFrame(
        {
            "occ_prof_affinity": _map(OCC_PROF_MAP),
            "occ_student_affinity": _map(OCC_STUDENT_MAP),
            "occ_family_affinity": _map(OCC_FAMILY_MAP),
        },
        index=series.index,
    )


def validate_vocabulary(audience_df: pd.DataFrame) -> None:
    """Assert audience/industry dictionaries stay aligned with the closed vocabs."""
    audience_keys = set(AUDIENCE_TERMS)
    preferred_keys = set(AUDIENCE_TO_PREFERRED_BLOCKS)
    score_keys = set(AUDIENCE_TERM_TO_SCORE_COLUMN)
    if preferred_keys != audience_keys:
        raise AssertionError(
            f"AUDIENCE_TO_PREFERRED_BLOCKS keys {sorted(preferred_keys)} != "
            f"AUDIENCE_TERMS {sorted(audience_keys)}"
        )
    if score_keys != audience_keys:
        raise AssertionError(
            f"AUDIENCE_TERM_TO_SCORE_COLUMN keys {sorted(score_keys)} != "
            f"AUDIENCE_TERMS {sorted(audience_keys)}"
        )
    missing_score_cols = [
        col
        for cols in AUDIENCE_TERM_TO_SCORE_COLUMN.values()
        for col in cols
        if col not in audience_df.columns
    ]
    if missing_score_cols:
        raise AssertionError(f"audience profile missing score columns {missing_score_cols}")
    if not INDUSTRY_TO_POI_CONTEXT:
        raise AssertionError("INDUSTRY_TO_POI_CONTEXT is empty")
    empty_industries = [k for k, v in INDUSTRY_TO_POI_CONTEXT.items() if not v]
    if empty_industries:
        raise AssertionError(f"industry vocabulary has empty POI maps: {empty_industries}")


def get_audience_profile() -> pd.DataFrame:
    """Copy of the cached per-screen profile. Callers must not mutate the cache."""
    return get_relevance_engine().profile.copy()


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

    # --- audience scores -----------------------------------------------------
    df["income_score"] = _normalize(df["income_index"].fillna(0.0))
    df["white_collar_flag"] = (df["dominant_occupation"] == "white_collar").astype(float)
    df["professional_score"] = 0.7 * df["income_score"] + 0.3 * df["white_collar_flag"]

    df["young_adult_score"] = _normalize(df["pct_age_18_34"].fillna(0.0)) * 0.85
    df["student_score"] = 0.6 * df["young_adult_score"] + 0.4 * df["has_university_nearby"].astype(
        float
    )

    # Renormalized: the raw sum peaks at 1.25 (measured 1.140 on this inventory), which
    # exceeds the ScreenCandidate 0-1 bound and would raise on artifact write.
    df["family_score"] = (
        1.0 * _normalize(df["pct_age_35_54"].fillna(0.0))
        + 0.25 * _normalize(df["pct_age_18_34"].fillna(0.0))
    ) / FAMILY_SCORE_MAX

    # --- impression rollups --------------------------------------------------
    # The 12 granular columns stay the source of truth; these are conveniences.
    for dt in ALL_DAY_TYPES:
        df[f"impressions_{dt}"] = sum(df[block_column(b, dt)] for b in ALL_TIME_BLOCKS)
    df["total_impressions"] = df["impressions_weekday"] + df["impressions_weekend"]

    peak = sum(df[block_column(b, dt)] for b in PEAK_BLOCKS for dt in ALL_DAY_TYPES)
    df["peak_impressions"] = peak
    df["commuter_score"] = (peak / df["total_impressions"].replace(0.0, 1.0)).clip(0.0, 1.0)

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

    def score(
        self, spec: CampaignSpec, eligible: set[str], top_n: int
    ) -> tuple[pd.DataFrame, list[int], list[str]]:
        """Rank the eligible inventory.

        Returns the top-N rows, the campaign's target time blocks, and any sub-score that
        fell back to a neutral default for the whole pool.

        Vectorized: every sub-score is computed as a column over the filtered frame, not
        row by row.
        """
        df = self.profile[self.profile["screen_id"].isin(eligible)].copy()
        df = self._apply_exclusions(df, spec)
        if df.empty:
            return df, self.preferred_blocks(spec.audience_terms), []

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
        # Percentile rather than min-max: volume spans ~380x between screen types, and a
        # min-max would crush all but the busiest metro stations to ~0.
        df["transit_score"] = df["addressable_impressions"].rank(pct=True).fillna(0.0)

        notes = [n for n in (audience_note, context_note, tod_note, history_note) if n]
        ranked = df.sort_values(["relevance_score", "screen_id"], ascending=[False, True])
        return ranked.head(top_n).reset_index(drop=True), blocks, notes

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
        if not industry:
            return pd.Series(NEUTRAL, index=df.index), (
                "spec has no industry_vertical — context_fit defaulted to 0.5"
            )
        wanted = INDUSTRY_TO_POI_CONTEXT.get(industry)
        if wanted is None:
            return pd.Series(NEUTRAL, index=df.index), (
                f"industry_vertical '{industry}' has no POI mapping — context_fit defaulted to 0.5"
            )
        target = set(wanted)
        return df["poi_type_set"].map(lambda s: 1.0 if s & target else 0.2).astype(float), None

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

        if row.get("num_nearby_pois"):
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
            reasons.append(
                f"Shares its audience with {pool_size - 1} other screen(s) at "
                f"{row['pool_key']} — their impressions must be deduplicated, not summed"
            )

        return reasons, defaults


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
        f"audience_terms={spec.audience_terms or '[]'}, day_type_focus={spec.day_type_focus}"
    )
    ranked, blocks, notes = engine.score(spec, eligible, limit)

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
    summary = _summarize(engine, ranked, candidates, spec, blocks, notes, len(eligible))

    ref = write_records(ARTIFACT_KIND, candidates, provenance="computed", summary=summary)
    run_state.set_artifact(run_id, ARTIFACT_KIND, ref)
    if summary["defaults_applied"]:
        # A sub-score that fell back to a neutral default is a data gap, not a ranking.
        info(f"STAGE 2-3 neutral defaults applied: {summary['defaults_applied']}")
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
        "audience_terms_vocabulary": list(AUDIENCE_TERMS),
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
                city_id=_or_none(record.get("city_id")),
                zone_id=_or_none(record.get("zone_id")),
                corridor_id=_or_none(record.get("corridor_id")),
                screen_type=_or_none(record.get("screen_type")),
            )
        )
    return candidates


def _or_none(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def _summarize(
    engine: RelevanceEngine,
    ranked: pd.DataFrame,
    candidates: list[ScreenCandidate],
    spec: CampaignSpec,
    blocks: list[int],
    notes: list[str],
    eligible: int,
) -> dict[str, Any]:
    """Artifact summary — aggregates only, safe to render into a prompt.

    `preferred_time_blocks` is published here so the pricing stage prices the blocks this
    campaign's audience is actually in, reading one authoritative copy rather than
    reimplementing the mapping.
    """
    scores = [c.relevance_score for c in candidates]
    # Deduplicated daily audience: one figure per physical pool, not per screen.
    pooled = ranked.groupby("pool_key")["addressable_impressions"].max().sum()
    return {
        "eligible_screens": eligible,
        "candidates": len(candidates),
        "relevance_min": round(min(scores), 4),
        "relevance_mean": round(sum(scores) / len(scores), 4),
        "relevance_max": round(max(scores), 4),
        "preferred_time_blocks": [str(b) for b in blocks],
        "day_type_focus": spec.day_type_focus,
        "audience_terms": spec.audience_terms,
        "distinct_audience_pools": int(ranked["pool_key"].nunique()),
        "pooled_daily_audience": round(float(pooled), 1),
        "naive_daily_audience": round(float(ranked["addressable_impressions"].sum()), 1),
        "demand_source": engine.demand_source,
        "defaults_applied": notes,
    }


TOOLS = [describe_inventory, build_screen_candidates, describe_relevance_model]

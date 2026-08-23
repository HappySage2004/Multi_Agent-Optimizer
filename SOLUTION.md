# SOLUTION.md — Transit Media Campaign Recommendation System

## 1. Purpose and current architecture

This document is the **implementation handoff and source of truth for logical consistency**.
Coding agents should preserve the contracts, units, ownership boundaries, and non-negotiable
rules below when changing implementation.

The system takes a natural-language campaign brief and produces a validated, sales-ready
transit-media recommendation:

```text
Natural language brief
        ↓
CampaignSpec
        ↓
Audience / relevance engine
        ↓
ScreenCandidates
        ↓
Pricing + availability / ScreenEconomics
        ↓
MILP inventory optimization
        ↓
OptimizedPackage
        ↓
Deterministic validation
        ↓
Sales recommendation + explanations
```

### Agent architecture

There are **two specialist agents**, not three:

```text
                         MASTER AGENT
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       Relevance Engine    ML Agent        OR Agent
       deterministic       pricing         optimization
       Master-owned        specialist      specialist
```

- **Master Agent:** intake, orchestration, state, verification, final response.
- **Relevance Engine:** deterministic; **not an LLM agent**.
- **ML Agent:** pricing, availability, seasonality, booking-probability diagnostics, and
  the per-screen demand-value premium.
- **OR Agent:** mathematical optimization and tradeoff interpretation.
- **Validation:** deterministic Python; never let an LLM decide whether a constraint passed.

Do not create an agent for every business stage. LLMs reason and explain; tools calculate.

---

# 2. Current implementation status

All major pipeline capabilities are **BUILT**:

| Capability | Implementation |
|---|---|
| Brief intake + geography resolution | `app/tools/master_tools.py` |
| Audience / screen profile | `v_screen_profile` |
| Relevance scoring | `app/tools/relevance_tools.py` |
| Demand history / audience volume | `v_screen_demand_history` |
| Availability / occupancy | `app/ml/occupancy.py` |
| Market price band | `app/ml/price_band.py` |
| Booking probability | `app/ml/booking_probability.py` |
| Seasonality / event adjustment | `app/ml/seasonality.py` |
| Price recommendation | `app/ml/price_optimizer.py` |
| Demand value / mispricing premium | `app/ml/demand_value.py` |
| Client negotiation profile (advisory) | `app/ml/client_profile.py` |
| Agent-tunable pricing levers | `app/ml/levers.py` |
| MILP optimization | `app/optimize/` |
| Reach accounting | `app/tools/or_agent_tools.py` |
| Validation | `app/agents/validation.py` |
| Master + ML + OR agents | `app/agents/` |

No stage is a stub. The old greedy optimizer has been replaced by a **HiGHS MILP via
`scipy.optimize.milp`**. Solves target a 1% relative gap and report whether the result is
`optimal` or merely `feasible`.

---

# 3. The most important unit rule: blocks vs slots

A **time block is a 4-hour period**. Inside each block there are exactly **6 rotation slots**.

```text
Time block 1 = 00:00–04:00
Time block 2 = 04:00–08:00
...
Time block 6 = 20:00–24:00

Within ONE 4-hour block:

Slot 1 → Slot 2 → Slot 3 → Slot 4 → Slot 5 → Slot 6 → Slot 1 → ...
```

The six slots rotate continuously for the entire 4-hour block; they are **seconds-level
share-of-voice positions, not six separate hours**.

Therefore:

- `slots_booked_per_day = k` means the creative occupies **k of every 6 positions in
  every loop**.
- Slot position has no meaning.
- Exposure is linear in the number of slots.
- The only meaningful diminishing return is at the **shared audience pool**, not between
  slot 1 and slot 6.
- The same six-slot structure repeats every day across the campaign date range.

This distinction must never be lost in variables, schemas, prompts, or optimization logic.

---

# 4. CampaignSpec — central state contract

Everything downstream consumes the normalized campaign specification.

```python
class AudienceTarget(BaseModel):
    age_range: tuple[int, int] | None = None
    income_range: tuple[float, float] | None = None
    occupations: list[str] = []
    commuter: bool | None = None
    other_attributes: dict = {}

class CampaignSpec(BaseModel):
    campaign_objective: str
    industry_vertical: str | None = None
    ad_type: str | None = None

    city_ids: list[str] = []
    zone_ids: list[str] = []
    corridor_ids: list[str] = []

    target_audience: AudienceTarget
    audience_terms: list[str] = []

    start_date: date
    duration_days: int
    budget: float
    requested_num_screens: int | None = None

    preferred_dayparts: list[str] = []
    preferred_time_blocks: list[str] = []
    day_type_focus: Literal["weekday", "weekend"] | None = None

    optimization_goal: Literal[
        "reach", "frequency", "awareness", "conversion"
    ]

    hard_constraints: dict = {}
    soft_preferences: dict = {}
```

Rules:

- budget > 0; duration > 0; dates valid.
- Resolve natural-language geography to IDs.
- Do not invent hard constraints.
- Preserve original user query for traceability.
- `audience_terms` is a **closed vocabulary**:

```text
young_professionals
professionals
students
families
high_income
commuters
```

Unknown terms must be rejected, not silently scored as neutral.

`day_type_focus` affects relevance scoring only. The campaign still runs every day in its
date range; economics uses the actual weekday/weekend calendar mix.

---

# 5. Audience / Relevance Engine

## 5.1 Ownership

The audience stage is a deterministic Master-owned engine in:

```text
app/tools/relevance_tools.py
```

It is intentionally **not an agent**. There is no judgement that benefits from an LLM:
geography filtering, feature computation, scoring, ranking, and audience-volume calculation
are deterministic.

Primary views:

```text
v_screen_profile
v_screen_demand_history
```

`v_screen_profile` has one row per screen for all 11,163 screens.

## 5.2 Screen profile

Important fields include:

```text
screen_id
city_id / zone_id / corridor_id
location_type
screen_type / screen_size / position / inventory_class
pool_key
resident_population
population_density_per_sqkm
median_age
pct_age_18_34
median_household_income
income_index
pct_bachelor_or_higher
dominant_occupation
daytime_population_multiplier
num_nearby_pois
weighted_nearby_footfall
closest_poi_distance_km
nearby_poi_types
pool_partition_count
```

`pool_partition_count` is 1 for stop-mounted inventory and the number of vehicles working
the corridor for vehicle-mounted inventory. It lets the optimizer reconstruct the full
audience pool from one vehicle's share.

### Important known limitations

- Vehicle-mounted screens currently have NULL demographics because vehicles have no zone.
  They therefore receive a scoring floor rather than a measured demographic score. They are
  also excluded from the POI context judgement rather than penalized by it: POIs join on
  `location_id`, which a vehicle does not have, so an empty POI set is architectural and
  scoring it as a mismatch was measuring nothing.
- `daytime_population_multiplier` is carried but not currently used by relevance scoring.
- Event features are not part of the audience profile. Events affect pricing seasonality only.
- Volume is schedule/ridership-derived; there is no pedestrian/ambient term.
- Time block 1 (00:00–04:00) reports zero **measured** audience because no scheduled service
  starts in it — while 8,544 of 191,110 bookings (4.5%) sit there, so the inventory
  demonstrably sells. This is a **gap in the model, not a fact about block 1**: zero means
  “not modelled”, not “nobody there”. `impressions_block_1_estimated` publishes an
  8%-of-block-6 **assumption** per day type, deliberately excluded from every total, from
  off-peak and from `commuter_score` so that no validated figure moves with it.
- `nearby_ambient_footfall` (POI foot traffic) is carried but quarantined: it correlates only
  weakly with transit ridership (~0.12–0.26) and can disagree ~20x at one location, so it is
  used only as a relevance tie-break and never added into volume, reach or price.

## 5.3 Relevance funnel

```text
All screens
   ↓
Hard eligibility filters
   ↓
Eligible screens
   ↓
Relevance score
   ↓
Top-N candidate pool (~100–300, configurable)
   ↓
Optimization
```

Geography is a **HARD filter**, not a soft score.

Within the eligible set:

```text
1.0 exact requested zone/corridor or city-wide brief
0.8 mobile screen whose corridor touches requested zone
0.6 right city but not finer requested geography
0.0 unreachable / guard only
```

## 5.4 Relevance formula

```text
relevance_score =
    0.40 * audience_similarity
  + 0.20 * geographic_fit
  + 0.15 * context_fit
  + 0.15 * time_of_day_fit
  + 0.10 * historical_performance
```

`transit_score` is reported but **not included** in relevance. It is volume information,
which belongs to optimization rather than audience fit.

Audience score columns are normalized once over the full inventory:

```text
income_score               = min-max income_index
young_adult_score          = min-max pct_age_18_34
middle_age_score           = min-max pct_age_35_54
professional_score         = 0.6*income + 0.4*occ_professional
young_professionals_score  = 0.4*young_adult + 0.3*income + 0.3*occ_professional
student_score              = 0.4*young_adult + 0.3*university_nearby + 0.3*occ_student
family_score               = 0.6*middle_age + 0.2*young_adult + 0.2*occ_family
high_income_score          = income_score
commuter_score             = peak-block impressions / total impressions
```

Every weight set sums to 1.0 over inputs already bounded 0–1, so each score is inside the
contract's range **by construction**. `family_score` previously summed a 1.0- and a
0.25-weighted term, peaked at 1.140 on real data, and had to be divided by a 1.25 ceiling;
that constant is gone.

`occ_*` grades `zone_demographics.dominant_occupation`, which has exactly five values across
30 zones — `mixed` (14), `white_collar` (7), `blue_collar` (3), `retail_service` (3),
`student` (3). The previous binary `white_collar_flag` scored `mixed`, the most common value
by a wide margin, identically to `student`. Three affinity maps cover all five values; an
unmapped value warns loudly (schema drift) while a NULL from a vehicle having no zone is the
documented mobile floor, and the two are deliberately not conflated.

Audience terms map to the column that actually measures them:

```text
young_professionals -> young_professionals_score   (was mean(professional, STUDENT))
high_income         -> high_income_score           (was professional_score, 40% occupation)
```

Every sub-score either computes a real value or falls back to **0.5 with an explicit
reason**. Defaults must never be hidden — and a sub-score that comes out **identical across
the whole pool** is reported as `constant_subscores`, because it contributes nothing but its
weight while looking exactly like a normal ranking.

`ScreenCandidate` must carry:

```python
class ScreenCandidate(BaseModel):
    screen_id: str
    relevance_score: float

    audience_match_score: float
    geography_score: float
    contextual_score: float
    time_of_day_score: float
    historical_performance_score: float
    transit_score: float

    reasons: list[str]
    defaults_applied: list[str]
    hard_constraints_passed: bool

    pool_key: str | None
    pool_partition_count: int

    impressions_by_block: dict[str, float]  # 12 keys: block x weekday/weekend
    impressions_weekday: float
    impressions_weekend: float
```

---

# 6. Audience volume and exposure units

This is the second most important part of the system.

There are **three different quantities**:

```text
daily_unique_audience
    = people PASSING the audience pool

reachable_daily_audience
    = people who LOOK
    = daily_unique_audience × viewability
    = reach ceiling

viewed_exposures_per_slot_per_day
    = viewed exposures earned by ONE slot on ONE day
    = daily_unique_audience
      × LOOP_PASSES_PER_TRIP / 6
      × viewability
```

The conversion is implemented **once** in:

```text
app/optimize/exposure.py
```

Do not duplicate the constants or conversion elsewhere.

### Slot mechanics

A 4-hour block has 6 continuously rotating slots. If a campaign owns `k` slots, it owns
`k/6` of every loop pass. Thus viewed exposures are strictly linear in `k`.

`LOOP_PASSES_PER_TRIP = 8` and `VIEWABILITY_*` are **ASSUMED constants**, not measured
from the CSVs. They must remain centralized and clearly labelled as assumptions.

At 8 loop passes:

```text
1 slot on a saturated pool = 8/6 = 1.33 viewed exposures/person/day
30 days ⇒ unavoidable floor ≈ 40 exposures/person reached
```

This is why the wear-out cap is relative to flight length rather than an arbitrary absolute
exposure count.

---

# 7. Reach: the non-negotiable definition

**Reach is never the sum of exposures.**

Screens at the same stop/corridor share people. Every screen has a `pool_key` identifying
the shared physical audience.

Reported reach is:

```text
For each (pool_key, time_block):

    reach_pool =
        min(
            gross viewed exposures bought in that group,
            reachable daily audience for that pool
        )

total reach = SUM(reach_pool)
```

Both terms are in **viewed units**.

Consequences:

- `gross_impressions_viewed` = exposures; scales with slots × days.
- `expected_reach` = distinct people; saturates within a pool.
- Buying more screens/slots at an already saturated pool mostly increases frequency.
- `reach <= exposures` is always enforced.
- Naively summing per-screen exposures over-counted the audience by ~25x on a realistic
  candidate pool (1,656,829 against 65,801 on the canonical brief).
- The dedupe unit is the **site**, not the `location_id`. One physical station is modelled as
  several location rows, and screens on opposite platforms see the same crowd: 910
  stop-mounted `location_id`s resolve to 878 sites, for 972 pools in total including the 94
  corridors. Grouping on name alone would over-merge by ~31% (626), because station names are
  a low-cardinality template unrelated stations share — hence `(city, name, corridor set)`.

The same definition is used by:

1. `app/tools/or_agent_tools.py::_package_metrics`
2. `app/agents/validation.py::_reach_checks`
3. the MILP through `R[p] <= E[p]` and `R[p] <= P[p]`

Do not replace this with a fitted saturation constant.

A separate exponential saturation curve exists only as `curve_reach_diagnostic`. Its
`REACH_LAMBDA` is unmeasured and must **never** become client-facing reach or the optimizer's
objective.

---

# 8. Demand model

Demand is currently an aggregation of observed transit ridership, not a fitted ML predictor.

`v_screen_demand_history` contains average daily riders by:

```text
screen × time_block × day_type
```

There are two structurally different paths.

### Stop-mounted

```text
screen
 → location
 → route_stops
 → route/schedules
 → ridership_actuals
```

For each route/block/day/date:

1. SUM actual ridership across trips.
2. MEAN across dates.
3. SUM across routes serving the stop.

Averaging trips too early would understate stop volume by roughly the trip count.

### Vehicle-mounted

```text
screen → vehicle → corridor
corridor block total / vehicles working corridor
```

This is an approximation because schedules do not identify vehicle IDs.

### Data degradation

`ridership_actuals` is optional/gitignored. If unavailable, stop-mounted demand falls back
to `route_schedules.estimated_ridership`. The source is reported as `demand_source`.

Historical ridership is entirely in the past relative to future campaigns, so the model
works at day-type granularity rather than pretending it has future date-level ground truth.

### Not currently built

- Held-out MAE/RMSE/MAPE/R².
- Prediction intervals.
- Ambient/pedestrian volume.
- Event/POI adjustments in the volume model.

Do not invent confidence values. Current contracts retain confidence fields but no stage
produces a real confidence.

---

# 9. Pricing and availability

The pricing engine is a **process-wide singleton**:

```python
get_pricing_engine()
```

It loads/fits once (~12s build) and is reused across requests.

## 9.1 Availability / occupancy

Capacity is exactly:

```text
6 slots per screen × time block × day
```

Availability is evaluated day by day over the entire requested flight:

```text
occupied[day] = sum(slots_booked_per_day for overlapping bookings)

min_available_slots = min(6 - occupied[day])
avg_occupancy_rate  = mean(occupied[day] / 6)
```

`min_available_slots` is the contract because selling against average availability could
oversell the tightest day.

If required slots do not fit on any day:

```text
feasible = false
pricing = null
diagnostics retained
```

Never silently drop infeasible rows.

## 9.2 Market price band

`app/ml/price_band.py`

Price is seller-side: what the network should quote.

Band = p25 / p50 / p90 of comparable historical
`contracted_price_per_slot_per_day`.

Fallback ladder — deterministic and bounded, no free-form retries:

```text
Z1: screen_size × screen_type × position × zone × daypart, n >= 30
Z2: screen_size × screen_type × position × zone,           n >= 30
A:  screen_size × screen_type × position × city × daypart, n >= 30
B:  screen_size × screen_type × position × city,           n >= 30
C:  screen_size × screen_type × position,                  final floor
```

Level C always resolves: all 15 `(screen_size, screen_type, position)` combinations present
in `screens` also appear in `bookings`.

**Zone sits above city on measurement.** Holding city, size, type and position fixed, the
median contracted price still varies **1.87×–2.52× across zones of the same city** — DAT S
metro_station platform runs from a 33.47 zone median to an 84.30 one over 10 zones and
6,260 bookings. Segmenting on city alone quoted every one of those off the same blended
band, underselling the strong zones and overselling the weak ones. Coverage supports the
depth: 95.6% of bookings sit in a Z1 cell with n >= 30, 99.7% in a Z2 cell, and ~70% of
screens resolve at a zone rung in practice.

Zone is NULL for all 2,615 vehicle-mounted screens, so the zone rungs never match for them
and mobile inventory falls through to the city levels. That is why there is no branch on
inventory class in the lookup.

**Each location rung is also split by deal shape** (`is_bundle`) and tried split-first.
`is_bundle=False` deals hold exactly one screen (max 1); bundled deals a median of 20. A
package this system produces is commercially a bundle, a single-screen quote is not, and
they should not come off the same comparables.

How big the effect is depends on how it is measured, and the two readings disagree:

- By **mean price index** at city+daypart grain the split looks inert: 1.0617 non-bundle
  against 1.0459 bundle, medians 0.9939 and 1.0017.
- By the **actual band quantiles** at zone+daypart grain — the cells this ladder resolves
  at — across the 302 cells where both shapes clear n >= 30, single-screen deals sit
  consistently higher: floor ×1.090, target ×1.079, cap ×1.065, on a band 7.5% narrower.

The quantile reading governs, for two reasons: the price decision consumes **quantiles**
(`floor + position × (cap - floor)`), and a mean of per-booking ratios against a blended
median is a different statistic from the quantiles of each population; and zone grain is
where the band actually resolves. End to end the split moves a single-screen quote ~3.8%
above a bundled one. Coverage survives it — 89.5% of bookings sit in a Z1+shape cell with
n >= 30, 98.9% at Z2+shape.

Shape is nonetheless the **first** dimension surrendered when a cell runs short, being worth
6.5–9% against zone's 87–152%. `is_bundle=None` means "do not split" and is what a caller
that does not know the deal shape must send — never a guess, since guessing bundle on a
single-screen quote reads ~9% low. The rung name records which shape was used
(`..._bundle`, `..._single_screen`, or unsuffixed for a blended fallback).

**One dimension was tested and rejected**, recorded in the module so it is not re-added on
intuition: `duration_days`. The effect is real but small once bundle is controlled, and
exists only inside the bundle population — 1.0577 (<=14d) → 1.0648 (15–30) → 1.0425
(31–60) → 0.9924 (61–120) → 0.9307 (>120), i.e. ~2% between the buckets most campaigns
fall in, and non-monotone on thin non-bundle cells.

Industry adjustment applies only with n >= 15 and is clamped to:

```text
[0.85, 1.20]
```

`position = "not_applicable"` for the 1,400 metro-rail-coach screens.

Always segment by the **screen's own city_id**, not the campaign's city list.

## 9.3 Booking probability

`app/ml/booking_probability.py`

Logistic regression on `log(price)` with screen/city/industry controls.

Labels:

```text
1 = won bookings

0 = lost leads where:
    loss_reason ∈ {price_too_high, budget_mismatch}
```

Only price-driven losses are negatives.

Because the class imbalance is ~490:1:

1. Fit with `class_weight="balanced"`.
2. Recalibrate on a held-out split using Platt scaling.

The price coefficient must be **negative**. Current fit: approximately `-1.18`, AUC ~0.82.

However, the true win rate is ~99.79%, so calibrated probabilities are ~1.0 for almost
everything. Booking probability is therefore a **diagnostic**, not the price-setting signal.

## 9.4 Price decision

`app/ml/price_optimizer.py`

Do **not** maximize `price × P(booked)` as the final pricing rule. It was tested and
degenerates toward the cap because booking probability is too flat.

Current rule — each step annotated with the lever that moves it (§9.7):

```text
1. Check availability / feasibility gate.        <- NO lever reaches this
2. Price band floor/target/cap
     × industry adjustment                       <- industry_weight
3. × day-of-week / holiday multiplier            <- seasonality_weight
   × event boost                                 <- event_weight
4. position = avg_occupancy_rate ** gamma        <- occupancy_gamma
     or an explicit position in the band         <- band_position (overrides 4)
5. recommended_price = floor + position × (cap - floor)
6. × per-screen demand premium                   <- demand_premium_weight
7. × blanket commercial adjustment               <- commercial_multiplier
8. recommended_price = max(price, floor)         <- respect_band_floor
9. Report booking probability as a diagnostic.
```

Price is flat across slot counts by design. `price_by_slot_count` maps 1..6 to the same
price where available and null where unavailable.

Seasonality currently adjusts **price**, not demand: the day-of-week multiplier averages
0.913 over a full week, so a whole-week flight takes a ~9% haircut off a band already built
from actual contracted prices. `seasonality_weight = 0.0` (§9.7) is the documented way to
drop that term without losing the event premium. Event matching uses exact anchor location or
a damped city-zone match because there is no lat/lon data.

---

## 9.5 Demand value — the one signal allowed to disagree with history

`app/ml/demand_value.py`

Sections 9.2 and 9.3 both answer the same question — what did comparable inventory sell
for — which makes the engine a mirror. A screen that has always been sold cheap is quoted
cheap forever, and no amount of history can say the history was wrong. This is the second
opinion, and it **never sees a price**.

`merit` scores a screen on what it physically delivers, as a percentile within its own
`screen_type × city`:

```text
merit = 0.50 × audience volume      (v_screen_demand_history, all day types)
      + 0.20 × zone income_index
      + 0.15 × daytime_population_multiplier
      + 0.15 × weighted_nearby_footfall
```

Ranked within `screen_type × city` because the realized price index is normalized against a
segment that already contains both. Ranking globally would reintroduce the pooling artifact
this model exists to see past — `metro_station` median volume is ~380× `bus`, so a global
volume rank is mostly a screen-type indicator.

**It must not be fitted to price.** A model trained to *predict* price from location and
audience treats the historical average as correct, so a systematically underpriced
*category* is learned as correct and reports no mispricing at all. It can find deviation
from a norm, never a wrong norm. Measured: bus stops in the top audience quintile carry 6.2×
the riders of the bottom quintile and 1.59× the price. A price-trained model calls that
correct.

Four gates, each of which withholds real screens. 385 of 6,690 eligible screens are flagged:

```text
fixed inventory only   merit/absorption correlation is +0.37 bus_stop and +0.31
                       metro_station but -0.28 bus and -0.20 metro_rail_coach — a symptom
                       of the corridor/vehicle volume artifact, not a market fact. Mobile
                       also has no zone demographics, so 3 of 4 merit components are
                       structurally zero, which is why its merit/price-rank correlation is
                       0.02-0.14 against ~0.5 for the fixed types.
>= 10 bookings         below that there is no reliable read on what the screen sells for.
merit >= 0.50          a weak screen priced weakly is the market being right. The residual
                       alone flags 191 below-median-merit screens.
absorption >= 0.30     withholds 281 screens averaging 283,334 riders a day — the HIGHEST
                       of any bucket this model produces — on an absorption rank of 0.166.
                       Separates "undervalued" from "unwanted"; this is what makes the
                       model defensible.
```

**A ranking trap this hit once, recorded so it is not reintroduced.** `pr_price` is a
percentile among screens that *have* a price index. Ranking the NULL-price screens in the
same window partition squeezed the 181 priced bus_stop/ACS screens into ranks 0.000–0.257
rather than 0.000–1.000: every price rank depressed, every residual inflated, and screens
transacting *above* their own comparables flagged as underpriced (811 premiums against the
correct 385). Hence the extra `(price_index IS NULL)` partition key.

The premium ramps linearly from residual 0.20 (+0%) to 0.60 (+15%, the cap), so the gate is
not a cliff. The cap is conservative by construction: at the observed mean of ×1.056 the
flagged set moves from an average transacted 64.51 to 68.14, still well **below** the 88.31
of the screens the model leaves alone and far below the 118.23 it flags as overpriced — it
narrows a gap rather than closing it. Commercial risk is measured, not assumed: in
`lost_leads`, deals lost to price wanted a third off (`price_gap_pct` 0.328 for
`price_too_high`, 0.309 for `budget_mismatch`) against 0.025 for deals lost to a competitor.

**This is the only adjustment that may carry a quote above the band cap**, deliberately: an
underpriced screen's own comparables are what understate it, so clamping the premium back
inside the band would delete exactly the correction it exists to make. A test asserts that
nothing else can exceed the cap and that the excess is bounded by the premium.

Auto-applied at `demand_premium_weight = 1.0`, and self-correcting — if a raised price stops
a screen selling, occupancy falls and step 4 of §9.4 pulls the quote back down on the next
run. Measured on the canonical brief: 24 priced lines carry a premium (mean ×1.083), package
cost 47,449 against 47,204, for **identical reach** of 288,446. Same audience delivered,
more revenue.

There is no ground truth for what a screen is worth, so merit ships with no held-out
accuracy metric and cannot acquire one from this dataset. Validate forward: do
premium-flagged screens keep their occupancy after the rise?

---

## 9.6 Client negotiation profile — advisory only

`app/ml/client_profile.py`

Wired to **no engine, no lever default and no artifact**. It answers one question for the
salesperson holding the conversation: how has this client behaved on price before? Surfaced
by the Master-owned read-only tool `get_client_negotiation_profile`, which resolves by
`client_id` or company name and returns `ambiguous` rather than guessing.

Worth building because the relationship data is dense: **96% of clients are repeat business**
(499 of 520 have more than one deal; median 36 deals, mean 109), and every one of the 807
lost leads with a known `client_id` belongs to a client who also has bookings. Every
identified loss is a recorded price objection from a client still on the books.

**Advisory is structural, not squeamish.** Price-driven loss rates are **flat** across
leverage tiers — 34.2% high, 32.5% medium, 34.8% low — so posture says nothing about whether
the deal is lost, only what it settles at. Applying it automatically would be pricing off the
half of the finding that does not hold. `suggested_commercial_multiplier` exists, is capped to
`[0.90, 1.15]`, and only the rep may act on it, through `set_pricing_levers`.

**`negotiation_leverage` is context, never a forecast.** It looks predictive only under
line-item weighting:

```text
weighting                   high    medium     low
per line item (median)    0.9651    1.0167  1.0188   <- monotone, looks clean
per line item (mean)      1.0128    1.0640  1.0778   <- monotone
per CLIENT (median)       1.0394    1.0129  1.0528   <- ordering breaks
per CLIENT (mean)         1.0513    1.0823  1.0659   <- ordering breaks
```

The label tracks **account size**, not price behaviour: high-leverage clients carry a median
328 priced line items against 172 (low) and 165 (medium). Weighted by volume the big
accounts dominate and do transact better; give every client one vote and the ordering
inverts. The question here is about a client, so the per-client figure governs — and it says
the tier is a population reference.

**What the profile leads with instead**, both measured per client:

- Their **realized price index** — the mean of `price ÷ its own segment median` across their
  line items — judged against its own **standard error**.
- Their **objection history**: price-driven losses, the discount they asked for, negotiation
  rounds, competitor mentions.

Confidence is deliberately a property of the **departure**, not the sample size. A 1.001
index on 500 line items is precisely measured and actionably meaningless, so it scores
"weak" and suggests nothing. The gate is real: within-client spread averages 0.214 against a
between-client spread of p10 0.956 to p90 1.239, so a client's index is a central tendency
and a poor per-deal predictor.

Two data traps handled explicitly:

- `price_gap_pct` is not populated on every price-driven lead. A missing gap reads "discount
  asked not recorded", never "asking for 0% off", which would tell a rep the exact opposite
  of the truth.
- The segment medians include the client's own bookings, so a client holding a large share of
  a thin segment is partly measured against themselves. That pulls their index toward 1.0 —
  conservative rather than wrong — and is recorded as a known limitation rather than
  corrected.

---

## 9.7 Pricing levers — the decision is parameterized

`app/ml/levers.py`

Steps 2, 3, 4, 6, 7 and 8 of §9.4 used to be derived once and applied on every run, so a
second agent turn could not act on anything the sales rep said. Each is now a bounded
run-scoped parameter:

```text
seasonality_weight     [0.0, 2.0]   day-of-week / holiday multiplier
event_weight           [0.0, 2.0]   event-overlap boost
industry_weight        [0.0, 2.0]   industry band adjustment
demand_premium_weight  [0.0, 2.0]   the §9.5 premium
occupancy_gamma        [0.25, 4.0]  occupancy -> position in band
band_position          [0.0, 1.0]   fixed position, overrides occupancy
commercial_multiplier  [0.70, 1.30] blanket negotiation adjustment
respect_band_floor     bool         hold the quote at or above the band floor
note                   free text    the rep's reason, carried onto the artifact
```

Four invariants, each pinned by `tests/test_pricing_levers.py`:

- **All defaults are identity**, where identity means "use the model's derived value" —
  `seasonality_weight = 1.0` applies the derived seasonality multiplier. Only
  `demand_premium_weight` *does* something at its default, because §9.5 is auto-applied by
  design; 0.0 returns the engine to pricing purely off historical comparables.
- **Clamped in code, not in the prompt.** `clamp()` bounds and *reports* rather than raising,
  because a rejected tool call in an agent loop becomes a retry against a per-minute rate
  limit to arrive at the number clamping returns directly. `industry_weight` is re-clamped
  after weighting, so §9.2's `[0.85, 1.20]` guarantee holds at any weight.
- **No lever reaches feasibility.** Availability, occupancy and the band's comparables are
  inventory truth; a sold-out screen stays sold out.
- **Weights dial influence, not magnitude.** `effective_multiplier(m, w) = 1 + w(m-1)`, so
  `w=0` means exactly "term off" whichever side of 1.0 the multiplier sits on. Scaling would
  turn a 1.15 event premium into 0.0.

The two seasonality terms are weighted **separately** and only then multiplied, so a rep can
keep the event premium while dropping the weekday haircut. That makes
`seasonality_weight = 0.0` the documented answer to the price-vs-demand double count noted in
§9.4 and §21. `SeasonalityAdjustment.combined_multiplier` is the *unweighted* product and is
no longer what the engine applies; `ScreenEconomics.seasonality_multiplier` reports the figure
the price was actually computed from.

Levers live on the **run**, set by `master_tools.set_pricing_levers` and read by
`estimate_screen_economics` — never passed through a delegation message, per the "run handles,
not payloads" rule in §15. A float that has to survive an LLM paraphrase is a float that will
eventually arrive wrong. They surface in `get_active_run`'s `campaign_inputs` (so moving one
is a REBUILD, not a question), on the `screen_economics` artifact summary, and in
`inspect_package`: a quote a human moved is a different claim from a quote the model derived,
and the recommendation must say which one it is.

---

# 10. ScreenEconomics contract

One row per:

```text
candidate screen × time block
```

Infeasible rows are retained.

```python
class ScreenEconomics(BaseModel):
    screen_id: str
    time_block_id: str
    feasible: bool = True

    max_slots_per_day: int = 0
    occupancy_rate: float | None = None
    price_by_slot_count: dict[int, float | None] = {}

    viewed_exposures_per_slot_per_day: float = 0.0
    daily_unique_audience: float = 0.0
    reachable_daily_audience: float = 0.0
    viewability_factor: float | None = None
    pool_key: str | None = None

    demand_forecast: DemandForecastSummary | None = None
    pricing: PricingRecommendation | None = None
    expected_revenue: float = 0.0

    seasonality_multiplier: float | None = None
    event_match_type: str | None = None

    pricing_internal_reach_proxy: float | None = None

    # demand value / mispricing, from app/ml/demand_value.py
    demand_value_index: float | None = None      # merit percentile, price-blind
    historical_price_index: float | None = None  # what it actually transacted at
    demand_premium: float | None = None          # 1.0 = none
    demand_value_reason: str | None = None

    reach_owner: str = "audience_engine"
    assumptions: list[str] = []
```

Key interpretation:

- `daily_unique_audience`: people passing; **not reach**.
- `reachable_daily_audience`: people who look; **reach ceiling**.
- `viewed_exposures_per_slot_per_day`: one slot's viewed exposures on one day.
- `pool_key`: deduplication unit.
- `max_slots_per_day`: tightest-day availability.
- `pricing=None` means infeasible.
- `demand_value_index`: merit, computed **without ever seeing a price** (§9.5).
- `demand_premium`: the **only** adjustment permitted to carry `recommended_price` above
  `pricing.cap`. `None` or `1.0` means no premium, and `demand_value_reason` says which gate
  stopped it — the screens *without* a premium are the interesting case.

The old `pricing_internal_reach_proxy` is quarantined. It uses hand-set dwell/visibility
factors and mismatched mobile/fixed units; it must not populate demand or reach fields.

---

# 11. MILP optimization

The optimizer is deterministic and lives in:

```text
app/optimize/
```

Solver: **HiGHS via `scipy.optimize.milp`**, target relative gap 1%.

The OR wrapper is:

```text
app/tools/or_agent_tools.py
```

## 11.1 Decision variables

For candidate cell `i = (screen, time_block)`:

```text
y[i,k] binary   cell gets at least k+1 slots, k=0..5
z[s]   binary   screen s is used

E[p]   continuous   viewed exposures in pool p
R[p]   continuous   reach in pool p

c[g]   continuous   coverage shortfall
w[p]   continuous   wear-out shortfall
```

Slot ordering:

```text
y[i,k+1] <= y[i,k]
slots_i = SUM(y[i,k])
```

`z[s]` is required because screen count means **distinct screens**, not screen×block cells.

## 11.2 Objective profiles

```text
             w_reach  w_freq  w_conv
reach          1.00    0.00    0.00
awareness      0.70    0.30    0.00
frequency      0.20    0.80    0.00
conversion     0.40    0.20    0.40
```

Weights are normalized by natural ceilings.

- `reach`: distinct people.
- `awareness`: breadth + repetition.
- `frequency`: depth.
- `conversion`: uses `conv_fit`, an audience-engine industry→POI context proxy.

**Conversion is not a true conversion model.** There is no conversion data. The system
must disclose this whenever conversion is requested.

Cost is only a tiny tie-breaker among otherwise equal solutions. It must not materially
sacrifice reach.

## 11.3 Reach formulation

For every audience pool:

```text
R[p] <= E[p]
R[p] <= P[p]
```

where:

- `E[p]` = viewed exposures allocated to the pool.
- `P[p]` = reachable population ceiling.

Maximizing `R[p]` therefore gives:

```text
R[p] = min(E[p], P[p])
```

This is the exact same definition used by reporting and validation.

Do not reintroduce the exponential `lambda` curve into the main objective.

## 11.4 Hard constraints

Enforced by the solver and independently checked by validation:

```text
total cost <= budget
purchased slots <= tightest-day availability
campaign dates valid
geographically eligible screens only
requested/min/max distinct screen count
allowed screen types
required time blocks
min_zone_coverage
min_budget_utilization only if explicitly declared by the brief
max_slots_per_day      only if declared; binds PER SCREEN PER DAY, across blocks
```

## 11.5 Elastic constraints

Penalized and reported rather than automatically failing:

```text
coverage groups
wear-out frequency cap
```

If a hard constraint forces an elastic breach, return the package and disclose the breach.

**Never invent a minimum budget-utilization requirement.**

## 11.6 The brief's slot structure

`hard_constraints["max_slots_per_day"]` is how a brief states the leasing structure — "1
rotating slot (15 seconds per minute) on digital screens only" and similar phrasings, which
are common because that is how the rate card is written.

**The incident this closes.** There was no channel for it at all. `CampaignSpec` has no slot
field; `hard_constraints` is a free-form dict that every consumer reads against its own
hardcoded key list; and the only real control was `optimize_package`'s `slots_per_day_cap=3`
default argument, mentioned in neither the OR agent's prompt nor the Master's, so no model
ever set it. A brief asking for one slot shipped a package buying three,
`verify_package` passed it clean, and the constraint sat visible in
`normalized_spec.hard_constraints` and in `get_active_run`'s `campaign_inputs` the whole
time. Every layer reported success.

**It binds per SCREEN per day, summed across time blocks.**

```text
for each screen s:   sum over its cells i, over slot levels k:  y[i,k]  <=  cap
```

The alternative reading — cap each `(screen, time block)` cell — is all that an `available`
clip can express, and it is wrong for the constraint briefs actually write: under it a screen
bought in Block 3 and Block 5 takes 1 + 1 = 2 slots that day and passes. Measured on a
45-day brief, a per-cell cap of 1 returned a plan whose busiest screen carried 2 slots/day.
The per-screen reading is also the stricter one, and for a compliance constraint the stricter
reading is the right default. The applied cap, its `source` and the semantics string travel
in the package payload, so the reported figure always says which reading produced it. The
`available` clip stays, as an implication of the per-screen limit that shrinks the search —
never as the constraint.

**Read off the run, never off a tool argument** — the same rule `PricingLevers` follows. A
constraint that has to survive an LLM paraphrase is a constraint that will eventually arrive
wrong, and one that arrives wrong here ships as an over-delivery the client is contractually
owed. A caller override may tighten a declared cap and never widen it: widening on request is
precisely how a constraint gets relaxed until a package appears.

**Honouring it costs no reach, and that was measured rather than assumed.** Across four
briefs, four cap levels and all four objectives, reach is identical at every cap — 45,328 /
44,717 / 173,603 / 169,439 / 95,355 — because reach is bounded by each pool's *daily*
reachable audience while exposures accumulate over a flight of 30 days or more, so one slot
already over-saturates its pool and extra depth buys frequency rather than people. This is
the claim to lead with: it is the same on every brief measured, and it means the compliance
argument costs the client nothing.

**Whether it also reduces repetition is brief-dependent.** The intuitive claim — a 1-slot
cap pulls frequency toward the flight floor — holds only where the cap actually binds. Two
measurements, opposite results:

```text
2-zone 30-day reach brief, 150k budget
  cap 3 -> 111.8 exposures/person, spend 92,417   |  cap 1 -> 40.0, spend 34,334
  (40.0 is exactly the flight's unavoidable floor; reach 173,603 either way)

2-zone 45-day brief, 50k budget
  cap 1 / 3 / 6 are IDENTICAL on every objective, because the solver was already
  choosing 1 slot per screen — the cap never bound
```

And on the exposure-weighted objectives (`awareness`, `frequency`, `conversion`) the cap does
not bound frequency **at all**: those profiles reward exposures, so the solver stacks extra
SCREENS into the same pool once it cannot stack slots, and the 45-day brief returns 169.6
exposures per person at every cap level. A slot cap constrains the depth bought on one
screen. It is not a frequency cap, and must not be sold as one — the wear-out cap is the
frequency instrument, and it is elastic.

**`MAX_CELLS_PER_POOL = 4` is what bounds that screen-stacking, and it must not be
"fixed".** The intuition is that 4 cells at a 1-slot cap buy a third of the depth 4 cells buy
at 3 slots, so the allowance should scale to hold pool depth constant. That was implemented
and measured on the 30-day brief above: reach identical, spend 34,334 → 115,628, exposures
per person 40.0 → 112.2. The extra cells let the solver stack *screens* into a pool it could
no longer stack slots into. That is the same mechanism that makes the cap inert on the
exposure-weighted objectives, and widening the allowance would hand it to every objective
including `reach`. The constant is deliberately cap-independent and
`contract._prune_saturated_pools` records why.

**"On digital screens only" is a no-op in this dataset.** `datasets/screens.csv` carries
`screen_id, city_id, screen_type, location_id, vehicle_id, position, screen_size` — verified,
and no digital/static attribute among them. The inventory model itself implies digital: 6 ad
slots rotating continuously through a 4-hour block is not something a static poster does. So
the constraint cannot be filtered and is already satisfied by every screen. Say that in the
answer rather than inventing a filter; `describe_inventory` returns the same statement in
`no_digital_flag` so an agent does not have to infer it.

## 11.7 Unrecognized hard constraints fail verification

The generalization, and the part that stops this class of miss recurring under a different
key. `hard_constraints` stays free-form, but `ENFORCED_HARD_CONSTRAINTS` in
`app/models/campaign.py` is the closed vocabulary of keys some stage actually enforces, and
the `hard_constraints_recognized` check **fails** a package whose spec carries a key outside
it.

A fail rather than a warning, because the outcomes are not symmetric. A false fail costs one
turn — the Master tells the rep the constraint cannot be enforced and they restate it. A
silent pass ships a package that breaches a written brief. Adding a key to that set without
the code that enforces it re-creates the bug the set exists to prevent.

---

# 12. Wear-out

Wear-out is expressed relative to the flight's unavoidable exposure floor:

```text
E[p] <= F_max × R[p] + w[p]
```

At `LOOP_PASSES_PER_TRIP = 8`, a saturated pool receives:

```text
8/6 viewed exposures/person/day
```

so a 30-day flight has an unavoidable floor of ~40 exposures/person reached.

Therefore an absolute cap below that floor would make the campaign mathematically
infeasible for reasons unrelated to selection. The cap should limit **additional stacking**
within already saturated pools, not pretend it can eliminate the flight's structural floor.

---

# 13. OptimizedPackage contract

```python
class Allocation(BaseModel):
    screen_id: str
    time_block_id: str
    slots_per_day: int
    duration_days: int
    price_per_slot_per_day: float
    viewed_exposures: float
    expected_revenue: float

class OptimizedPackage(BaseModel):
    allocations: list[Allocation]

    total_cost: float
    gross_impressions_viewed: float
    expected_reach: float
    expected_frequency: float

    budget_utilization: float
    constraint_status: dict[str, bool]
    objective_value: float
    optimization_method: str

    curve_reach_diagnostic: float | None = None
    unmet_coverage: dict[str, float] = {}
    wear_out_exposures_over_cap: float = 0.0
```

Interpretation:

```text
gross_impressions_viewed = gross exposures
expected_reach            = deduplicated people
expected_frequency        = exposures / reached people
```

Never call gross exposures “reach”.

`optimization_method` must identify MILP/HiGHS, objective profile, gap and solver status.

---

# 14. Recommendation generation

The Master/Recommendation layer must **not recalculate analytical values**.

Inputs:

```text
CampaignSpec
ScreenCandidates
ScreenEconomics
OptimizedPackage
Validation results
```

Output should contain:

```text
executive_summary
recommended_package
key_recommendations
screen_explanations
pricing_explanation
audience_explanation
optimization_explanation
risks
alternatives
```

Explanations must cite real values/features.

Bad:

> This screen is highly relevant.

Good:

> This screen ranks highly because the zone has a high 18–34 population share,
> strong daytime population uplift, and high evening transit demand.

---

# 15. Master / specialist interfaces

All agent boundaries use `run_id`, not large payloads.

Run state contains:

```text
CampaignSpec
artifact references
optimization result
validation result
```

Candidate tables and price tables never enter LLM context directly.

## Master tools

`app/tools/master_tools.py`

```python
get_active_run(session_id)
resolve_geography_terms(terms)
create_campaign_spec(...)
get_run_state(run_id)
get_client_negotiation_profile(client)
set_pricing_levers(run_id, ...)
verify_package(run_id)
inspect_package(run_id, limit=10)
check_explanations(run_id, explained_screen_ids)
```

`set_pricing_levers` (§9.7) and `get_client_negotiation_profile` (§9.6) are Master-owned
because both serve the conversation with the sales rep rather than a pipeline stage. Neither
is mapped to a stage in the UI's stage rail, and neither creates a run — the client profile is
read-only and touches no package, which a test pins.

`get_active_run` is what makes a follow-up turn cheap: it returns the session's latest run
plus a `campaign_inputs` dict of exactly what the optimizer consumed — pricing levers
included — so the Master can decide ANSWER (read the existing package) vs REBUILD (an input
changed) rather than re-running the pipeline on every message.

## Relevance tools

`app/tools/relevance_tools.py`

```python
describe_inventory(run_id)
build_screen_candidates(run_id, top_n=None)
describe_relevance_model(run_id)
```

## ML tools

`app/tools/ml_agent_tools.py`

```python
estimate_screen_economics(run_id, time_blocks=None, slots_needed=1)
describe_pricing_model(run_id)
```

The ML agent does not own demand/reach. Audience-volume fields are mapped from candidates
and converted through the single exposure module.

## OR tools

`app/tools/or_agent_tools.py`

```python
optimize_package(run_id, slots_per_day_cap=None)
compare_objectives(run_id, objectives=None)
```

The OR agent must state:

- gross viewed exposures vs distinct reach,
- optimization objective,
- solver status/gap,
- wear-out disclosure,
- unmet coverage if any.

---

# 16. Shared artifacts

Do not pass huge DataFrames between agents.

Use durable artifacts:

```text
CampaignSpec
ScreenProfiles
ScreenCandidates
ScreenEconomics
OptimizedPackage / InfeasibilityReport
CampaignRecommendation
```

Specialists receive:

```text
artifact reference
+ schema
+ concise summary
+ task
```

rather than raw rows.

Any lever a human moved must travel with the artifact it affected: the `screen_economics`
summary carries the active `PricingLevers`, so a quote is always traceable to whether the
model derived it or a rep moved it.

`provenance` must be preserved on artifact references. Current pipeline should report:

```text
stub_stages = []
```

---

# 17. Data layer

Use:

```text
CSV
 ↓
DuckDB
 ↓
SQL views
 ↓
Python / ML / optimization tools
```

`app/data/db.py` registers the 14 source CSVs as DuckDB views, including the large
`ridership_actuals` without materializing it.

Important derived views:

```text
v_screen_geography
v_corridor_zones
v_screen_profile
v_screen_demand_history
v_screen_poi
v_schedule_block
v_route_block_demand
v_corridor_block_demand
v_corridor_vehicle_count
```

There are no separate `v_historical_pricing` or `v_campaign_inventory` materializations;
pricing reads explicit projected columns through `app/ml/loaders.py`.

### Core joins

```text
bookings.screen_id              → screens.screen_id
bookings.client_id              → client_facts.client_id
bookings.time_block_id          → dim_slot.time_block_id

screens.location_id             → locations.location_id
screens.vehicle_id              → vehicles.vehicle_id

vehicles.corridor_id            → route_schedules.corridor_id
vehicles.corridor_id            → route_stops.corridor_id

route_schedules.schedule_id     → ridership_actuals.schedule_id
route_schedules.route_id        → route_stops.route_id

locations.zone_id               → zone_demographics.zone_id
locations.city_id               → cities.city_id

events.poi_id                   → points_of_interest.poi_id
events.anchor_location_id       → locations.location_id
points_of_interest.anchor_location_id → locations.location_id
```

Avoid accidental many-to-many explosions; verify expected cardinality for every join.

---

# 18. Validation

Validation is deterministic Python in:

```text
app/agents/validation.py
```

Every check returns `pass`, `fail`, or `skipped`. Any `fail` fails the package.

Core checks:

```text
package_non_empty
budget_respected
cost_reconciles
budget_utilization_reconciles
impressions_reconcile
reach_not_above_impressions
frequency_reconciles
reach_reconciles
curve_reach_bounded
screens_exist
time_blocks_valid
no_duplicate_allocations
geography_eligible
duration_within_campaign
start_date_not_in_past
inventory_availability
hard_constraints
hard_constraints_recognized
model_confidence
```

`model_confidence` is currently **SKIPPED** because no stage has a real held-out confidence.

Conditional checks apply only when declared:

```text
requested_num_screens
min_screens
max_screens
allowed_screen_types
required_time_blocks
min_zone_coverage
max_slots_per_day
```

`max_slots_per_day` re-sums slots **per screen across time blocks** from the allocations,
which is a different assertion from `inventory_availability`. That one checks
`slots_per_day <= max_slots_per_day` per *cell* — inventory truth about what is unsold, not a
client commitment — and a plan can satisfy it while breaching the brief. Only an independent
re-derivation surfaces a constraint the optimizer ignored; without one the miss is silent,
which is exactly what happened.

The validator independently recomputes:

1. **Cost:** `sum(price × slots × days)`.
2. **Reach:** pool-level deduplicated reach using `pool_key`.
3. Exposure/reach/frequency reconciliation.

Do not share the same implementation between solver accounting and validation; the
independent implementation is intentional.

---

# 19. Infeasibility

Never fabricate a package.

Fixed reason codes:

```text
BUDGET_CONSTRAINT
TOO_MANY_SCREENS_REQUESTED
INSUFFICIENT_INVENTORY
DAYPART_UNAVAILABLE
GEOGRAPHY_UNAVAILABLE
CONFLICTING_HARD_CONSTRAINTS
DATES_UNAVAILABLE
NO_CANDIDATES
```

Return either:

```text
OptimizationResult.package
```

or:

```text
OptimizationResult.infeasibility
```

never a speculative partial package.

Relaxation options must include actionable numbers whenever possible.

A declared `max_slots_per_day` is diagnosed by **re-solving without it**, the same way the
utilisation floor and the exact screen count are. A 1-slot cap removes up to five sixths of a
screen's purchasable depth, so it can make a brief infeasible that the budget alone would
have covered, and reporting `BUDGET_CONSTRAINT` for it sends the rep to ask for money that
would not help. The report names the cap, states the depth an uncapped plan would have used,
and offers relaxing it as the client's decision — the cap is never quietly widened to make a
package appear.

Known upstream issue: top-N relevance selection is not coverage-aware. A candidate pool
can contain only one of two required zones, making `min_zone_coverage` impossible before
the solver sees it. If changing candidate selection, preserve hard geography filtering
and make coverage-aware selection explicit rather than weakening the solver constraint.

---

# 20. Evaluation and test invariants

Tests should protect **invariants**, not merely restate implementation.

## Audience / relevance

Must hold:

```text
all scores ∈ [0,1]                      (no renormalization constant: weights sum to 1.0)
12 impression columns exist and contain no NaNs
pool_key never null, and there are exactly 972 pools (878 sites + 94 corridors)
locations sharing (city, name, corridor set) share a pool_key
stop shares sum to exactly 1.0 per route
a corridor's stops sum to exactly 1.000x its ridership (same source)
mobile volume is untouched by the stop-share correction
block 1's MEASURED value stays zero; its estimate is a separate field
peak + offpeak == total exactly, on measured columns only
all audience terms map to real score columns/blocks
set(INDUSTRY_VERTICALS) == set(INDUSTRY_TO_POI_CONTEXT)
unknown audience terms AND unknown industry verticals are rejected
every dominant_occupation value is mapped
geography is hard-filtered
allowed screen types are enforced
no mobile screen takes the POI mismatch penalty
exact relevance ties are ordered by size, then footfall, then screen_id
a requested screen_type_mix yields every available requested type
a corridor pool is never smaller than a station on it
different audience terms can change blocks and ranking
missing data is recorded in defaults_applied
a pool-wide constant sub-score is reported loudly
reasons cite real values
pooled audience < naive screen sum
real weekday/weekend calendar mix is used
exposure conversion occurs exactly once
candidates are ranked best-first
```

## Pricing

Must hold:

```text
price coefficient < 0
calibration matches true base rate
probability decreases with price
probabilities ∈ [0,1]
negative class = price-driven losses only
floor <= target <= cap at every rung
industry adjustment within clamp at any industry_weight
every screen attribute combination has a band
recommended price lies within band, EXCEPT for the demand premium
availability never exceeds capacity
tightest day defines availability
infeasible rows retained

zone rungs carry most of the inventory
mobile screens fall through the zone rungs with no branch on inventory class
the zone band differs from the city blend it replaced
deal shape is omitted by default and reproduces the blended band
single-screen comparables price above bundled ones
the rung name discloses which deal shape was used
deal shape is surrendered before zone

merit weights sum to 1, and merit is computed with no price input
price rank is ranked only among screens that HAVE a price
mobile inventory never receives a demand premium
a screen that does not sell is withheld however good it looks
a cheap but weak screen is not called underpriced
the premium ramps from the gate rather than jumping at it
ONLY the demand premium can carry a quote above the band cap
disabling the premium returns the engine to comparables only
the premium does not change what is purchasable

every lever default is identity and reports no changes
out-of-range levers are bounded, not rejected
no lever can make a sold-out screen feasible
levers do not change occupancy
band_position beats occupancy_gamma when both are set
seasonality_weight=0 leaves only the event term
a client suggestion never escapes its own bound
a thin or noisy client history suggests no change at all
the leverage tier is never presented as a forecast
the client profile tool creates no run and touches no package
```

## Optimization / validation

Must hold:

```text
budget respected
inventory respected
geography respected
dates respected
distinct screen count respected
all four objectives produce feasible validated packages
MILP reaches at least independent greedy baseline on reach
reach <= min(total exposures, total reachable population)
gross exposures cannot be reported as reach
declared spend floor is enforced/reported
wear-out cap is relative to the flight floor
declared slot cap binds per SCREEN across blocks, not per cell
the slot cap is read off the run; an override may tighten it, never widen it
honouring a 1-slot cap costs no reach and does not increase repetition
pool pruning is NOT scaled by the slot cap
```

Also protect the known adversarial case:

```text
A package claiming gross exposures as reach MUST fail validation.
A package breaching a brief-declared slot cap MUST fail validation.
A spec carrying a hard constraint no stage enforces MUST fail validation.
```

---

# 21. Known limitations — do not silently “fix” these by changing definitions

The following are known model limitations and should remain explicit until the underlying
data/model changes:

1. `LOOP_PASSES_PER_TRIP` and `VIEWABILITY_*` are assumptions without ground truth.
2. `REACH_LAMBDA` is diagnostic-only and not used for reported reach.
3. Demand has no held-out accuracy metric or confidence interval.
4. Vehicle-mounted screens lack demographic data and currently receive a scoring floor.
   They are excluded from the POI context judgement rather than penalized by it.
5. Block 1 has zero **modelled** audience despite 8,544 real bookings — a gap in the model,
   not a fact about block 1. The 8%-of-block-6 figure is an explicit assumption and is kept
   out of every measured total.
6. No ambient/pedestrian audience term. `nearby_ambient_footfall` exists but is quarantined
   to a tie-break.
12. The candidate pool is a relevance truncation, and truncation can be categorical: a whole
    screen type can sit below another's floor (`bus_stop` 0.5891 vs `metro_station` 0.6066 —
    2.9% — and 0 of 250 kept). `CampaignSpec.screen_type_mix` stratifies the cut;
    `allowed_screen_types` is a filter and cannot produce a mix.
13. A corridor's pool population is built from scheduled ridership while a stop's audience is
    built from observed actuals. The two reconcile to ~0.97x, and
    `corridor_pool_sanity()` asserts the ordering invariant between them.
7. Events affect pricing, not demand.
8. Booking probability is highly saturated because the historical win rate is ~99.79%;
   it is diagnostic, not price-setting.
9. Seasonality currently adjusts price, not demand.
10. No true conversion model exists; conversion uses a POI-context proxy.
11. No flighting model exists; duration is fixed by the brief.
12. Historical bundle selection is not modeled.
13. Candidate top-N selection is not coverage-aware.
14. No confidence is emitted because there are no held-out error bars.
15. `budget_sensitivity` is not currently wired.
16. The pricing impressions proxy is quarantined and must not become demand/reach.
17. Fixed/mobile audience units have different modelling paths; preserve the explicit
    `pool_partition_count` logic.
18. `merit` (§9.5) has no held-out accuracy metric and cannot acquire one from this data —
    there is no ground truth for what a screen is worth. Validate it forward by tracking
    whether premium-flagged screens keep their occupancy.
19. The demand premium is restricted to fixed inventory. Mobile screens have no zone
    demographics, so 3 of 4 merit components are structurally zero for them.
20. `negotiation_leverage` tracks account size, not price behaviour. It is population
    context and must never be presented as a per-deal forecast.
21. Client price indices are measured against segment medians that include the client's own
    bookings, which pulls the index toward 1.0. Conservative, uncorrected, deliberate.
22. `price_gap_pct` is not populated on every price-driven lead. A missing value means "not
    recorded", never 0%.
23. `duration_days` was tested as a price-band dimension and rejected (~2% between the
    buckets most campaigns fall in, non-monotone on thin cells). Do not re-add it on
    intuition.

---

# 22. Non-negotiable design principles

### 1. Do not over-agentify
Use agents only where reasoning/delegation is valuable. The relevance engine is a
Master-owned deterministic tool.

### 2. LLMs reason; tools calculate
No LLM numerical optimization, audience aggregation, pricing arithmetic, or constraint
checking.

### 3. Preserve units
Always distinguish:

```text
4-hour time block
6 rotating slots
people passing
people who look
viewed exposures
distinct reach
```

### 4. Reach is never exposures
Deduplicate by `pool_key`; cap by reachable pool population.

### 5. Slots are linear
All six slots rotate continuously through the same block. Buying more slots increases
share of voice; it does not make the loop run faster.

### 6. Hard constraints are deterministic
Budget, inventory, geography, dates, screen count, required blocks, and declared hard
constraints cannot be “explained away”.

And a constraint the brief declares must be **read from the run and independently
re-derived by validation**, or it is not enforced at all. A declared input qualifies for a
validator check in a way an assumed constant like `REACH_LAMBDA` does not (principle 8b), and
without the re-derivation a miss is silent: a slot-depth constraint was dropped end to end
while intake recorded it, the run persisted it, the Master echoed it back, and verification
passed the breaching package. `hard_constraints` keys therefore live in the closed
`ENFORCED_HARD_CONSTRAINTS` vocabulary, and one outside it fails verification rather than
being ignored.

### 7. Do not invent constraints
In particular, do not introduce a minimum budget utilization unless the campaign spec
explicitly declares one.

### 8. Keep artifacts structured
Use Pydantic contracts and artifact references.

### 9. Keep calculations centralized
Exposure conversion belongs in one module. Reach accounting has one reporting definition
and an independent validation implementation.

### 10. Assumptions must be visible
A number affected by an assumed constant must identify that assumption.

### 11. Models are replaceable
Keep business artifact contracts stable when replacing ML/OR internals.

### 12. Validate independently
The Master must verify specialist output before final recommendation.

### 13. Infeasibility must be explicit
Return a structured infeasibility report rather than a plausible-looking package.

### 14. Build deterministic systems first
Agent orchestration must sit on top of working analytical components.

---

# 23. Canonical tool/file ownership

```text
app/agents/
  master.py
  ml_agent.py
  or_agent.py
  validation.py

app/tools/
  master_tools.py
  relevance_tools.py
  ml_agent_tools.py
  or_agent_tools.py

app/ml/
  occupancy.py
  price_band.py
  booking_probability.py
  seasonality.py
  price_optimizer.py
  demand_value.py       # merit vs realized price -> mispricing premium
  client_profile.py     # advisory only; no engine, no lever default
  levers.py             # the agent-tunable parameter surface
  engine.py             # process singleton
  impressions.py        # quarantined diagnostic only
  loaders.py

app/optimize/
  config.py
  exposure.py           # ONE people→viewed-exposure conversion
  contract.py
  pooled.py             # diagnostic saturation curve
  solver.py

app/data/
  db.py
```

When changing a rule, update the owning module and its contract/tests rather than
duplicating the logic in prompts or downstream agents.

---

# 24. Canonical end-to-end workflow

```text
1. Master parses user brief.
2. Create and validate CampaignSpec.
3. Resolve geography.
4. Run deterministic relevance engine.
5. Store ScreenCandidates artifact.
6. ML Agent prices candidates and checks availability.
7. Store ScreenEconomics artifact.
8. OR Agent solves MILP.
9. Return OptimizedPackage or InfeasibilityReport.
10. Master runs deterministic validation.
11. If valid, generate sales recommendation.
12. Explain:
      - selected screens
      - time blocks
      - price
      - gross viewed exposures
      - distinct reach
      - frequency
      - budget utilization
      - risks
      - alternatives
13. Never recompute or reinterpret numerical metrics in prose.
```

The core product promise is:

```text
Natural language
    ↓
Structured intent
    ↓
Data-driven audience understanding
    ↓
Inventory relevance
    ↓
Audience volume
    ↓
Pricing + availability
    ↓
Mathematical optimization
    ↓
Independent validation
    ↓
Explainable sales recommendation
```

This document is intentionally compact. When implementation details conflict with this
document, preserve the **business definitions, units, ownership boundaries, artifact
contracts, and validation rules** first; then update this document to reflect the new
source of truth.

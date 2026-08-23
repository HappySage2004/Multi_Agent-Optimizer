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
- **ML Agent:** pricing, availability, seasonality, booking-probability diagnostics.
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
  They therefore receive a scoring floor rather than a measured demographic score.
- `daytime_population_multiplier` is carried but not currently used by relevance scoring.
- Event features are not part of the audience profile. Events affect pricing seasonality only.
- Volume is schedule/ridership-derived; there is no pedestrian/ambient term.
- Time block 1 (00:00–04:00) therefore reports zero audience even though real bookings exist.
  **Zero means “not modelled”, not “nobody there”.**

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
income_score        = min-max income_index
professional_score  = 0.7*income_score + 0.3*white_collar_flag
young_adult_score   = min-max pct_age_18_34, scaled 0.85
student_score       = 0.6*young_adult_score + 0.4*university_nearby
family_score        = (min-max pct_35_54 + 0.25*min-max pct_18_34) / 1.25
commuter_score      = peak-block impressions / total impressions
```

Every sub-score either computes a real value or falls back to **0.5 with an explicit
reason**. Defaults must never be hidden.

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
- Naively summing per-screen exposures over-counted the audience by ~23x on a realistic
  candidate pool.

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

Fallback ladder:

```text
A: screen_size × screen_type × position × city × daypart, n >= 30
B: screen_size × screen_type × position × city, n >= 30
C: screen_size × screen_type × position, final floor
```

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

Current rule:

```text
1. Check availability.
2. Get price band: floor / target / cap.
3. Apply seasonality/event multiplier.
4. recommended_price =
       floor + avg_occupancy_rate × (cap - floor)
5. Report booking probability as a diagnostic.
```

Price is flat across slot counts by design. `price_by_slot_count` maps 1..6 to the same
price where available and null where unavailable.

Seasonality currently adjusts **price**, not demand. Event matching uses exact anchor
location or a damped city-zone match because there is no lat/lon data.

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
```

## 11.5 Elastic constraints

Penalized and reported rather than automatically failing:

```text
coverage groups
wear-out frequency cap
```

If a hard constraint forces an elastic breach, return the package and disclose the breach.

**Never invent a minimum budget-utilization requirement.**

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
resolve_geography_terms(terms)
create_campaign_spec(...)
get_run_state(run_id)
verify_package(run_id)
inspect_package(run_id, limit=10)
check_explanations(run_id, explained_screen_ids)
```

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
optimize_package(run_id, slots_per_day_cap=3)
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
```

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
all scores ∈ [0,1]
12 impression columns exist and contain no NaNs
pool_key never null
block 1 remains the known zero-audience gap
all audience terms map to real score columns/blocks
geography is hard-filtered
allowed screen types are enforced
different audience terms can change blocks and ranking
unknown audience terms are rejected
missing data is recorded in defaults_applied
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
floor <= target <= cap
industry adjustment within clamp
every screen attribute combination has a band
recommended price lies within band
availability never exceeds capacity
tightest day defines availability
infeasible rows retained
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
```

Also protect the known adversarial case:

```text
A package claiming gross exposures as reach MUST fail validation.
```

---

# 21. Known limitations — do not silently “fix” these by changing definitions

The following are known model limitations and should remain explicit until the underlying
data/model changes:

1. `LOOP_PASSES_PER_TRIP` and `VIEWABILITY_*` are assumptions without ground truth.
2. `REACH_LAMBDA` is diagnostic-only and not used for reported reach.
3. Demand has no held-out accuracy metric or confidence interval.
4. Vehicle-mounted screens lack demographic data and currently receive a scoring floor.
5. Block 1 has zero modelled audience despite real bookings.
6. No ambient/pedestrian audience term.
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
  impressions.py       # quarantined diagnostic only
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

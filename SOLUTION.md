# SOLUTION.md --- Transit Media Campaign Recommendation System

## Purpose

This document is the implementation handoff for the Transit Media
campaign recommendation system.

The system should accept a natural-language campaign brief, understand
the business requirements, identify relevant transit-media inventory,
forecast demand, estimate pricing, optimize the available inventory
under business constraints, and return a sales-ready recommendation with
explanations and validation.

The architecture deliberately separates:

-   LLM-based reasoning and orchestration
-   deterministic data/feature tooling
-   ML prediction
-   mathematical optimization
-   final recommendation generation

The system should **not over-agentify** the workflow. The original design
had one Master Deep Agent supervising three specialists:

1.  **Data Intelligence Agent**
2.  **ML / Forecasting Agent**
3.  **OR / Optimization Agent**

**As built there are two specialists, not three.** The Data Intelligence
Agent was removed: relevance scoring turned out to be a wholly
deterministic engine, and wrapping it in an LLM shell added latency and a
chance to paraphrase its numbers wrongly without adding any judgement. It
is now a Master-owned tool (`app/tools/relevance_tools.py`). See section
16.

The Master Agent owns orchestration, verification, final response
generation, and stage 2.

------------------------------------------------------------------------

# 0. Implementation Status

This document is both the design intent and the record of what exists. It
evolves as capabilities land. Sections carry one of four markers:

  ------------------------------------------------------------------------
  Marker             Meaning
  ------------------ -----------------------------------------------------
  **[BUILT]**        Implemented and exercised by the test suite. The text
                     describes what the code actually does.

  **[HEURISTIC]**    Computes a real answer from real data and honours every
                     constraint, but makes no optimality claim.
                     **Nothing carries this marker any more.**

  **[STUB]**         A contract-shaped placeholder runs today so the
                     pipeline is end-to-end. The text is the target design.
                     **Nothing carries this marker any more.**

  **[NOT BUILT]**    Nothing runs. The text is the target design and the
                     wiring instructions.
  ------------------------------------------------------------------------

## Current state

  --------------------------------------------------------------------------
  Capability                            Status            Where
  ------------------------------------- ----------------- ------------------
  Brief intake + geography resolution   **[BUILT]**       `app/tools/master_tools.py`

  Audience & context features           **[BUILT]**       `v_screen_profile`

  Relevance scoring                     **[BUILT]**       `app/tools/relevance_tools.py`

  Demand forecasting                    **[BUILT]**       `v_screen_demand_history`
  (impressions / reach / frequency)

  Availability & occupancy              **[BUILT]**       `app/ml/occupancy.py`

  Market price band                     **[BUILT]**       `app/ml/price_band.py`

  Booking probability                   **[BUILT]**       `app/ml/booking_probability.py`

  Seasonality & event adjustment        **[BUILT]**       `app/ml/seasonality.py`

  Price recommendation                  **[BUILT]**       `app/ml/price_optimizer.py`

  Inventory optimization                **[BUILT]**       `app/optimize/` +
                                                          `app/tools/or_agent_tools.py`

  Validation layer                      **[BUILT]**       `app/agents/validation.py`

  Master orchestration + 2 subagents    **[BUILT]**       `app/agents/`
  --------------------------------------------------------------------------

**Nothing carries [HEURISTIC] any more either.** The greedy value-per-dollar fill was
replaced by a MILP (HiGHS via `scipy.optimize.milp`) in `app/optimize/`, ported from the OR
handoff bundle. The marker is kept in the legend because it is the honest label for
anything that computes a real answer from real data without an optimality claim, and
because a solve that stops inside its gap is reported as `feasible` rather than
`optimal` — a distinction the package carries in `optimization_method`.

## The one thing to know

**Reach is never the sum of exposures.**

Screens at the same stop, or on the same corridor, see the *same people*. The
audience engine tags each screen with a `pool_key` for exactly this reason. On a
realistic 250-screen candidate pool, naively summing per-screen exposures
over-counts the audience by **~23x**.

The definition this system reports:

``` text
reach = SUM over (pool_key, time_block) of
            min( gross viewed exposures bought in that group,
                 that pool's reachable daily audience )
```

`gross_impressions_viewed` is exposures and scales with slots x days.
`expected_reach` is distinct people and **saturates** --- buying more slots, more
days, or more screens at the same stop raises frequency, not reach. The min()
also guarantees `reach <= exposures` for any flight length.

Both sides are in **viewed** units. Capping viewed exposures at the undiscounted crowd
would let a saturated plan claim every passer-by when only ~35% of them look --- an
over-claim of ~2.9x on the one number a client actually reads.

It is computed in `or_agent_tools._package_metrics`, recomputed independently in
`validation._reach_checks`, and --- since the MILP landed --- it is also what the solver
maximizes: `min()` of two linear functions is concave, so `R <= min(E, P)` is exactly two
linear constraints with no free parameter (section 10). Three implementations, one
definition, no fitted constant anywhere in it.

A **second** saturation model exists and is deliberately not that definition. The handoff
bundle bounded `P x (1 - exp(-lambda x E / P))` by its tangent lines. `REACH_LAMBDA` is
ASSUMED with no ground truth in the 14 CSVs, so it belongs in neither a validated nor a
client-facing figure --- and maximizing it does not maximize the reported reach: measured
141,501--157,869 reached where the exact bound returns 261,329 on the same brief and
budget. It survives as `curve_reach_diagnostic`, guarded by
`curve_reach <= min(sum E, sum P)` --- a bound that holds for any lambda, which is why it
is worth asserting.

The second thing to know: **a zero audience figure means "not modelled", not
"nobody there".** Volume is derived entirely from scheduled transit service with
no pedestrian or ambient term, so time block 1 (00:00-04:00) reports zero for
every screen in the network --- even though that block has 8,544 real bookings.

------------------------------------------------------------------------

# 1. End-to-End Architecture

The logical pipeline, with current status per stage:

``` text
                         ┌─────────────────────┐
                         │   Campaign Brief    │
                         │       / Query       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  1. Brief Intake &  │  [BUILT]
                         │     Normalization   │
                         └──────────┬──────────┘
                                    │
                           CampaignSpec
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │ 2. Audience & Context       │  [BUILT]
                    │    Intelligence             │
                    └─────────────┬───────────────┘
                                  │
                         ScreenProfiles
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 3. Campaign-Screen          │  [BUILT]
                    │    Relevance Scoring        │
                    └─────────────┬───────────────┘
                                  │
                      RankedScreenCandidates
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 4a. Demand Forecast         │  [BUILT]
                    │     impressions / reach     │
                    ├─────────────────────────────┤
                    │ 4b. Pricing + Availability  │  [BUILT]
                    │     band / occupancy / P    │
                    └─────────────┬───────────────┘
                                  │
                       ScreenEconomics
                       pricing + audience
                       volume populated
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 5. Inventory / Slot         │  [BUILT]
                    │    Optimization (MILP)      │
                    └─────────────┬───────────────┘
                                  │
                       OptimizedPackage
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 6. Recommendation &         │  [BUILT]
                    │    Explanation              │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                         Sales Recommendation
```

Stage 4 is one business capability but two independent halves, deliberately
decoupled so pricing could ship without demand. 4b answers *what should this
cost and is the slot actually free*; 4a answers *how many people will see it*.
Both exist now, but they are owned in different places: 4b is the pricing engine
in `app/ml/`, while 4a is computed upstream by the relevance engine and only
*unit-converted* by the ML stage. The decoupling held.

The six logical stages are implemented by two specialist agents plus
deterministic tools --- stages 1, 2, 5 and 6 are Master-owned tool calls with no
delegation at all. Do not create six independent LLM agents solely because there
are six business stages.

------------------------------------------------------------------------

# 2. Central State Object --- CampaignSpec  **[BUILT]**

Everything downstream should consume a normalized campaign
specification.

The user's input may be vague or conversational. The first stage
converts it into a structured object.

Example input:

> "I have a \$50K budget for 30 days and want to reach young commuters
> around the eastern metro corridor."

Example normalized object:

``` json
{
  "campaign_objective": "reach",
  "industry_vertical": "consumer_tech",
  "ad_type": "product_launch",
  "target_audience": {
    "age_range": [18, 34],
    "commuter": true
  },
  "geography": {
    "city_ids": ["LH"],
    "zone_ids": ["LH-ZONE-010"],
    "corridor_ids": ["LH-RT-B001"]
  },
  "start_date": "2026-09-01",
  "duration_days": 30,
  "budget": 50000,
  "requested_num_screens": null,
  "preferred_dayparts": [],
  "preferred_time_blocks": [],
  "optimization_goal": "reach",
  "hard_constraints": {},
  "soft_preferences": {}
}
```

Implement this as a Pydantic model.

``` python
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
    audience_terms: list[str] = []       # closed vocabulary, see below

    start_date: date
    duration_days: int
    budget: float

    requested_num_screens: int | None = None

    preferred_dayparts: list[str] = []
    preferred_time_blocks: list[str] = []
    day_type_focus: Literal["weekday", "weekend"] | None = None

    optimization_goal: Literal[
        "reach",
        "frequency",
        "awareness",
        "conversion"
    ]

    hard_constraints: dict = {}
    soft_preferences: dict = {}
```

Requirements:

-   Validate budget \> 0.
-   Validate duration \> 0.
-   Validate dates.
-   Validate referenced city/zone/corridor IDs.
-   Normalize synonyms.
-   Resolve natural-language geography into IDs.
-   Preserve the original user query for traceability.
-   Do not invent missing hard constraints.
-   Reject `audience_terms` outside the closed vocabulary.

### audience_terms --- a closed vocabulary  **[BUILT]**

The relevance engine scores against six named segments and nothing else:

``` text
young_professionals   professionals   students
families              high_income     commuters
```

Intake's LLM picks from that list; `CampaignSpec` **rejects** anything else in
code rather than scoring it. An invented segment would not error on its own --- it
would silently collapse the 0.40-weighted audience component to a neutral 0.5 for
every screen, which looks like a working answer and is not one. `AudienceTarget`
(age range, income, occupations, commuter flag) is still captured for traceability;
`audience_terms` is what the engine actually consumes.

`day_type_focus` exists because weekday and weekend ridership differ by roughly
6x. It affects **scoring only** --- the flight still runs every day in its window,
and the economics stage weights the real calendar mix regardless.

------------------------------------------------------------------------

# 3. Step 1 --- Campaign Brief Intake  **[BUILT]**

## Purpose

Convert natural-language input into `CampaignSpec`.

## Inputs

``` text
User query
Optional client context
Optional uploaded campaign documents
Optional previous conversation state
```

## Potential data dependencies

-   `client_facts`
-   `cities`
-   time-slot dimension data

The client facts contain information such as typical campaign budget,
campaign frequency, average duration, preferred geographies, bundle
affinity, negotiation leverage, and account status.

## Processing

The LLM should:

1.  Extract campaign objective.
2.  Extract industry.
3.  Extract ad type.
4.  Extract audience.
5.  Extract geography.
6.  Extract dates/duration.
7.  Extract budget.
8.  Extract screen requirements.
9.  Extract daypart/time preferences.
10. Identify hard constraints.
11. Identify soft preferences.
12. Identify missing information.
13. Normalize the result.
14. Pass the result to a deterministic validator.

## Output

``` text
CampaignSpec
```

The LLM is useful here because the input is unstructured.

The validator must remain deterministic.

------------------------------------------------------------------------

# 4. Step 2 --- Audience & Context Intelligence  **[BUILT]**

Implemented as the DuckDB view `v_screen_profile` (`app/data/db.py`), consumed by
`app/tools/relevance_tools.py`. One row per screen for all 11,163: geography,
zone demographics, POI context, and `pool_key`.

`app/features/` stayed empty --- the engine and its tools live together in
`app/tools/relevance_tools.py` by design decision, so the whole audience
capability reads in one place.

Do not implement this as a giant LLM-only "Audience Profile Agent." There is no
LLM in this stage at all.

Instead use:

``` text
Data processing
    +
Feature engineering
    +
ML/scoring logic
    +
Optional LLM interpretation
```

The primary output is a screen-level feature table represented
conceptually as `ScreenProfile[]`.

### What was actually built, and what was not

`v_screen_profile` carries: `city_id`, `zone_id`, `corridor_id`,
`location_type`, `screen_type`, `screen_size`, `position`, `inventory_class`,
`pool_key`, `city_zone`, `zone_name`, `resident_population`,
`population_density_per_sqkm`, `median_age`, `pct_age_18_34`, `pct_age_35_54`,
`median_household_income`, `income_index`, `pct_bachelor_or_higher`,
`dominant_occupation`, `daytime_population_multiplier`, `num_nearby_pois`,
`weighted_nearby_footfall` (inverse-distance weighted, `1/(km + 0.1)`),
`closest_poi_distance_km`, `nearby_poi_types`, `pool_partition_count`.

`pool_partition_count` was added for the optimizer: 1 for stop-mounted screens, and
the vehicles working the corridor for vehicle-mounted ones. `v_screen_demand_history`
divides a corridor's ridership by that count to get one vehicle's share, so the
optimizer needs it to recover the pool's whole crowd for the reach ceiling ---
otherwise it under-buys in-vehicle inventory by exactly that factor. The divisor is
defined once, in `v_corridor_vehicle_count`, and used by both views so they cannot
drift; the reconstruction is verified exact against the corridor totals.

Not built: **event features**. `events` is not joined into the profile, so
`nearby_events` and `event_attendance` below have no counterpart. Events do reach
the pipeline, but only through the pricing engine's seasonality module.

Also not built: `daytime_population_multiplier` is carried but unused by any
score, despite being an obvious signal for a daytime commuter brief.

**Mobile screens have no demographics.** A vehicle has no zone by construction,
so all 2,615 vehicle-mounted screens hold NULL for every demographic column and
score 0 --- a floor, not a measurement. Averaging the zones a corridor touches
(`v_corridor_zones` already resolves them) is the obvious next improvement.

## ScreenProfile

Example:

``` json
{
  "screen_id": "LH-SCR-000123",

  "location": {
    "city_id": "LH",
    "zone_id": "LH-ZONE-005",
    "location_type": "bus_stop"
  },

  "screen_attributes": {
    "screen_type": "bus_stop",
    "position": "left",
    "screen_size": "M"
  },

  "audience": {
    "population": 107560,
    "density": 14480,
    "median_age": 32.6,
    "pct_age_18_34": 39.0,
    "median_income": 134615,
    "daytime_population_multiplier": 3.39
  },

  "transit": {
    "routes_serving": 4,
    "estimated_daily_ridership": 12000
  },

  "context": {
    "poi_footfall": 29380,
    "nearby_events": 2,
    "event_attendance": 45000
  }
}
```

## Relevant source tables

Use the table join map to construct these features.

Primary relationships:

``` text
screens
    -> locations
    -> cities
    -> zone_demographics

screens
    -> vehicles
    -> route_schedules
    -> route_stops
    -> ridership_actuals

locations
    -> points_of_interest
    -> events

bookings
    -> screens
    -> clients
    -> time blocks
```

The demographics data contains:

-   resident population
-   population density
-   median age
-   age bands
-   median household income
-   income index
-   education
-   dominant occupation
-   daytime population multiplier

The POI data contains:

-   POI type
-   scale
-   estimated daily footfall
-   distance to location
-   network hub status
-   side of road
-   peak daypart

Events provide:

-   event type
-   expected attendance
-   attendance tier
-   primary impact daypart
-   impact radius

## Feature engineering requirements

Create reusable deterministic feature builders:

``` python
build_screen_profiles()
build_demographic_features()
build_transit_features()
build_poi_features()
build_event_features()
build_screen_context_features()
```

Do not make the LLM calculate these values.

------------------------------------------------------------------------

# 5. Step 3 --- Campaign-to-Screen Relevance Scoring  **[BUILT]**

`app/tools/relevance_tools.py`. Deterministic, Master-owned, no subagent.

This stage identifies promising inventory before expensive optimization.

Do not send the entire inventory universe into the optimizer.

Recommended funnel:

``` text
All screens
    ↓
Hard filtering
    ↓
Feasible screens
    ↓
Relevance scoring
    ↓
Top ~100–300 candidates
    ↓
Optimization
```

The exact candidate count should be configurable.

## Inputs

``` text
CampaignSpec
ScreenProfile[]
```

## Output

``` python
class ScreenCandidate(BaseModel):
    screen_id: str

    relevance_score: float

    # the five weighted components
    audience_match_score: float
    geography_score: float
    contextual_score: float
    time_of_day_score: float
    historical_performance_score: float

    # reported, NOT weighted into relevance_score
    transit_score: float                 # volume percentile in the eligible pool

    reasons: list[str]
    defaults_applied: list[str]          # sub-scores that fell back to 0.5, and why

    hard_constraints_passed: bool

    # audience volume -- PEOPLE PASSING, not viewed exposures
    pool_key: str | None                 # the reach unit
    pool_partition_count: int            # 1 for stop-mounted; vehicles on the
                                         # corridor for vehicle-mounted, so the
                                         # optimizer can recover the pool's whole
                                         # crowd from this screen's share
    impressions_by_block: dict[str, float]   # 12 keys, "{block}_{weekday|weekend}"
    impressions_weekday: float
    impressions_weekend: float
```

Two fields carry the load downstream. `pool_key` is the physical-audience unit
(location for stop-mounted screens, corridor for vehicle-mounted), without which
reach cannot be computed. `impressions_by_block` is **whole-block daily** traffic
for that pool --- not per slot, not per campaign; the ML stage converts it.

Example:

``` json
{
  "screen_id": "LH-SCR-001928",
  "relevance_score": 0.91,

  "audience_match_score": 0.94,
  "geography_score": 0.97,
  "contextual_score": 0.83,
  "transit_score": 0.89,

  "reasons": [
    "High 18-34 population",
    "High daytime population",
    "Strong evening commuter traffic",
    "Located within requested corridor"
  ],

  "hard_constraints_passed": true
}
```

## Scoring implementation  **[BUILT]**

A transparent weighted sum, as recommended --- no supervised model, because there
are no relevance labels in this dataset to fit one against.

``` text
relevance_score =
    0.40 * audience_similarity      demographic score columns for the audience terms
  + 0.20 * geographic_fit           graded match inside the eligible pool
  + 0.15 * context_fit              industry -> POI type match
  + 0.15 * time_of_day_fit          share of traffic in the audience's target blocks
  + 0.10 * historical_performance   completion rate of past bookings, same vertical
```

`transit_score` (volume percentile) is reported but **excluded** from the sum on
purpose. Volume is the optimizer's objective quantity; folding it into relevance
would rank a busy screen as a better *match* than a quiet one sitting in exactly
the right zone.

Audience score columns, min-max normalized once over the **whole** inventory at
engine build so they stay comparable between campaigns:

``` text
income_score        min-max of zone income_index
professional_score  0.7 x income_score + 0.3 x white_collar_flag
young_adult_score   min-max of pct_age_18_34, scaled 0.85
student_score       0.6 x young_adult_score + 0.4 x university_nearby
family_score        (min-max pct_35_54 + 0.25 x min-max pct_18_34) / 1.25
commuter_score      peak-block impressions / total impressions
```

The `/ 1.25` on `family_score` is a deliberate change from the source notebook:
the raw sum peaks at 1.25 (measured 1.140 on this inventory) and broke the
`ScreenCandidate` 0-1 bound. Dividing by the ceiling preserves the ordering;
clamping would have flattened the top of the distribution.

### Geography is a HARD filter

`eligible_screen_ids()` runs **before** scoring. This is not the source design's
soft geography penalty, and the change was required: the validation layer fails
any package containing an ineligible screen, so a soft penalty produced candidate
pools that could not pass verification (measured 109 of 250 outside the requested
zone on the canonical brief).

Inside the eligible pool, `geographic_fit` grades *how well* a screen matches:

``` text
1.0  exact match on a requested zone or corridor, or a city-wide brief
0.8  mobile screen whose corridor passes through a requested zone --- it serves
     the area without being sited in it
0.6  right city, but a finer geography was requested and this is not in it
0.0  unreachable after the hard filter; kept as a guard
```

The 0.8 tier exists because `eligible_screen_ids` admits mobile inventory via
corridor-touches-zone, and scoring those screens 0.6 alongside genuinely
out-of-area ones would have made all 2,615 of them uncompetitive.

### Every default is reported, never hidden

Each sub-score returns either a computed value or a neutral 0.5 plus a reason.
Pool-wide fallbacks land in the artifact summary's `defaults_applied`; per-screen
ones (no booking history in the vertical; no demographics for a mobile screen) land
on that row's `defaults_applied`. A 0.5 that nobody can trace is worse than a gap.

The score is explainable: `reasons` cites real feature values --- zone age bands
and income index, POI count and weighted footfall, riders/day in the target blocks
with the share of the screen's traffic, past completion rate with its sample size,
and how many other screens share the audience pool.

------------------------------------------------------------------------

# 6. Step 4 --- Demand Forecasting & Pricing

Expose this as one business capability, internally separated into four
independent pieces. All four exist, but they are not all owned here:

``` text
Demand Forecast          [BUILT]   section 7   <-- owned UPSTREAM, by the
                                                   relevance engine. This stage
                                                   only converts its units.
Availability / Occupancy [BUILT]   section 7A
Market Price Band        [BUILT]   section 8, Model A
Booking Probability      [BUILT]   section 8, Model B
```

As-built architecture (`app/ml/`, assembled by `PricingEngine`):

``` text
                    PricingEngine  (process singleton, ~12s build)
                              │
        ┌─────────────┬───────┴───────┬──────────────┐
        ▼             ▼               ▼              ▼
   Occupancy     Price Band     Booking Prob.   Seasonality
   (M1)          (M2)           (M3)            (M6)
   bookings      bookings       bookings vs     ridership DOW
   day-by-day    p25/p50/p90    lost_leads      + events
        │             │               │              │
        └─────────────┴───────┬───────┴──────────────┘
                              ▼
                      Price Optimizer (M4)
                      feasibility gate, then
                      floor + occupancy x (cap - floor)
                              │
                              ▼
                       ScreenEconomics
                              │
                    ┌──────────┴────────────────────┐
                    │ viewed_exposures_per_slot_per │ <-- audience volume,
                    │ _day                          │     mapped in from the
                    │ daily_unique_audience         │     relevance engine via
                    │ reachable_daily_audience      │     app/optimize/
                    │ pool_key, demand_forecast     │     exposure.py
                    └───────────────────────────────┘
```

The engine is a **process-wide singleton** (`get_pricing_engine()`). Build
costs ~12s --- loading bookings, indexing occupancy, computing the band
groupbys, and fitting the probability model. Never construct it per request.

A separate module, `app/ml/impressions.py` (M5), computes a
`pricing_internal_reach_proxy`. It is **not** demand and is deliberately not
wired to any exposure field. See section 7B.

------------------------------------------------------------------------

# 7. Demand Forecasting  **[BUILT]**

Built as part of the audience relevance engine, not as a separate model. The
DuckDB view `v_screen_demand_history` gives average daily riders per screen, per
`dim_slot` time block, per day type (weekday/weekend) --- 111,630 rows over
11,163 screens.

## The two join paths

They are structurally different, and the difference matters:

``` text
STOP-MOUNTED (8,548 screens)
  screen -> location -> route_stops -> route -> schedules -> ridership_actuals
  actual_ridership is PER TRIP. A screen at a stop is passed by EVERY trip in
  the block, so:
      1. SUM across trips within each (route, block, day_type, date)
      2. MEAN across dates
      3. SUM across every route serving that stop
  Averaging at step 1 instead would describe a single vehicle rather than the
  volume passing the stop, understating it by roughly the trip count (~42x).

VEHICLE-MOUNTED (2,615 screens)
  screen -> vehicle -> corridor -> schedules
      corridor block total / vehicles working that corridor
  A screen inside ONE vehicle sees only the trips THAT vehicle makes, and
  route_schedules carries no vehicle_id --- so one vehicle's share is the
  available approximation. This is a stated modelling judgement, not a
  settled correction like the stop-mounted case.
```

**Consequence to keep in view:** `metro_station` median daily volume is ~380x
`bus`. The gap is directionally real --- a station concourse is not one bus --- but
large enough that any impressions-per-dollar ranking picks fixed inventory almost
exclusively. Fixing the mobile unit is the highest-value next step on this model.

## Degradation and provenance

`ridership_actuals` is gitignored and optional. Without it the stop-mounted path
falls back to `route_schedules.estimated_ridership` --- the same quantity the
corridor path already uses, so units stay consistent, at lower fidelity. The
engine reports which source is live in `demand_source`, and it is echoed on the
artifact summary.

**Data constraint that shaped the design.** `ridership_actuals` spans 2026-02-19
to 2026-08-19 --- all in the past relative to any campaign window, with exactly two
holiday dates. So the model deliberately works at **day-type granularity, not
calendar date**: day-of-week, time-block and corridor features generalize to a
future flight; date-keyed features do not. That is also why `DemandForecast` (the
per screen/date/time-block contract in section 9) is still unpopulated --- there is
nothing honest to put in a per-date row.

## The unit chain --- three quantities, easily confused

``` text
v_screen_demand_history.daily_impressions
    riders PASSING a screen's POOL during a 4-hour block, on a typical day
    of that day type
        |
        v  weight by the flight's real weekday/weekend day counts
ScreenCandidate.impressions_by_block
    whole-block daily PEOPLE PASSING, per day type, 12 columns
        |
        +--> x viewability(screen_type)
        |    ScreenEconomics.reachable_daily_audience
        |        distinct people who LOOK -> the reach CEILING. Does NOT scale
        |        with slots or days.
        |
        +--> x LOOP_PASSES_PER_TRIP / 6 x viewability(screen_type)
             ScreenEconomics.viewed_exposures_per_slot_per_day
                 viewed exposures ONE slot earns on ONE day -> scales with
                 slots x days
```

Every step of that conversion lives in exactly one module,
`app/optimize/exposure.py`, called from exactly one place
(`ml_agent_tools._to_contract`). The constants are ASSUMED, so the single call site
matters more than the values: a second copy would let two stages disagree about what an
"impression" is without either being wrong on its own.

A time block is a 4-hour window in which all 6 rotation slots cycle continuously
(1->2->...->6->1). Slot POSITION is meaningless --- `slots_booked_per_day` is share of
voice, never which positions --- so holding k of 6 slots puts the creative on k of every 6
loop passes and viewed exposures are strictly **linear** in k. The only concavity in this
system is at the audience pool.

**The dwell assumption, stated plainly.** `LOOP_PASSES_PER_TRIP = 8` says a person in range
of a screen sees eight rotations of the loop. An earlier version of this document divided by
6 and stopped there, which is the same model with `LOOP_PASSES_PER_TRIP = 1`: exposure
proportional to airtime, dwell short relative to one rotation. Neither is measurable --- there
is no dwell data anywhere in the 14 CSVs, the same gap that left section 7B's `dwell_factor`
hand-set. The 8-pass form was adopted because the alternative labels `footfall x 8` as
"impressions" at a viewability of 1.0, an 8x over-count of people, and because a wear-out
judgement is only meaningful in viewed units.

**The consequence to state, not bury.** At 8 loop passes, one slot on a saturated pool
delivers `8/6 = 1.33` viewed exposures per person per day, so a 30-day flight cannot deliver
fewer than **~40 exposures per person reached** no matter what the optimizer chooses.
Duration is the brief's, the minimum purchase is one slot, and there is no flighting. That is
why the wear-out cap in section 10 is a multiple of that floor rather than an absolute
number, and why it is a property of flight length rather than of selection.

**A provenance note, because it was cited in support of these constants and does not hold.**
`or_engine/config.py` justifies them with "their metro_station median of 227,981 x 0.2275 =
51,866 against our independently derived 48,706". 0.2275 is 0.35 x 0.65 --- the static and
in-vehicle factors multiplied together. `metro_station` is static, so that comparison should
have used 0.35, giving 79,793 against 48,706, a 1.6x miss. The cross-check therefore
validates a compound factor that is not shipped, and neither shipped constant is evidenced
by it. They are held on the two arguments above, not on that arithmetic.

## Not built

-   **Accuracy metrics.** No MAE/RMSE/MAPE, no comparison against the naive
    `route_schedules.estimated_ridership` baseline. The model is an aggregation of
    observed ridership rather than a fitted predictor, so there is no held-out
    split --- but that also means there is no measured error bar, which is why no
    stage emits a per-screen confidence and the validator still skips that check
    (section 18). This is the same gap section 26 flags for pricing.
-   **Prediction intervals.** `DemandForecast.lower_bound` / `upper_bound` have no
    counterpart.
-   **Any ambient/pedestrian term.** Volume is schedule-derived only, so a block
    with no scheduled service reports exactly zero. Time block 1 (00:00-04:00) is
    zero for all 11,163 screens despite 8,544 real block-1 bookings. A zero here
    means "not modelled".
-   **Event and POI features in the volume model.** `weighted_nearby_footfall` is
    computed and carried on the profile, but is not added into any impressions
    figure. Events reach the pipeline only through pricing seasonality.

## What a fitted model would add

The section below is the original target design, retained because a supervised
model on top of these aggregates is still the right next step --- it would give
the error bars and the confidence figure this one cannot.

Relevant tables:

``` text
route_schedules
ridership_actuals
route_stops
vehicles
screens
events
points_of_interest
```

The schedule data provides:

-   schedule ID
-   route ID
-   corridor
-   direction
-   day type
-   start time
-   estimated ridership

The historical ridership data provides:

-   schedule ID
-   route ID
-   date
-   day of week
-   holiday flag
-   actual ridership

## Feature pipeline

``` text
screen
  ↓
location
  ↓
routes serving location
  ↓
schedule
  ↓
historical ridership
```

Augment with:

``` text
day of week
holiday
time block
event activity
POI footfall
demographics
location characteristics
```

## DemandForecast

``` python
class DemandForecast(BaseModel):
    screen_id: str
    date: date
    time_block_id: str

    predicted_impressions: float
    lower_bound: float
    upper_bound: float

    demand_index: float
    confidence: float
```

## Model requirements

The first implementation should prioritize:

-   strong baseline
-   interpretability
-   fast training
-   robust validation

Possible models:

``` text
Historical average baseline
Seasonal baseline
LightGBM/XGBoost
Gradient boosting regression
```

Do not introduce unnecessary deep learning unless it materially improves
results. `lightgbm` is already a declared dependency.

Report MAE / RMSE / MAPE / R-squared against the naive
`route_schedules.estimated_ridership` baseline (section 26). The pricing
half ships no accuracy metric; do not repeat that.

------------------------------------------------------------------------

# 7A. Availability & Occupancy  **[BUILT]**

`app/ml/occupancy.py`. This was not a named capability in the original
design; it turned out to be the strongest signal in the data and now drives
both feasibility and the price itself.

Capacity is a known constant: **6 slots per screen, per time block, per
day**, confirmed empirically against `bookings`.

Occupancy is not a point estimate. Existing bookings start and end on
different dates, so every query is evaluated **day by day across the whole
requested flight**:

``` text
for each day in [start_date, end_date]:
    occupied[day] = sum(slots_booked_per_day
                        for bookings overlapping that day)

min_available_slots = min(6 - occupied[day])   <-- the reported figure
avg_occupancy_rate  = mean(occupied[day] / 6)  <-- drives the price
```

`min_available_slots` --- the **tightest single day**, not an average --- is
the availability contract. Selling against a mean would oversell the busiest
day of the flight.

A screen with fewer free slots than the campaign needs on any single day is
**infeasible**. Infeasible rows are returned with `feasible=False`, null
pricing, and the occupancy diagnostics intact --- never silently dropped ---
so the caller can distinguish a near miss from a sold-out screen.

------------------------------------------------------------------------

# 7B. Impressions Proxy --- do not use as demand  **[BUILT, quarantined]**

`app/ml/impressions.py` computes:

``` text
proxy = base_traffic x dwell_factor(screen_type)
                     x visibility(screen_size, position)
```

It is surfaced on `ScreenEconomics` as `pricing_internal_reach_proxy`, with
`reach_owner = "audience_engine"`, and is **still deliberately not** mapped onto
any exposure field --- those carry the relevance engine's real ridership figures,
converted once in `app/optimize/exposure.py`. The proxy stays quarantined for pricing diagnostics.
Two reasons, both still disqualifying:

1.  `dwell_factor` and `visibility` are hand-set constants. There is no
    ground-truth exposure data in this dataset to fit them against.
2.  **The two join paths are on different units.** Fixed screens use
    `points_of_interest.est_daily_footfall`, which is per day. Mobile screens
    use `route_schedules.estimated_ridership` averaged per corridor, which is
    per *departure* (~139 departures per corridor per weekday). Measured base
    traffic is 57--48,388 for fixed screens against 22--260 for mobile: a
    ~36x gap that would render all 2,615 mobile screens invisible to any
    impressions-per-dollar ranking.

The unit fix is known: `SUM(estimated_ridership) GROUP BY corridor_id,
day_type` puts mobile on a daily basis (median 2,890 weekday, comparable to
fixed screens' 2,059--2,774). That is essentially what `v_corridor_block_demand`
now does for the real audience model. This module was never promoted and does not
need to be --- `reach_owner = "audience_engine"` was accurate all along; the
audience engine now exists.

------------------------------------------------------------------------

# 8. Pricing Model  **[BUILT]**

Historical bookings are the primary pricing dataset.

Relevant fields include:

-   screen
-   client
-   industry
-   campaign objective
-   time block
-   slots per day
-   rotation type
-   duration
-   contracted price
-   line-item value
-   deal value
-   bundle status
-   booking status

Model:

``` text
price_per_slot_per_day =
    f(
       screen,
       city,
       location,
       screen_size,
       daypart,
       demand,
       duration,
       slots,
       industry,
       campaign_objective,
       client characteristics,
       inventory pressure
    )
```

Both models below are built. Prices are **seller-side**: what the network
should quote for the slot.

## Model A --- Market Price Band  **[BUILT]**

`app/ml/price_band.py`. Rather than predicting a point price, this emits a
**band** --- p25 / p50 / p90 of comparable historical
`contracted_price_per_slot_per_day` --- because the band is what the
downstream price decision moves within.

Segmentation is by physical screen attributes first (the strongest price
signal in the data), then city and daypart. `industry_vertical` is applied
only as a bounded secondary adjustment, never as a primary key.

Deterministic fallback ladder, bounded --- no free-form retries:

``` text
Level A: screen_size x screen_type x position x city x daypart   n >= 30
Level B: screen_size x screen_type x position x city             n >= 30
Level C: screen_size x screen_type x position                    final floor
```

Level C always resolves: all 15 `(screen_size, screen_type, position)`
combinations present in `screens` appear in `bookings`. A test asserts this,
and the code raises rather than falling through if it ever stops holding.

Industry adjustment is the ratio of the (segment x industry) mean to the
segment mean, applied only when that slice has n >= 15, and clamped to
**[0.85, 1.20]**.

The level used and every adjustment applied are recorded per row in
`assumptions`, so any quoted price is traceable to its sample.

Two data-handling rules the segmentation depends on:

-   `position` is null for exactly the 1,400 `metro_rail_coach` screens
    (no entrance/back concept inside a train car). It is filled to
    `"not_applicable"` as an explicit category rather than dropped.
-   The band is looked up against **the screen's own `city_id`**, not the
    campaign's. `CampaignSpec.city_ids` is a list; using the campaign's city
    would mis-segment every out-of-city screen in a multi-city brief.

## Model B --- Booking Probability  **[BUILT]**

`app/ml/booking_probability.py`. Logistic regression on `log(price)` with
screen/city/industry controls.

``` text
WON  (label=1)  bookings          price = contracted_price_per_slot_per_day
LOST (label=0)  lost_leads where  price = quoted_price_per_slot_per_day
                loss_reason in (price_too_high, budget_mismatch)
```

Only **price-driven** losses become negatives. Ghosted, competitor and
timing losses say nothing about price elasticity.

`daypart` is deliberately excluded as a feature: `lost_leads` has no
daypart or time_block column at all, so including it would either drop
every negative example or teach the model that "unknown daypart -> always
lost". Daypart price variation is already captured by Model A; this model
estimates the curve *given* a price already anchored to the right band.

**Two-stage fit**, because the class imbalance is ~490:1 (191,109 won vs
393 lost):

1.  Fit with `class_weight="balanced"`. This is not optional --- an
    unweighted fit produces a **wrong-signed** price coefficient
    (+0.35 unweighted vs -1.18 balanced), because 393 negatives cannot
    outweigh noise in the majority class under plain MLE.
2.  Re-calibrate on a held-out split with Platt scaling against the true
    label distribution, restoring absolute probabilities that balancing
    distorted toward an artificial 50/50 prior.

**The one check that gates trust:** the price coefficient must be negative
after controls. A positive coefficient means premium screens' higher prices
and higher booking rates are not disentangled, and the model must not price
anything. Current fit: **-1.1803**, AUC 0.822, calibrated.

### Known limitation --- read before using booking_probability

The true base rate is **99.79%** (191,109 won against 393 price-driven
losses). Calibration therefore pins predicted probability near 1.0 for
essentially every screen; the observed mean across a live run is **0.9996**.

Two consequences:

-   `expected_revenue = price x booking_probability` is currently
    approximately equal to `price`. It carries almost no extra information.
-   Within one segment's floor-cap band, predicted P(booked) moves only
    ~0.1--0.2%.

The coefficient is correctly signed, so the elasticity is real --- it is
simply far too diffuse to discriminate between prices inside a single
segment. Booking probability is therefore **reported as a diagnostic and
does not set the price**. See section 8A.

------------------------------------------------------------------------

# 8A. The Price Decision  **[BUILT]**

`app/ml/price_optimizer.py`. This is where sections 7A and 8 combine.

The original design (section 8, Model B) called for
`argmax over price of [price x P(booked | price)]`. **That was implemented,
tested, and rejected**: it degenerates to the cap price for every screen,
for the reason given above --- P(booked) is near-flat within a segment's
band, so the product is monotonically increasing in price.

Occupancy is the signal that is strong, exact and validated against real
bookings. It drives the recommendation instead:

``` text
1. Feasibility gate
   occupancy.check_feasibility(screen, block, start, end, slots_needed)
   -> infeasible: return diagnostics, no price. STOP.

2. Band
   price_band.get_price_band(screen, daypart, industry)    # screen's own city
   -> floor, target, cap

3. Seasonality (section 6, M6)
   floor, target, cap  x=  seasonality.combined_multiplier

4. Price
   recommended_price = floor + avg_occupancy_rate x (cap - floor)

   A screen with no committed slots quotes at floor; a fully committed one
   quotes at cap. Scarcity sets the position inside the band.

5. Diagnostics
   booking_probability = P(booked | recommended_price, context)
   expected_revenue    = recommended_price x booking_probability

   The probability-only argmax is still computed and compared. If it
   diverges >15% from the occupancy-driven price, that is recorded in
   `assumptions` rather than silently overriding.

6. Slot-count curve
   price_by_slot_count = {1..6: recommended_price if n <= available else None}
```

**Price is flat across slot counts, by design.** Two candidate curves were
tested and both rejected: occupancy escalation does not exist in the
implementation (`slots_needed` never enters the price formula), and the
apparent volume discount (1.00 -> 0.926 from 1 to 6 slots) is a composition
confound --- larger purchases skew toward cheaper inventory. Controlling for
the price-band segment, the residual effect is ~1.6%, inside noise. The map
is returned explicitly so the interface states this rather than leaving
consumers to infer it.

### Seasonality --- two caveats carried forward

`app/ml/seasonality.py` applies `day_of_week_holiday_multiplier x
event_boost` to the **price**, not to demand. Two things to revisit when the
demand model lands:

-   Ridership day-of-week factors run Friday 1.21 down to Sunday 0.32, and
    the mean over a full week is **0.913**. Every campaign spanning whole
    weeks therefore takes a ~9% haircut off a band already derived from
    actual contracted prices --- which arguably double-counts a
    weekday/weekend mix the band already reflects. A weekend-only flight
    prices at ~0.37x.
-   The holiday multiplier is **inert**: `ridership_actuals` holds two
    holiday dates, both before 2026-08-19, so no future flight can match one.

Event matching has no lat/lon anywhere in the dataset, so `impact_radius_km`
cannot be honoured. It uses exact `anchor_location_id` match as a strong
signal and `city_zone` name match, damped 0.5, as a weak one. Mobile screens
report `not_applicable` rather than a silent 1.0, so callers can tell "no
event nearby" from "cannot check for this screen type".

------------------------------------------------------------------------

# 9. ScreenEconomics  **[BUILT]**

The consolidated object the optimizer receives. **One row per candidate
screen per time block.**

Rows with no purchasable slot are **retained** with `feasible=False` and
`pricing=None`, not dropped, so the caller can see what was excluded and
why. Always branch on `feasible` before reading `pricing`.

As implemented in `app/models/economics.py`:

``` python
class PricingRecommendation(BaseModel):
    floor: float
    target: float
    cap: float
    recommended_price: float
    booking_probability: float
    confidence: float = 0.5          # not populated -- see below


class DemandForecastSummary(BaseModel):
    viewed_exposures_per_slot_per_day: float
    demand_index: float
    confidence: float = 0.5


class ScreenEconomics(BaseModel):
    screen_id: str
    time_block_id: str

    feasible: bool = True

    # --- availability -------------------------------------------------
    availability: list[TimeSlotAvailability] = []   # empty by design
    max_slots_per_day: int = 0        # tightest single day of the flight
    occupancy_rate: float | None = None             # window mean, 0-1
    price_by_slot_count: dict[int, float | None] = {}

    # --- audience volume, mapped in from the relevance engine ----------
    demand_forecast: DemandForecastSummary | None = None
    viewed_exposures_per_slot_per_day: float = 0.0   # scales with slots x days
    daily_unique_audience: float = 0.0    # people PASSING -- not the ceiling
    reachable_daily_audience: float = 0.0 # people who LOOK -- the reach ceiling
    viewability_factor: float | None = None          # which factor was applied
    pool_key: str | None = None           # the dedup unit

    # --- pricing ------------------------------------------------------
    pricing: PricingRecommendation | None = None    # None when infeasible
    expected_revenue: float = 0.0
    confidence: float = 0.5

    # --- diagnostics, for explainability ------------------------------
    seasonality_multiplier: float | None = None
    event_match_type: str | None = None
    pricing_internal_reach_proxy: float | None = None   # NOT reach
    reach_owner: str = "audience_engine"
    assumptions: list[str] = []
```

Field notes:

-   THREE AUDIENCE FIGURES, AND THE UNIT IS IN THE NAME.
    `daily_unique_audience` is people PASSING --- upstream truth, carried for
    traceability. `reachable_daily_audience` is the subset who LOOK
    (`x viewability`) and is the reach ceiling the optimizer and validator cap
    against. `viewed_exposures_per_slot_per_day` is what one slot earns on one day
    and scales with slots x days. The single conversion between them lives in
    `app/optimize/exposure.py`. Never sum exposures across a shared `pool_key` and
    call the result reach --- see section 0.
-   `demand_forecast.demand_index` is **relative**: this line's daily audience over
    the median across the priced pool. 1.0 is a typical line, 3.0 is three times
    the typical audience.
-   `max_slots_per_day` **is** the availability contract. `availability` is
    left empty deliberately: a per-date list would restate the same number
    once per day. Nothing consumes it.
-   `confidence` is retained in the contract but **not populated** by any
    stage. The validation layer skips its check rather than passing on a
    defaulted 0.5 (section 18). Neither model ships a held-out accuracy metric:
    the price band is descriptive (quantiles of comparables) and the audience
    model is an aggregation of observed ridership, so neither has an error bar to
    turn into a confidence. Trust signals are expressed instead as per-row
    `assumptions` (which fallback level, which adjustments), the model-wide
    price-coefficient sign check, and the audience model's `defaults_applied`.
-   `price_by_slot_count` keys are ints 1--6; the value is the absolute
    price, null beyond availability.

Real example (feasible row):

``` json
{
  "screen_id": "LH-SCR-006059",
  "time_block_id": "2",
  "feasible": true,

  "max_slots_per_day": 2,
  "occupancy_rate": 0.39,
  "price_by_slot_count": {"1": 97.13, "2": 97.13, "3": null,
                          "4": null, "5": null, "6": null},

  "pool_key": "LH-LOC-0025",
  "viewed_exposures_per_slot_per_day": 13497.4,
  "daily_unique_audience": 28923.0,
  "reachable_daily_audience": 10123.1,
  "viewability_factor": 0.35,
  "demand_forecast": {"viewed_exposures_per_slot_per_day": 13497.4,
                      "demand_index": 1.34},

  "pricing": {
    "floor": 75.11,
    "target": 91.11,
    "cap": 131.72,
    "recommended_price": 97.13,
    "booking_probability": 0.998
  },
  "expected_revenue": 96.94,

  "seasonality_multiplier": 0.8771,
  "event_match_type": "none",
  "reach_owner": "audience_engine",
  "assumptions": [
    "industry adjustment applied: x1.05",
    "seasonality/event multiplier applied: x0.877",
    "probability-only argmax (131.72) diverges >15% from occupancy-driven price (97.13)"
  ]
}
```

Real example (retained infeasible row):

``` json
{
  "screen_id": "LH-SCR-001448",
  "time_block_id": "2",
  "feasible": false,
  "max_slots_per_day": 0,
  "occupancy_rate": 1.0,
  "pricing": null,
  "assumptions": ["infeasible: only 0 slots available, 1 needed"]
}
```

------------------------------------------------------------------------

# 10. Step 5 --- Inventory Optimization  **[BUILT]**

The optimization layer must be mathematical, not LLM-based. It is both.

**What runs today:** a MILP in `app/optimize/`, HiGHS via `scipy.optimize.milp`,
solved to a 1% relative gap and reported as
`optimization_method = "milp_highs_pooled_reach_min[<objective>,gap<=1%,<status>]"`.
`app/tools/or_agent_tools.py` is a thin wrapper: it maps run state onto the solver
and the solve back onto the Pydantic contracts, and holds the reach accounting.
Ported from an OR handoff bundle; every change made on the way in is documented at
its site in `app/optimize/solver.py`.

Formulation, per candidate cell `i = (screen, time block)`:

``` text
y[i,k]  binary      "cell i gets at least k+1 rotation slots", k = 0..5
z[s]    binary      "screen s is used"
E[p]    continuous  viewed exposures in audience pool p
R[p]    continuous  reach in pool p,  R <= min(E[p], P[p])
c[g]    continuous  shortfall on elastic coverage group g
w[p]    continuous  shortfall on the wear-out cap in pool p

max  w_reach*sum(R) + w_freq*sum(E) + w_conv*sum(conv_fit*E)
     - coverage_penalty*sum(c) - wear_out_penalty*sum(w) - tie_breaker*cost

hard:     budget, per-day availability, slot ordering, screen count,
          required time blocks, min_zone_coverage
elastic:  coverage groups, wear-out cap
```

**Slots are linear; only the pool is concave.** All 6 rotation slots loop
continuously through the block, so slot position is meaningless and the k-th slot
is worth exactly what the first is. The bundle originally modelled a per-pass
notice probability as `(1-p)^(K x slots)`, which treated extra slots as extra loop
passes --- buying more slots does not make the loop run faster. Diminishing returns
are real and live at the audience pool.

**Reach enters as `R <= min(E, P)`, exactly.** `min()` of two linear functions is
concave, so as an upper bound on a maximized variable it is two linear constraints
with no parameter and no piecewise error. Decisively, it makes the solver maximize
the *same* quantity `_package_metrics` reports and `validation._reach_checks`
recomputes. The bundle instead bounded `P x (1 - exp(-lambda E / P))` by six
tangent lines per pool; that curve is a different function, so maximizing it left
audience on the table --- measured 141,501--157,869 reached where the exact bound
returns 261,329 at the same budget, and it turned a 0.1s solve into a 30s timeout.

**No spend floor is invented.** The bundle enforced `MIN_SPEND_FRAC = 0.90` as a
hard constraint. No campaign spec declares a minimum utilisation, and section 2
forbids inventing hard constraints; where it binds it forces leftover budget into
depth, which on a reach brief buys repetition rather than people. A brief that
genuinely requires utilisation says so via
`hard_constraints["min_budget_utilization"]`, and that *is* enforced as hard and
reported as a conflict with the achievable figure if it cannot be met. (Measured
honestly: with the current formulation the floor is inert on the canonical brief ---
floor 0.0 and 0.90 return identical plans, because the reach-optimal package
already uses 99.4% of budget.)

**The screen count counts screens.** The bundle summed the level-1 binary over all
(screen x block) cells, so a screen bought in three blocks counted three times and
`min_screens=8` could return five screens. `requested_num_screens` is validated
against distinct screen ids, so this needed an explicit `z[s]` indicator.

### Wear-out, and why the cap is relative

`E[p] <= F_max x R[p] + w[p]`, with `w` penalized above any exposure reward, so it
binds like a hard constraint but yields rather than manufacturing an infeasibility.

`F_max` is a **multiple of the flight's unavoidable exposure floor**, not an
absolute exposure count. At `LOOP_PASSES_PER_TRIP = 8` one slot on a saturated pool
delivers `8/6` viewed exposures per person per day, so a 30-day flight cannot go
below ~40 exposures per person whatever the optimizer does (section 7). An absolute
cap below that is satisfiable by no plan at all --- gating on one withheld every
package rather than the bad ones. What the cap does do is stop **stacking**: piling
slots and screens into a pool the plan has already saturated. A breach means a hard
constraint forced it, and it is reported.

### What survives any future rewrite

`_package_metrics` is **not** part of the solver. It is the definition of reach this
system reports, `validation._reach_checks` recomputes it independently, and the
formulation now agrees with both. Replace the model; keep the accounting.

Sections 11--12 are the design this implements.

The OR Agent should:

1.  Interpret campaign objective.
2.  Construct decision variables.
3.  Construct objective function.
4.  Add constraints.
5.  Call an optimization solver.
6.  Diagnose infeasibility.
7.  Generate alternatives where necessary.
8.  Return a structured solution.

The LLM should formulate and interpret the problem; the solver should
perform the actual optimization.

------------------------------------------------------------------------

# 11. Optimization Decision Variables  **[BUILT]**

The original target was one integer variable per screen and time block:

``` text
x[s,t] = number of slots purchased on screen s during time block t
```

**As built** that integer is expanded into six "at least k slots" binaries, because
the expansion is what lets each slot carry its own marginal cost and lets the
availability limit be a bound rather than a branch:

``` text
y[i,k]  binary      cell i = (screen, time block) gets at least k+1 slots
                    with y[i,k+1] <= y[i,k], so slots_i = sum_k y[i,k]
```

Slots are linear in value (section 7), so every `y[i,k]` carries the same marginal
audience; the ordering constraints exist for the cost curve and availability, not
for diminishing returns. Four further variable families were added, each for a
reason the target list anticipated:

``` text
z[s]    binary      screen s is used            -> screen COUNT constraints
E[p]    continuous  exposures in pool p         -> frequency
R[p]    continuous  reach in pool p             -> reach, R <= min(E[p], P[p])
c[g]    continuous  coverage shortfall          -> geography coverage, elastic
w[p]    continuous  wear-out shortfall          -> frequency ceiling, elastic
```

`z[s]` is not redundant with `y[i,0]`: a screen can appear in several time blocks,
so a count over cells is not a count over screens (section 10).

Not built, from the target list: **campaign duration** is fixed by the brief rather
than chosen (there is no flighting, which is what makes the wear-out floor in
section 10 unavoidable), and **bundle selection** is not modelled at all even though
71% of historical bookings are bundled deals --- every line is treated as
independent.

------------------------------------------------------------------------

# 12. Optimization Objectives and Constraints  **[BUILT]**

Objective should depend on `CampaignSpec.optimization_goal`.

**As built,** the goal selects the objective's weight blend
(`app/optimize/solver.py::PROFILES`):

``` text
             w_reach  w_freq  w_conv
reach          1.00    0.00    0.00   breadth: distinct people, nothing else
awareness      0.70    0.30    0.00   breadth with some repetition
frequency      0.20    0.80    0.00   depth against a smaller group
conversion     0.40    0.20    0.40   weights conv_fit -- see below
```

Both terms are normalized by their natural ceiling (total pool population, and that
times an effective frequency) so the weights are true shares. Unnormalized, the
bundle measured a nominal 70/30 reach/frequency split actually delivering 28/72.

`objective_value` is deduplicated reach for `reach` and `conversion`, gross viewed
exposures for `awareness` and `frequency`.

**`awareness` and `frequency` no longer share a ranking.** Measured on the canonical
brief: awareness 182,402 reached at 102x frequency, frequency 170,762 at 109x. That
closes the gap this section used to flag.

**The reach profile carries no exposure weight, and that is a change from the
bundle.** It used `w_freq = 0.05` because pure reach maximisation gave its LP
relaxation no guidance --- reach entered only through tangent-bounded variables. The
exact `min(E, P)` bound removes that problem at the source, so the crutch became
harmful: it made the reach profile spend budget on depth. Measured, same brief:
`w_freq = 0.05` returned 227,119 reached at 76x, `w_freq = 0.00` returns 261,329 at
40x, and solves in 0.1s instead of timing out.

**`conversion` is still not a conversion model.** `conv_fit` is the audience
engine's industry-to-POI context score --- a proxy. This system has no conversion
data of any kind. The substitution is recorded in the solver log and the OR agent's
prompt requires reporting it, rather than presenting the result as a conversion
optimum.

Cost enters only as a tie-breaker (`1e-6`, normalized by budget): at equal reach
prefer the cheaper package, and stop buying once there is nobody left to reach. The
magnitude matters --- at `1e-3` it silently cost 2,042 people (0.9% of reach) on an
80-candidate brief, because the solver was then reporting "optimal" against an
objective that was no longer pure reach.

## Reach

Maximize:

``` text
total expected unique impressions
```

## Awareness

Maximize:

``` text
weighted expected impressions
```

## Frequency

Maximize:

``` text
expected frequency
```

subject to a minimum reach constraint where applicable.

## Conversion

Maximize:

``` text
expected conversions
```

if a suitable conversion model is available.

## Core constraints

These **are** enforced today, in the MILP, and independently re-checked by the
validation layer (section 18):

``` text
HARD, in the solver and re-checked by the validator
[BUILT] Total cost <= budget
[BUILT] Purchased slots <= available slots   (tightest day of the flight)
[BUILT] Campaign dates are valid
[BUILT] Only eligible geography is selected  (hard-filtered in stage 2)
[BUILT] requested_num_screens / min_screens / max_screens   (distinct SCREENS)
[BUILT] allowed_screen_types, required_time_blocks
[BUILT] min_zone_coverage       cardinality over distinct zones, not over cells
[BUILT] min_budget_utilization  only when the brief declares it

ELASTIC, penalized and reported rather than failing the solve
[BUILT] coverage groups          -> unmet_coverage
[BUILT] wear-out frequency cap   -> wear_out_exposures_over_cap
```

The elastic pair is penalized above any reward it could earn, so both behave as
hard unless another hard constraint forces a breach --- in which case the plan is
still returned and the breach is reported rather than hidden. `min_zone_coverage`
is hard because the validator fails a package that misses it; a shortfall there
would be a package that cannot pass verification.

There is deliberately **no** default minimum budget utilisation. See section 10.

At minimum:

``` text
Total cost <= budget
```

``` text
Purchased slots <= available slots
```

``` text
Campaign dates are valid
```

``` text
Only eligible geography is selected
```

Additional constraints should include:

``` text
requested_num_screens
minimum/maximum screens
minimum geographic coverage
preferred dayparts
inventory availability
screen-type restrictions
```

Different campaigns may have different hard and soft constraints.

------------------------------------------------------------------------

# 13. OptimizedPackage  **[BUILT]**

``` python
class Allocation(BaseModel):
    screen_id: str
    time_block_id: str

    slots_per_day: int
    duration_days: int

    price_per_slot_per_day: float

    viewed_exposures: float          # gross, = per_slot_per_day x slots x days
    expected_revenue: float


class OptimizedPackage(BaseModel):
    allocations: list[Allocation]

    total_cost: float
    gross_impressions_viewed: float  # exposures. NEVER a count of people
    expected_reach: float            # distinct people, deduplicated + capped
    expected_frequency: float

    budget_utilization: float

    constraint_status: dict[str, bool]

    objective_value: float

    optimization_method: str

    # --- solver diagnostics, additive and never load-bearing ---------------
    curve_reach_diagnostic: float | None = None
    unmet_coverage: dict[str, float] = {}
    wear_out_exposures_over_cap: float = 0.0
```

Example:

``` json
{
  "total_cost": 48750,
  "gross_impressions_viewed": 1250000,
  "expected_reach": 875000,
  "expected_frequency": 1.43,
  "budget_utilization": 0.975,

  "allocations": [
    {
      "screen_id": "LH-SCR-001928",
      "time_block_id": "5",
      "slots_per_day": 2,
      "duration_days": 30,
      "price_per_slot_per_day": 76,
      "viewed_exposures": 552000,
      "expected_revenue": 4560
    }
  ],

  "constraint_status": {
    "budget": true,
    "inventory": true,
    "geography": true,
    "campaign_dates": true
  },

  "objective_value": 1250000,
  "optimization_method": "MILP"
}
```

The exact numerical values above are illustrative only.

## As produced today

A real run: \$50k, 30 days from 2026-10-01, Downtown Core, young commuters,
technology vertical, `optimization_goal = "reach"`.

``` json
{
  "total_cost": 49707.9,
  "budget_utilization": 0.9942,

  "gross_impressions_viewed": 10453140.0,
  "expected_reach": 261329.0,
  "expected_frequency": 40.0,

  "objective_value": 261328.5,
  "optimization_method": "milp_highs_pooled_reach_min[reach,gap<=1%,optimal]",
  "curve_reach_diagnostic": 261329.0,

  "constraint_status": {
    "budget": true,
    "inventory": true,
    "geography": true,
    "campaign_dates": true,
    "requested_num_screens": true,
    "coverage": true
  }
}
```

25 screens, 27 lines, across **18 audience pools**. Proven optimal within the 1%
gap, in 0.1s. Read the figures correctly:

-   10.45M is **gross viewed exposures**; 261k is **distinct people**. Cost per
    person reached is \$0.19.
-   The 40.0x ratio is frequency, and it is exactly the floor for a 30-day flight:
    `LOOP_PASSES_PER_TRIP / 6 x 30 = 40` (section 7). Every pool saturated and the
    plan bought **no** depth beyond that --- which is what a reach objective should
    do, and what it did not do before the exposure weight came off it.
-   `objective_value` is the deduplicated reach, matching `expected_reach`.
-   `curve_reach_diagnostic` coincides here because every pool saturated; it is not
    generally equal, and it is never the reported figure.
-   `Allocation.viewed_exposures` is per-line gross exposures and sums to
    `gross_impressions_viewed` --- the validator checks this. `expected_revenue` is
    `price x slots x days`.

**The composition changed, and the reason is worth understanding.** 16
`metro_station` + 9 `bus_stop`, in 1 zone (geography is a hard filter, so one
requested zone means one zone). The greedy version bought 16 screens, all
`metro_station`. Bus stops now earn their place because reach saturates per pool:
once a station pool is covered, a cheap *distinct* pool is worth more than more
depth on an expensive one. The ~380x fixed/mobile volume gap from section 7 is still
there in the underlying model --- this is the optimizer routing around it, not a fix.

For the same brief, `compare_objectives` returns:

``` text
objective    spend     screens  pools  reach     frequency  cost/person
reach        49,882    25       18     260,245   40.0       0.19
awareness    49,679    15        6     182,402  101.8       0.27
frequency    49,679    10        6     170,762  109.0       0.29
```

260,000 people 40 times, or 171,000 people 109 times. That is a media planner's
judgement, and the OR agent's prompt requires presenting it rather than resolving it
silently.

`InfeasibilityReport` is fully implemented, with reason codes drawn from a
fixed vocabulary in `app/models/optimization.py`:

``` text
BUDGET_CONSTRAINT            TOO_MANY_SCREENS_REQUESTED
INSUFFICIENT_INVENTORY       DAYPART_UNAVAILABLE
GEOGRAPHY_UNAVAILABLE        CONFLICTING_HARD_CONSTRAINTS
DATES_UNAVAILABLE            NO_CANDIDATES
```

`OptimizationResult` is a discriminated union --- exactly one of `package`
or `infeasibility` is set. There is no third state in which a partial or
speculative package is returned.

------------------------------------------------------------------------

# 14. Step 6 --- Recommendation Generator  **[BUILT]**

The Recommendation Agent receives analytical outputs and converts them
into a sales-ready recommendation.

It must **not recalculate analytical values**.

Inputs:

``` text
CampaignSpec
ScreenCandidates
ScreenEconomics
OptimizedPackage
Validation results
```

Output:

``` python
class ScreenExplanation(BaseModel):
    screen_id: str
    explanation: str
    supporting_factors: list[str]


class AlternativePackage(BaseModel):
    name: str
    description: str
    package: OptimizedPackage
    tradeoffs: list[str]


class CampaignRecommendation(BaseModel):
    executive_summary: str

    recommended_package: OptimizedPackage

    key_recommendations: list[str]

    screen_explanations: list[ScreenExplanation]

    pricing_explanation: str

    audience_explanation: str

    optimization_explanation: str

    risks: list[str]

    alternatives: list[AlternativePackage]
```

Example final language:

> Recommended package: 18 screens across 4 locations for 30 days at
> \$48,750.

Then explain:

-   why the locations fit the audience
-   why particular time blocks were selected
-   why the price is appropriate
-   how the package uses the budget
-   what the expected reach/impressions are
-   what tradeoffs exist

------------------------------------------------------------------------

# 15. Agent Architecture

Do not create an agent for every small function.

Use:

``` text
                   ┌──────────────────┐
                   │   MASTER AGENT   │
                   │    / MANAGER     │
                   └────────┬─────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
    DATA AGENT          ML AGENT           OR AGENT
          │                 │                  │
          ▼                 ▼                  ▼
     Data Tools          ML Tools          OR Tools
          │                 │                  │
          └─────────────────┼──────────────────┘
                            ▼
                    Shared Artifacts
```

As built --- **two** specialists, one module each, so either can be replaced
without touching the other:

``` text
app/agents/
  master.py        create_deep_agent(...) + shared rate limiter
  prompts.py       MASTER_SYSTEM_PROMPT only
  ml_agent.py      NAME + DESCRIPTION + PROMPT + build()
  or_agent.py      NAME + DESCRIPTION + PROMPT + build()
  subagents.py     assembles the two in pipeline order
  validation.py    deterministic verification, Master-owned

app/tools/
  master_tools.py     intake, geography, verify, inspect
  relevance_tools.py  THE AUDIENCE ENGINE + its tools -- Master-owned, no subagent
  ml_agent_tools.py   thin wrapper over app/ml/
  or_agent_tools.py   thin wrapper over app/optimize/ + the reach accounting
app/ml/            the pricing engine (section 6)
app/optimize/      the MILP (section 10)
  config.py          every constant, tagged STRUCTURAL / ASSUMED / SOLVER
  exposure.py        people passing -> viewed exposures. ONE implementation
  contract.py        input validation, naming the component that owns each column
  pooled.py          pool population + the saturation curve (diagnostic only)
  solver.py          the formulation
app/features/      empty -- the audience engine lives in tools/relevance_tools.py
```

### Why there is no Data Intelligence subagent

The original design (and section 31.1) called for three specialists. The Data one
was removed, deliberately.

Stage 2 turned out to have **no judgement in it**. Given a campaign spec, the
candidate pool is a pure function: hard-filter the geography, compute five
weighted sub-scores, rank, take the top N. An LLM sitting in front of that could
only do two things --- decide *when* to call it, which the Master already does, and
restate its numbers in prose, which is a chance to get them wrong. It also cost
model calls against a per-minute rate limit.

So stage 2 is a Master-owned tool. This trades section 31.1 ("exactly three
specialists") against 31.2 ("LLMs reason; tools calculate"), and 31.2 wins:
delegation is now reserved for the two stages where a specialist genuinely reasons
about its own output --- the ML agent choosing what to say about price credibility,
and the OR agent interpreting a solve or an infeasibility.

The general rule this leaves behind: **prefer a Master-owned tool over a subagent
whenever a stage does not actually reason.**

Each remaining subagent file holds its own system prompt and is a thin delegation
shell: the tools do the work, so integrating a real implementation means replacing
a tool module and its engine package, not rewiring the agent graph.

------------------------------------------------------------------------

# 16. Audience Relevance Engine  **[BUILT]** --- formerly the Data Agent

`app/tools/relevance_tools.py`. **Not an agent.** See section 15 for why. The
responsibilities below are all still discharged; they are just discharged by a
deterministic engine the Master calls, plus the DuckDB view layer beneath it.

## Responsibilities

-   campaign data interpretation      -> reads `CampaignSpec` from run state
-   database exploration / table joins -> `v_screen_profile`,
                                          `v_screen_demand_history` (section 20)
-   filtering                          -> hard geography + `hard_constraints`
-   feature engineering                -> section 4
-   audience analysis                  -> six normalized demographic scores
-   screen profiling                   -> one row per screen, all 11,163
-   relevance scoring                  -> section 5
-   data quality checks                -> `defaults_applied`, per row and per pool

## Tools

The Master's own tool surface for this stage:

``` python
describe_inventory(run_id) -> dict
    Eligible screen count in the requested geography, by type and by
    fixed/mobile class. Cheap; call it to confirm the geography resolves.

build_screen_candidates(run_id, top_n=None) -> dict
    Runs the engine. Writes the screen_candidates artifact and returns a
    reference plus aggregates. Everything it needs comes from the spec, so the
    same spec always yields the same pool.

describe_relevance_model(run_id) -> dict
    How relevance and audience volume are computed, and the six known
    limitations verbatim. This is what lets the Master state the model's limits
    accurately instead of guessing at them.
```

## Input

``` text
run_id
```

That is the whole input. `audience_terms`, `day_type_focus`, geography,
`industry_vertical` and `hard_constraints` are all read from the spec, so no
caller can desync them from what intake recorded.

## Output

``` text
ScreenCandidate[]  as an artifact reference (parquet), ranked best-first
+ aggregates: eligible_screens, candidates, relevance min/mean/max,
              preferred_time_blocks, day_type_focus, audience_terms,
              distinct_audience_pools, pooled_daily_audience,
              naive_daily_audience, demand_source, defaults_applied
```

`preferred_time_blocks` on that summary is load-bearing: the pricing stage reads
it to price the blocks this campaign's audience is actually in, so the
audience-to-blocks mapping has exactly one authoritative copy.

Never returns raw DataFrames --- the artifact reference is the handoff.

------------------------------------------------------------------------

# 17. ML / Pricing Agent  **[BUILT]**

Defined in `app/agents/ml_agent.py`. Renamed from "ML / Forecasting" because
it forecasts nothing --- it prices.

The original design had this agent training models at request time and
reasoning over their metrics. That was rejected. Models are fitted **once at
engine build** and are deterministic thereafter; a per-request retrain would
make identical briefs return different prices and cost ~12s each time. The
agent's job is to invoke the engine and report faithfully, not to do model
selection in the loop.

## Responsibilities

**Owns:**

-   market price band (p25/p50/p90, segmented, with a bounded fallback ladder)
-   the price decision --- occupancy-driven position within that band
-   booking probability, reported as a diagnostic
-   slot availability and feasibility, day by day across the flight
-   seasonality and event adjustment
-   per-row explainability: which fallback fired, which adjustments applied

**Explicitly does not own:** demand, impressions, reach, frequency, CPM,
demand index, per-screen confidence. The prompt requires it to say so rather
than substitute a proxy.

## Input

``` text
run_id
```

That is the entire input. The agent reads `CampaignSpec` and the
`screen_candidates` artifact reference from run state itself. Candidate rows,
price tables and DataFrames never enter LLM context.

## Tools

``` python
estimate_screen_economics(run_id, time_blocks=None, slots_needed=1) -> dict
describe_pricing_model(run_id) -> dict
```

`estimate_screen_economics` writes the `screen_economics` artifact and
returns a reference plus aggregates. `describe_pricing_model` reports how the
models were fitted --- sample sizes, price coefficient and its sign check,
AUC, calibration --- so the agent can justify *why* a price is credible, and
so a failed sign check surfaces rather than hiding.

`app/tools/ml_agent_tools.py` is a thin wrapper. All decision logic lives in
`app/ml/`. Tools map contracts; they do not calculate.

## Internal logic flow

``` text
estimate_screen_economics(run_id)
    │
    ├── prerequisite check: screen_candidates artifact present?
    │      └── no -> return {"status": "prerequisite_missing"}   (recoverable)
    │
    ├── read CampaignSpec + candidate screen_ids from run state
    ├── resolve time blocks: explicit arg
    │                        else spec.preferred_time_blocks
    │                        else ("2", "3", "5")   commuter peaks + midday
    ├── validate blocks against dim_slot -> reject unknown ids
    ├── end_date = start_date + duration_days - 1   (inclusive)
    │
    ├── get_pricing_engine()        <-- process singleton, built once
    │
    └── for each time_block:
           daypart = dim_slot.nearest_daypart[block]   (derived, not passed)
           for each candidate screen:
               seasonality  = M6.get_adjustment(screen, start, end)
               pricing      = M4.price_screen(..., city = screen's own,
                                              price_multiplier = seasonality)
               -> map to ScreenEconomics                (section 8A)
    │
    ├── all rows infeasible? -> status "no_availability" + relaxation options
    └── write artifact (provenance="computed") -> return reference + aggregates
```

Two signature decisions worth preserving:

-   **`daypart` is derived, never passed.** In `bookings`, `time_block_id`
    and `daypart` are a strict 1:1 function of each other (blocks 1 and 6
    both map to `night`). Accepting both as free parameters lets a caller
    desync them and silently segment the price band on the wrong daypart.
-   **`city_id` comes from the screen, not the campaign.**
    `CampaignSpec.city_ids` is a list.

## Output

``` text
ScreenEconomics[]   as an artifact reference (parquet)
                    one row per candidate x time block,
                    infeasible rows retained
+ aggregates: screens_priced, lines_feasible, lines_infeasible,
              price min/mean/max, occupancy_mean, booking_probability_mean,
              viewed_exposures_per_slot_per_day_mean,
              demand_model="transit_ridership (relevance engine)"
```

The audience fields on each row (`viewed_exposures_per_slot_per_day`,
`daily_unique_audience`, `reachable_daily_audience`, `pool_key`,
`demand_forecast`) are **mapped through** from the candidate artifact, not
modelled here. The only arithmetic this stage does on them is the unit conversion
in section 7: weight the block's daily people-passing by the flight's real
weekday/weekend day counts, then call `app/optimize/exposure.py` --- the one
module that turns people into exposures.

Do not force the LLM to perform numerical calculations itself.

------------------------------------------------------------------------

# 18. OR Agent + Master Agent + Shared Artifact Architecture

## OR Agent  **[BUILT]**

`app/agents/or_agent.py`. See section 10 for the formulation. Its prompt's main job
is the reach-vs-exposures distinction: it must quote both, never add exposures
together and call the result reach, never quote `curve_reach_diagnostic` as the
campaign's reach, and say whether the solve proved optimality or stopped inside the
gap. It also has to pass up the wear-out disclosure and any unmet coverage verbatim.

Responsibilities:

-   interpret optimization objective
-   formulate decision variables
-   construct constraints
-   solve optimization
-   diagnose infeasibility
-   generate alternatives
-   explain tradeoffs

Tools, as built --- two, not five:

``` python
optimize_package(run_id, slots_per_day_cap=3) -> dict
compare_objectives(run_id, objectives=None) -> dict
```

The five-tool surface above collapsed into these. Model building, solving and
feasibility checking are one deterministic call, not three an LLM sequences;
`analyze_solution` is what the returned payload already is. `compare_objectives` is
`generate_alternative_solution` made specific: same brief, different objective,
plans side by side.

It **withholds** rather than returns a plan whose exposures per person exceed the
wear-out cap, reporting the measured figures and the reason. Nothing silently
surfaces a plan that is past the point of usefulness. The campaign's own stated
objective is never withheld or substituted --- it is served and disclosed.

The handoff bundle shipped four further tools --- `budget_sensitivity`, `diagnose`,
`scenario_reoptimize`, `explain_last_plan`. They are not wired in: each costs a
solve plus model calls against a per-minute rate limit, and `diagnose` mostly
produces narration that then has to be defended. `budget_sensitivity` is the one
worth revisiting, since "20% more budget adds N people at \$X each" is a real sales
answer.

The OR solver is deterministic. HiGHS via `scipy.optimize.milp`; CP-SAT was
available in the bundle and is not needed --- the formulation solves in 0.1s.

------------------------------------------------------------------------

## Master Agent  **[BUILT]**

`app/agents/master.py`. The only component responsible for end-to-end
orchestration. It owns stages 1, 5 and 6 itself and delegates 2--4 through
the built-in `task` tool.

Its tools: `resolve_geography_terms`, `create_campaign_spec`,
`get_run_state`, `verify_package`, `inspect_package`, `check_explanations`.

**Run handles, not payloads.** Every tool takes a `run_id`. The spec,
artifact references, optimization result and validation live in run state and
the artifact store, so candidate lists and price tables never enter an LLM's
context.

**Stages are strictly sequential.** Each consumes the previous stage's
artifact. A supervisor can still try to delegate two stages in one turn, so
dependent tools check their input and return a recoverable
`prerequisite_missing` result naming the producing stage --- they never crash.

Workflow:

``` text
USER
 │
 ▼
MASTER
 │
 ├── Understand Campaign
 │
 ├── Run the relevance engine   (own tool, no delegation, no LLM)
 │      │
 │      └── ScreenCandidates
 │
 ├── Ask ML Agent
 │      │
 │      └── ScreenEconomics  (pricing only)
 │
 ├── Ask OR Agent
 │      │
 │      └── OptimizedPackage  or  InfeasibilityReport
 │
 ├── VERIFY
 │      │
 │      ├── Budget ✓
 │      ├── Inventory ✓
 │      ├── Dates ✓
 │      ├── Geography ✓
 │      ├── Reach (recomputed from pool_key groups) ✓
 │      ├── Model confidence — SKIPPED (no stage emits one)
 │      └── Reconciliation ✓
 │
 └── FINAL RECOMMENDATION
```

The Master Agent must not blindly trust subagents.

It should validate:

``` text
1. Is the package within budget?
2. Are all selected screens available?
3. Are dates valid?
4. Are all screens geographically eligible?
5. Are requested hard constraints satisfied?
6. Are model outputs sufficiently confident?
7. Is the optimizer solution feasible?
8. Do allocation totals reconcile?
9. Do cost calculations reconcile?
10. Are explanations consistent with underlying values?
```

### As implemented --- `app/agents/validation.py`

Every check runs in Python against reference data. An LLM never decides
whether a constraint was met and cannot reason a violation away. Checks
report `pass` / `fail` / `skipped`; any `fail` fails the package.

`verify_package` runs 18 checks on a typical brief --- 16 pass, 2 skip:

``` text
package_non_empty              budget_respected
cost_reconciles                budget_utilization_reconciles
impressions_reconcile          reach_not_above_impressions
frequency_reconciles           reach_reconciles
curve_reach_bounded            screens_exist
time_blocks_valid              no_duplicate_allocations
geography_eligible             duration_within_campaign
start_date_not_in_past         inventory_availability

hard_constraints               SKIPPED  (none declared on this spec)
model_confidence               SKIPPED  (no stage emits one)
```

Three audience checks became live once reach existed. `reach_not_above_impressions`
and `frequency_reconciles` had been skipping on a zero reach; `reach_reconciles` is
the important one --- see below.

`curve_reach_bounded` arrived with the MILP. It does **not** re-derive the solver's
saturation curve: that formula's only real unknown is `REACH_LAMBDA`, so
reimplementing it here would validate nothing about it. It asserts the two things
that hold for any lambda --- reach exceeds neither the exposures bought nor the
people available to be reached --- which is exactly the over-count class worth
catching.

Conditional checks, added only when the spec declares the constraint:

``` text
requested_num_screens   min_screens          max_screens
allowed_screen_types    required_time_blocks min_zone_coverage
```

`explanations_consistent` is a 16th check, run separately by the
`check_explanations` tool once the Master has drafted its screen-level
claims. It rejects any explanation naming a screen not in the package.

One deviation from the checklist above:

-   **Check 6 (`model_confidence`) is skipped, not evaluated.** No stage
    emits a per-screen confidence, because neither the pricing band nor the
    audience aggregation has a held-out error bar to derive one from. Gating on
    the contract's defaulted 0.5 would report a pass that means nothing. Restore
    the gate when a stage produces a real confidence.

The validator **recomputes rather than trusting**, in two places specifically
because they are the two numbers an optimizer can most plausibly overstate:

-   `cost_reconciles` re-derives `sum(price x slots x days)` from the allocations
    and compares it against the reported `total_cost`.
-   `reach_reconciles` re-derives deduplicated reach from the `pool_key` groups,
    each capped at its **reachable** daily audience, and compares it against the
    reported `expected_reach`. Summing per-screen exposures and calling it reach
    over-counts ~23x and looks entirely reasonable on the way past --- so it gets an
    independent second implementation, deliberately not sharing code with
    `or_agent_tools._package_metrics`.
-   `curve_reach_bounded` bounds the solver's saturation diagnostic without
    re-deriving it: `curve_reach <= min(sum E, sum P)` holds for any value of
    `REACH_LAMBDA`, so it tests the pool bookkeeping rather than the constant. See
    section 31.8b for why re-deriving it here would have been the wrong move.

Tests tamper with a package on both axes (understated cost, inflated exposures,
reach claimed equal to gross exposures) to confirm the validator catches it.

------------------------------------------------------------------------

# 19. Shared Artifact Architecture  **[BUILT]**

Do **not** pass huge pandas DataFrames directly between agents.

Instead, specialist agents should create durable artifacts.

Example:

``` text
Agent A
   │
   │ writes
   ▼
screen_candidates.parquet
   │
   ▼
artifact metadata
```

Example artifact metadata:

``` json
{
  "artifact": "screen_candidates.parquet",
  "rows": 250,
  "columns": [
    "screen_id",
    "relevance_score",
    "audience_match_score",
    "geography_score",
    "contextual_score",
    "transit_score"
  ],
  "summary": {
    "min_relevance": 0.61,
    "max_relevance": 0.97,
    "mean_relevance": 0.81
  }
}
```

Agent B should receive:

``` text
artifact reference
+
schema
+
summary
+
task
```

rather than hundreds/thousands of rows in its LLM context.

This is especially important for Deep Agents because specialist agents
should perform detailed work internally and return concise structured
results to the supervisor.

------------------------------------------------------------------------

# 20. Data Layer  **[BUILT]**

Use a clean data-access layer beneath the agents.

Recommended conceptual architecture:

``` text
                 MASTER DEEP AGENT
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   DATA AGENT       ML AGENT          OR AGENT
        │               │                │
        └───────────────┼────────────────┘
                        │
                 DOMAIN TOOL LAYER
                        │
       ┌────────────────┼────────────────┐
       │                │                │
   SQL/Pandas       Python/ML        OR Solver
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                 DATA / ARTIFACTS
```

Recommended initial data storage:

``` text
CSV files
   ↓
DuckDB
   ↓
SQL views / feature tables
   ↓
Python / ML / optimization tools
```

Do not make agents manually manipulate CSV files wherever possible.

Create stable views such as:

``` text
v_screen_profile
v_screen_availability
v_screen_demand_history
v_historical_pricing
v_campaign_inventory
```

**As built** --- `app/data/db.py` registers all 14 source CSVs as DuckDB
views (streaming, so the 124 MB `ridership_actuals` is never materialized)
plus two derived views:

``` text
[BUILT] v_screen_geography   resolves the fixed/mobile split for every screen.
                             Fixed: screen -> location -> zone.
                             Mobile: screen -> vehicle -> corridor, zone NULL.
                             Does NOT fan out to route_stops -- that would
                             multiply one screen into every stop on its corridor.
[BUILT] v_corridor_zones     corridor -> distinct zones, for geography filters
```

The audience engine added the rest of the originally-named layer:

``` text
[BUILT] v_screen_profile         one row per screen: geography, zone demographics,
                                 POI context (inverse-distance weighted footfall),
                                 and pool_key. Section 4.
[BUILT] v_screen_demand_history  avg daily riders per screen x time block x day
                                 type, 111,630 rows. Section 7.
[BUILT] v_screen_poi             POI aggregates per screen, feeding the profile
[BUILT] v_schedule_block         departures tagged with their dim_slot block
[BUILT] v_route_block_demand     route x block x day_type, from ridership_actuals
                                 (falls back to scheduled estimates without it)
[BUILT] v_corridor_block_demand  corridor x block x day_type / vehicles on corridor
[BUILT] v_corridor_vehicle_count vehicles working each corridor. Defined once and
                                 used by BOTH v_corridor_block_demand (as the
                                 divisor) and v_screen_profile (as
                                 pool_partition_count), so the two cannot disagree
```

`v_historical_pricing` and `v_campaign_inventory` were never created: the pricing
engine reads through `app/ml/loaders.py`, which issues explicit column-projected
SQL against the base views rather than materializing wide feature tables.

Views are created in dependency order --- DuckDB binds a view at definition time,
so a view cannot reference one that does not exist yet.

`ridership_actuals` is gitignored and **optional**: `has_ridership_actuals()`
reports whether it was provisioned, and the seasonality module degrades to a
neutral 1.0 multiplier rather than crashing when it is absent.

------------------------------------------------------------------------

# 21. Important Table Relationships

The implementation should preserve the existing join map.

Core relationships:

``` text
Commercial
──────────
bookings.booking_id
bookings.client_id       -> client_facts.client_id
bookings.screen_id       -> screens.screen_id
bookings.time_block_id   -> dim_slot.time_block_id

lost_leads.client_id     -> client_facts.client_id
lost_leads.anchor_screen_id -> screens.screen_id


Inventory
─────────
screens.location_id      -> locations.location_id
screens.vehicle_id       -> vehicles.vehicle_id

vehicles.corridor_id     -> route_schedules.corridor_id
vehicles.corridor_id     -> route_stops.corridor_id


Transit Network
───────────────
route_schedules.schedule_id -> ridership_actuals.schedule_id
route_schedules.route_id    -> route_stops.route_id

route_stops.location_id     -> locations.location_id


Geography
─────────
locations.zone_id       -> zone_demographics.zone_id
locations.city_id       -> cities.city_id


Context
───────
events.poi_id           -> points_of_interest.poi_id
events.anchor_location_id -> locations.location_id

points_of_interest.anchor_location_id -> locations.location_id
```

The implementation should avoid accidental many-to-many row explosions.
Every join must be checked for expected cardinality.

------------------------------------------------------------------------

# 22. Canonical Artifacts

The system should standardize these six business artifacts:

  --------------------------------------------------------------------------
  Stage                   Input                   Output
  ----------------------- ----------------------- --------------------------
  Brief Intake            User query + optional   `CampaignSpec`
                          client context          

  Audience Intelligence   `CampaignSpec` + raw    `ScreenProfile[]`
                          tables                  

  Relevance Scoring       `CampaignSpec` +        `ScreenCandidate[]`
                          `ScreenProfile[]`       

  Demand & Pricing        `CampaignSpec` +        `ScreenEconomics[]`
                          candidates + historical 
                          data                    

  Optimization            `CampaignSpec` +        `OptimizedPackage` or
                          economics               

  Recommendation          All previous artifacts  `CampaignRecommendation`
  --------------------------------------------------------------------------

These are the interfaces that should remain stable even if internal
models are replaced.

## Status per artifact

  --------------------------------------------------------------------------
  Artifact                Status              Note
  ----------------------- ------------------- ------------------------------
  `CampaignSpec`          **[BUILT]**         validated, geography resolved

  `ScreenProfile[]`       **[BUILT]**         `v_screen_profile`, all 11,163
                                              screens

  `ScreenCandidate[]`     **[BUILT]**         weighted relevance + audience
                                              volume + pool_key,
                                              `provenance="computed"`

  `ScreenEconomics[]`     **[BUILT]**         pricing, availability and
                                              audience volume;
                                              `provenance="computed"`

  `OptimizedPackage`      **[BUILT]**         real constraints, real reach
                                              objective, MILP solve

  `CampaignRecommendation` **[BUILT]**        assembled by the Master
  --------------------------------------------------------------------------

`provenance` is carried on every artifact reference and propagated into
`run_state.stub_stages()`. The Master's prompt requires it to disclose any stub
stage at the top of its answer. A current run reports `stub_stages: []` --- nothing
is a stub. The plumbing stays because it is the mechanism that would surface a
regression.

------------------------------------------------------------------------

# 23. Recommended Tool Interfaces

The coding implementation should expose narrow, deterministic tools.

Example:

``` python
def query_database(sql: str) -> QueryResult:
    ...


def build_screen_profiles(
    campaign_spec: CampaignSpec
) -> ArtifactReference:
    ...


def score_screens(
    campaign_spec: CampaignSpec,
    screen_profiles: ArtifactReference
) -> ArtifactReference:
    ...


def forecast_demand(
    campaign_spec: CampaignSpec,
    screen_candidates: ArtifactReference
) -> ArtifactReference:
    ...


def estimate_pricing(
    campaign_spec: CampaignSpec,
    screen_candidates: ArtifactReference,
    demand_forecasts: ArtifactReference
) -> ArtifactReference:
    ...


def optimize_inventory(
    campaign_spec: CampaignSpec,
    screen_economics: ArtifactReference
) -> OptimizedPackage:
    ...


def validate_package(
    campaign_spec: CampaignSpec,
    optimized_package: OptimizedPackage
) -> ValidationResult:
    ...
```

## As implemented

Every tool takes a `run_id` instead of inlined artifacts --- the spec and
artifact references live in run state, so nothing bulky crosses an agent
boundary. Every tool returns a JSON-safe dict, never a DataFrame.

``` python
# Master   app/tools/master_tools.py                          [BUILT]
resolve_geography_terms(terms: list[str]) -> dict
create_campaign_spec(...) -> dict                  # returns run_id
get_run_state(run_id) -> dict
verify_package(run_id) -> dict
inspect_package(run_id, limit=10) -> dict
check_explanations(run_id, explained_screen_ids) -> dict

# Relevance  app/tools/relevance_tools.py -- MASTER-OWNED, no subagent  [BUILT]
describe_inventory(run_id) -> dict
build_screen_candidates(run_id, top_n=None) -> dict
describe_relevance_model(run_id) -> dict

# ML       app/tools/ml_agent_tools.py                         [BUILT]
estimate_screen_economics(run_id,
                          time_blocks=None,
                          slots_needed=1) -> dict
describe_pricing_model(run_id) -> dict

# OR       app/tools/or_agent_tools.py                         [BUILT]
optimize_package(run_id, slots_per_day_cap=3) -> dict
compare_objectives(run_id, objectives=None) -> dict
```

Note there is no separate `forecast_demand` / `estimate_pricing` split.
`estimate_screen_economics` is the single stage-4 entry point, and demand
populates the reserved fields on that same artifact --- which is exactly how the
audience model landed, with no new stage and no signature change.

Nor is there a `build_screen_profiles` tool: the profile is a DuckDB view built
once inside the engine singleton, not a per-request call.

The LLM agents should decide **when and why** to call these tools.

The tools should decide **how** to perform the calculations.

------------------------------------------------------------------------

# 24. Error Handling and Infeasibility  **[BUILT]**

The system must not fabricate a recommendation when the optimization
problem is infeasible.

Possible reasons:

``` text
Budget too low
No inventory available
Requested geography unavailable
Requested dates unavailable
Too many screens requested
Required daypart unavailable
Conflicting hard constraints
```

The OR Agent should return something like:

``` json
{
  "status": "infeasible",
  "reason_codes": [
    "INSUFFICIENT_INVENTORY",
    "BUDGET_CONSTRAINT"
  ],
  "explanation": "No feasible package satisfies the requested screen count within the specified budget.",
  "relaxation_options": [
    "Increase budget",
    "Reduce number of screens",
    "Extend campaign duration",
    "Allow additional geography"
  ]
}
```

The Master Agent can then either:

1.  ask the user to relax a constraint, or
2.  automatically generate an alternative if the user has permitted
    flexibility.

## As built

`InfeasibilityReport` draws its codes from the fixed vocabulary in
`app/models/optimization.py` (section 13), and `OptimizationResult` is a
discriminated union --- exactly one of `package` / `infeasibility` is set, with no
third state in which a partial fill is returned.

The diagnosis distinguishes causes rather than blaming the budget for everything:

``` text
cheapest line > budget                 -> BUDGET_CONSTRAINT
exact screen count unreachable         -> TOO_MANY_SCREENS_REQUESTED
   (established by RE-SOLVING without the count: if a package exists at a lower
    count, the constraint binds, not the money -- and the achievable count is
    named in the explanation)
candidate pool spans too few zones     -> CONFLICTING_HARD_CONSTRAINTS
                                          + GEOGRAPHY_UNAVAILABLE
declared min_budget_utilization unmet  -> CONFLICTING_HARD_CONSTRAINTS
   (with the utilisation that IS achievable, so the number in the relaxation
    option is real)
no purchasable line in the window      -> INSUFFICIENT_INVENTORY
required_time_blocks excludes all      -> CONFLICTING_HARD_CONSTRAINTS
nothing priced at all                  -> NO_CANDIDATES
```

Every relaxation option carries the figure that would make it work, because
"increase budget" without a number is not an action.

One gap worth naming: a `min_zone_coverage` conflict can be created **upstream**.
Stage 2 takes the top N by relevance with no awareness of coverage constraints, so
on a two-zone brief the top 80 of 2,230 eligible screens can be 80/80 in one zone
and the constraint becomes unsatisfiable before the solver sees it. The optimizer
reports it honestly; the fix belongs in stage 2's candidate selection.

------------------------------------------------------------------------

# 25. Explainability Requirements

Every important recommendation should have traceable reasons.

For a screen:

``` text
Why selected?
```

For a price:

``` text
Why this price?
```

For a time block:

``` text
Why this time?
```

For the package:

``` text
Why this combination?
```

For rejected screens:

``` text
Why not selected?
```

Explanations must reference actual features or model outputs.

Avoid generic statements such as:

> "This screen is highly relevant."

Instead:

> "This screen ranks highly because the surrounding zone has a high
> 18--34 population share, strong daytime population uplift, and high
> transit demand during the selected evening period."

------------------------------------------------------------------------

# 26. Evaluation Framework

Build evaluation into the implementation rather than adding it at the
end.

## Data layer

Test:

``` text
Join correctness
Null handling
Duplicate handling
Expected row counts
Referential integrity
```

## ML

Evaluate:

``` text
MAE
RMSE
MAPE where appropriate
R²
Calibration for booking probability
Baseline comparison
```

**Current coverage.** `backend/tests/test_pricing_engine.py` (26 tests)
asserts the invariants the pricing models must hold:

``` text
[BUILT] price coefficient is negative          the gate on trusting Model B
[BUILT] calibration matches the true base rate within 0.01
[BUILT] probability curve decreases with price
[BUILT] probabilities stay in [0,1] at extreme prices
[BUILT] negative class is only price-driven losses
[BUILT] floor <= target <= cap for every screen
[BUILT] industry adjustment stays inside its clamp
[BUILT] every screen attribute combination has a band
[BUILT] recommended price sits inside the band, positioned by occupancy
[BUILT] slot capacity is never exceeded
[BUILT] availability is the tightest day, not the average
[BUILT] infeasible rows retained with diagnostics
[BUILT] the fast occupancy index matches a plain groupby
```

**Audience coverage.** `backend/tests/test_relevance_engine.py` (18 tests) pins
the invariants that were actually broken or dangerous during integration, rather
than restating the implementation:

``` text
[BUILT] every score stays inside its 0-1 contract bound   family_score hit 1.140
[BUILT] all 12 impression columns exist, no NaNs
[BUILT] pool_key is never null, and there are exactly 1,004 pools
[BUILT] block 1 is empty -- the known gap, asserted so it cannot change silently
[BUILT] every audience term maps to a real score column and to blocks
[BUILT] geography is a HARD filter: every candidate is in the eligible set
[BUILT] allowed_screen_types is enforced before scoring
[BUILT] different audience terms give different blocks AND a different ranking
[BUILT] an off-vocabulary audience term is rejected, not scored as 0.5
[BUILT] a missing audience term is reported in defaults_applied
[BUILT] reasons cite real numbers, never "highly relevant"
[BUILT] pooled audience is far below the naive sum (dedupe is doing work)
[BUILT] the flight's weekday/weekend day mix is counted from real dates
[BUILT] the exposure model is applied exactly once, and both sides of the
        reach min() are in viewed units
[BUILT] candidates are ranked best-first
```

Plus, in `test_pipeline_smoke.py`, the adversarial one: **a package claiming
gross exposures as reach must fail validation.**

**Gaps that remain, both the same shape:** neither model ships a held-out accuracy
metric. The price band is descriptive (quantiles of comparables), so MAE/RMSE
against a held-out set was never computed. The audience model is an aggregation of
observed ridership rather than a fitted predictor, so it has no held-out split
either --- and no comparison against the naive
`route_schedules.estimated_ridership` baseline that section 7 asks for. That
absence is exactly why no stage emits a per-screen confidence. A supervised model
on top of these aggregates is the next step that would close it.

## Optimization

Target list, and what `backend/tests/test_optimizer.py` plus
`test_pipeline_smoke.py` actually assert:

``` text
[BUILT] Budget constraint          validator recomputes sum(price x slots x days)
[BUILT] Inventory constraint       purchased slots <= tightest-day availability
[BUILT] Geography constraint       every allocated screen in the eligible set
[BUILT] Date constraint            no line outruns the flight; start not in past
[BUILT] Requested screen count     exact count returns exactly that many DISTINCT
                                   screens -- the bug this test was written for
[BUILT] Objective improvement      MILP reach >= an independent greedy baseline
                                   re-implemented in the test at equal budget
[BUILT] Solution feasibility       validator accepts every package the pipeline
                                   produces, for all four optimization goals
```

Also asserted, because each was a real defect or a real hazard:

``` text
[BUILT] a pool with no resolvable population RAISES rather than defaulting to 1
[BUILT] the curve diagnostic stays within min(sum E, sum P) -- a bound that holds
        for any saturation constant, so it tests the bookkeeping and not lambda
[BUILT] no spend floor is invented, and a declared one is reported as a conflict
[BUILT] the wear-out cap is reachable: cap(days) > floor(days) for every flight
[BUILT] compare_objectives withholds a stacked plan, shows the rest, and the
        breadth-first plan really does reach more people than the depth-first one
[BUILT] the exposure model is applied exactly once, in one module
[BUILT] a package claiming gross exposures as reach fails validation
```

Not tested: solve time. It is 0.1s on the canonical brief at a 1% gap, but nothing
guards against a formulation change making it 40s again --- which the tangent version
did.

## End-to-end

Create fixed test scenarios:

### Scenario A --- Reach

``` text
High budget
Broad geography
Reach objective
```

### Scenario B --- Budget constrained

``` text
Low budget
High screen requirement
```

### Scenario C --- Geography constrained

``` text
Specific zone
Specific corridor
```

### Scenario D --- Time constrained

``` text
Evening only
Limited inventory
```

### Scenario E --- Infeasible

``` text
Impossible budget + inventory combination
```

The final system should clearly explain why Scenario E cannot be
fulfilled rather than generating a plausible-looking package.

------------------------------------------------------------------------

# 27. Implementation Sequence

Do not implement the entire multi-agent layer first.

Implement in this order:

## Phase 1 --- Data

``` text
1. Load CSVs
2. Create DuckDB database
3. Validate schemas
4. Validate relationships
5. Create SQL views
6. Build reusable data-access functions
```

## Phase 2 --- Features

``` text
7. Build screen profiles
8. Build transit features
9. Build demographic features
10. Build POI/event features
11. Build historical demand features
12. Build historical pricing features
```

## Phase 3 --- Models

``` text
13. Build screen relevance model
14. Build demand baseline
15. Build demand ML model
16. Build price model
17. Build booking probability model
```

## Phase 4 --- Optimization

``` text
18. Implement deterministic optimizer    <-- MILP, HiGHS via scipy. app/optimize/
19. Add campaign objectives              <-- four weight profiles, section 12
20. Add hard constraints                 <-- section 12
21. Add soft preferences                 <-- elastic coverage + wear-out cap
22. Add infeasibility handling           <-- section 24
```

Built last, as the sequence intended: the deterministic pipeline ran end to end
under a greedy fill for the whole of the rest of the build, which is what made
replacing it a contained change --- one tool module and one new package, no agent
rewiring.

## Phase 5 --- Agents

``` text
23. Implement Data Agent          <-- NOT DONE, deliberately. Stage 2 is a
                                      deterministic Master-owned tool; see s15.
24. Implement ML Agent
25. Implement OR Agent
26. Implement Master Agent
```

## Phase 6 --- Product

``` text
27. Implement structured recommendation
28. Add explanations
29. Add validation report
30. Add API/UI
31. Add test scenarios
32. Add logging/tracing
```

------------------------------------------------------------------------

# 28. Two-Day Hackathon Prioritization

If implementation time is extremely limited, prioritize the following.

## Day 1

### Data layer

``` text
CSV → DuckDB
```

Create:

``` text
v_screen_profile
v_screen_availability
v_screen_demand_history
v_historical_pricing
```

Implement:

``` text
build_screen_profiles()
calculate_availability()
build_demand_features()
build_pricing_features()
```

### Basic relevance scoring

Start with transparent weighted scoring.

### Basic demand model

Start with a strong historical baseline.

### Basic pricing model

Start with historical comparable pricing.

------------------------------------------------------------------------

# 29. Day 2

Build:

``` text
Campaign parser
        ↓
Screen scoring
        ↓
Demand forecast
        ↓
Pricing
        ↓
OR optimizer
        ↓
Recommendation
```

Then wrap these components in the Deep Agent architecture.

The first working system should prioritize:

``` text
Correctness
+
Explainability
+
Reliable constraints
+
Good demo experience
```

over sophisticated agent-to-agent behavior.

------------------------------------------------------------------------

# 30. Final Target Architecture

The completed system should conceptually look like:

``` text
                              USER
                                │
                                ▼
                    ┌─────────────────────┐
                    │    MASTER AGENT     │
                    │                     │
                    │ Orchestration       │
                    │ State               │
                    │ Verification        │
                    │ Final response      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
     │ RELEVANCE      │ │ ML AGENT      │ │ OR AGENT       │
     │ ENGINE [BUILT] │ │      [BUILT]  │ │      [BUILT]   │
     │ (a tool, not   │ │ Price band    │ │ Formulation    │
     │  an agent)     │ │ Occupancy     │ │ Constraints    │
     │ Joins/Features │ │ Booking prob. │ │ MILP solve     │
     │ Audience       │ │ Seasonality   │ │ Reach dedupe   │
     │ Volume/Reach   │ │ Unit mapping  │ │ Objectives     │
     └───────┬────────┘ └───────┬───────┘ └───────┬────────┘
             │                  │                 │
             ▼                  ▼                 ▼
      ┌────────────┐     ┌────────────┐    ┌────────────┐
      │ SQL / Data │     │ Python /   │    │ OR Solver  │
      │ Tools      │     │ ML Tools   │    │ Tools      │
      └─────┬──────┘     └─────┬──────┘    └─────┬──────┘
            │                  │                 │
            └──────────────────┼─────────────────┘
                               ▼
                     ┌──────────────────┐
                     │ SHARED ARTIFACTS │
                     │                  │
                     │ CampaignSpec     │
                     │ ScreenProfiles   │
                     │ Candidates       │
                     │ Economics        │
                     │ OptimizedPackage │
                     └────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ VALIDATION LAYER    │
                    │            [BUILT]  │
                    │ Budget              │
                    │ Inventory           │
                    │ Geography           │
                    │ Dates               │
                    │ Reach (recomputed)  │
                    │ Model confidence    │ <- skipped, none emitted
                    │ Reconciliation      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SALES RECOMMENDATION│
                    │                     │
                    │ Package             │
                    │ Price               │
                    │ Screens             │
                    │ Occupancy           │
                    │ Reach       ✓ dedup │
                    │ Impressions ✓ gross │
                    │ Frequency   ✓       │
                    │ Reasons             │
                    │ Risks               │
                    │ Alternatives        │
                    └─────────────────────┘
```

------------------------------------------------------------------------

# 31. Non-Negotiable Design Principles

The coding agent should follow these rules throughout implementation.

### 1. Do not over-agentify

Use one manager plus a specialist agent for each stage that genuinely *reasons*.

As built that is **two**: ML and OR. The Data specialist was dropped once it was
clear stage 2 has no judgement in it --- see section 15. The principle held; the
count was never the point. Prefer a Master-owned tool over a subagent whenever a
stage is a pure function of its inputs.

### 2. LLMs reason; tools calculate

Do not ask an LLM to perform numerical optimization or large-scale
calculations.

### 3. Keep business artifacts structured

Use Pydantic models for inter-agent contracts.

### 4. Do not pass huge DataFrames through agent context

Use artifact references.

### 5. Separate prediction from optimization

ML predicts demand/pricing; OR chooses the package.

### 6. Separate optimization from explanation

The solver generates the answer; the LLM explains it.

### 7. Hard constraints are deterministic

Never allow an LLM to "interpret" a hard budget or availability
violation away.

### 8. Every recommendation must be explainable

Store supporting factors alongside scores. `ScreenCandidate.reasons` cites real
feature values and `defaults_applied` records every neutral fallback, per row and
per pool --- an untraceable 0.5 is worse than an acknowledged gap.

### 8a. Reach is never the sum of exposures

Deduplicate on `pool_key` first. Screens at one stop see the same people, and
naively summing over-counts the audience ~23x on a realistic pool. Any new
consumer of audience numbers must respect this.

### 8b. A validated number may not depend on an assumed constant

If a figure the validation layer checks moves with a constant nobody can measure,
the check validates everything except the one thing that is actually uncertain.
This is why `REACH_LAMBDA` reaches only `curve_reach_diagnostic` while the reported
reach is the lambda-free `min()`, and why `curve_reach_bounded` asserts
`reach <= min(sum E, sum P)` --- true for any lambda --- instead of re-deriving the
curve in a second place.

The corollary for the exposure model: `LOOP_PASSES_PER_TRIP` and `VIEWABILITY_*` are
also assumed, and they *do* scale a reported figure. They are tolerable only because
they live in one module with one call site (`app/optimize/exposure.py`), are named in
the fields they produce, and are stated as assumptions wherever those fields are
explained. An assumed constant in two places is a defect waiting to happen.

### 9. Infeasibility must be explicit

Never fabricate a feasible-looking solution.

### 10. Models are replaceable

The artifact contracts must not depend on a particular ML model.

### 11. Validate every agent result

The Master Agent should verify specialist outputs before producing the
final answer.

### 12. Build the deterministic pipeline first

The agent layer should orchestrate a working analytical system, not
compensate for an incomplete one.

------------------------------------------------------------------------

# 32. Final Implementation Goal

## Where the experience stands today

``` text
User:
"I have $50K to promote a new product to young commuters
 in Downtown Core for 30 days."
                    ↓
Master Agent -> CampaignSpec                              [BUILT]
   budget, dates, geography resolved to LH / LH-ZONE-001
   audience_terms: [young_professionals, commuters]
                    ↓
Relevance    -> "1,601 eligible screens, kept top 250      [BUILT]
engine           across 23 audience pools. Relevance
(own tool,       0.649-0.749. Target blocks 2 and 5.
 no LLM)         Pooled daily audience 1.15M; the naive
                 sum would say 26.9M. No defaults fired."
                    ↓
ML Agent     -> "250 screens priced across blocks 2 and 5. [BUILT]
                 375 lines purchasable, 125 sold out.
                 Prices $41.97-$177.88, mean $100.13.
                 Mean occupancy 24%.
                 16,553 viewed exposures per slot per day."
                    ↓
OR Agent     -> "25 screens, $49,707.90, 99.4% of          [BUILT]
                 budget. Reach 261,329 people;
                 10,453,140 gross viewed exposures;
                 frequency 40.0. MILP, proven optimal
                 within a 1% gap, 0.1s."
                    ↓
Master Agent -> "Validated: 18 checks, 16 pass,            [BUILT]
                 2 skipped, 0 fail."
                    ↓
User receives:
  Recommended package        ✓
  Price + why                ✓  band, occupancy, adjustments
  Selected screens           ✓
  Budget utilization         ✓
  Expected reach             ✓  261,329 deduplicated people
  Viewed exposures           ✓  10.45M gross exposures
  Expected frequency         ✓  40.0x, with the wear-out disclosure
  Objective trade-off        ✓  reach vs awareness vs frequency
  Rationale                  ✓  demographics, POI, ridership, price
  Risks                      ✓
  Alternative packages       ✓
```

Every stage of the pipeline is now real and traceable end to end.

**What is still honestly imperfect, and said out loud rather than hidden:**

-   40 viewed exposures per person is past the point of usefulness, and no
    allocation can lower it: `LOOP_PASSES_PER_TRIP / 6 x days` is a floor set by
    the flight length, and there is no flighting model. The wear-out cap stops
    stacking beyond that; it cannot fix the floor. Dwell data would.
-   `LOOP_PASSES_PER_TRIP`, `VIEWABILITY_*` and `REACH_LAMBDA` are all ASSUMED,
    with no ground truth anywhere in the 14 CSVs. The first two scale every
    exposure figure this system quotes; the third reaches only a diagnostic.
-   That package is 16 `metro_station` + 9 `bus_stop` in 1 zone. The ~380x
    fixed/mobile volume gap (section 7) is still in the model --- pool saturation
    is what makes cheap bus-stop pools competitive, not a fix to the gap.
-   Time block 1 sells but models as zero audience, so a reach objective will
    never buy it.
-   No stage emits a confidence, because no model has a held-out error bar.
-   Relevance never enters the objective. Geography and `hard_constraints` filter
    the pool, then the top-N cut is by relevance; after that the solver optimizes
    audience and cost alone. That is defensible --- but it means a declared
    `min_zone_coverage` can be made unsatisfiable by the top-N cut, since stage 2
    ranks with no awareness of it (pinned by a test).
-   The bundle's `budget_sensitivity` tool is not wired in, so "20% more budget
    buys N more people" is not yet an answer the system can give.

## The target

The original target, retained for reference. The pipeline above now reaches it,
with the caveats listed there. The one structural difference: "Data Agent" is a
Master-owned deterministic tool, not a delegated specialist.

``` text
User:
"I have $50K to promote a new product to young commuters
in the eastern part of the city for 30 days."

                    ↓

Master Agent

                    ↓

CampaignSpec

                    ↓

Relevance engine
"These 250 screens are relevant."

                    ↓

ML Agent
"These screens have the strongest expected demand and
these are the expected prices."

                    ↓

OR Agent
"This package maximizes expected reach under the $50K
budget and inventory constraints."

                    ↓

Master Agent
"Validated: all constraints satisfied."

                    ↓

User receives:

Recommended package
+ price
+ selected screens
+ expected reach
+ expected impressions
+ rationale
+ risks
+ alternative packages
```

The key outcome is not simply an "AI agent that recommends screens."

The system should demonstrate a complete decision pipeline:

``` text
Natural language
      ↓
Structured business intent
      ↓
Data-driven audience understanding
      ↓
Inventory relevance
      ↓
Demand prediction
      ↓
Dynamic pricing
      ↓
Mathematical optimization
      ↓
Constraint validation
      ↓
Explainable sales recommendation
```

That is the architecture the coding agent should implement.

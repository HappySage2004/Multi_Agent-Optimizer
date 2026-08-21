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

The system should **not over-agentify** the workflow. The recommended
implementation has one Master Deep Agent supervising three specialist
agents:

1.  **Data Intelligence Agent**
2.  **ML / Forecasting Agent**
3.  **OR / Optimization Agent**

The Master Agent owns orchestration, verification, and final response
generation.

------------------------------------------------------------------------

# 1. End-to-End Architecture

Implement the following logical pipeline:

``` text
                         ┌─────────────────────┐
                         │   Campaign Brief    │
                         │       / Query       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  1. Brief Intake &  │
                         │     Normalization   │
                         └──────────┬──────────┘
                                    │
                           CampaignSpec
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │ 2. Audience & Context       │
                    │    Intelligence              │
                    └─────────────┬───────────────┘
                                  │
                         ScreenProfiles
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 3. Campaign-Screen          │
                    │    Relevance Scoring         │
                    └─────────────┬───────────────┘
                                  │
                      RankedScreenCandidates
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 4. Demand Forecast +        │
                    │    Pricing                  │
                    └─────────────┬───────────────┘
                                  │
                       ScreenEconomics
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 5. Inventory / Slot         │
                    │    Optimization              │
                    └─────────────┬───────────────┘
                                  │
                       OptimizedPackage
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ 6. Recommendation &         │
                    │    Explanation               │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                         Sales Recommendation
```

The six logical stages above are implemented by the three specialist
agents plus deterministic tools. Do not create six independent LLM
agents solely because there are six business stages.

------------------------------------------------------------------------

# 2. Central State Object --- CampaignSpec

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

    start_date: date
    duration_days: int
    budget: float

    requested_num_screens: int | None = None

    preferred_dayparts: list[str] = []
    preferred_time_blocks: list[str] = []

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

------------------------------------------------------------------------

# 3. Step 1 --- Campaign Brief Intake

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

# 4. Step 2 --- Audience & Context Intelligence

Do not implement this as a giant LLM-only "Audience Profile Agent."

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

# 5. Step 3 --- Campaign-to-Screen Relevance Scoring

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

    audience_match_score: float
    geography_score: float
    contextual_score: float
    transit_score: float

    reasons: list[str]

    hard_constraints_passed: bool
```

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

## Scoring implementation

Start with a transparent weighted scoring model if there is insufficient
historical labeled data.

Example:

``` text
relevance_score =
    w1 * audience_match
  + w2 * geography_match
  + w3 * transit_score
  + w4 * contextual_score
```

If historical campaign outcomes provide enough labels, replace or
augment this with LightGBM/XGBoost or another supervised model.

The score must be explainable.

------------------------------------------------------------------------

# 6. Step 4 --- Demand Forecasting & Pricing

Expose this as one business capability but internally separate it into:

``` text
Demand Forecast
Pricing Model
Booking Probability Model
```

Architecture:

``` text
             Demand & Pricing Engine
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    Demand Model   Inventory    Pricing /
                   Pressure     Booking Model
```

------------------------------------------------------------------------

# 7. Demand Forecasting

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
results.

------------------------------------------------------------------------

# 8. Pricing Model

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

Build two related models where data permits.

## Model A --- Market Price Model

Predict:

``` text
expected_market_price
```

## Model B --- Booking Probability Model

Predict:

``` text
P(booking | price, client, campaign, inventory)
```

Then calculate:

``` text
expected_revenue =
    price × booking_probability
```

This allows the system to recommend a price based on expected revenue
rather than simply predicting a historical average price.

------------------------------------------------------------------------

# 9. ScreenEconomics

The optimizer should receive a consolidated economics object.

``` python
class PricingRecommendation(BaseModel):
    floor: float
    target: float
    cap: float
    recommended_price: float
    booking_probability: float
    confidence: float


class DemandForecastSummary(BaseModel):
    expected_impressions: float
    demand_index: float
    confidence: float


class TimeSlotAvailability(BaseModel):
    date: date
    time_block_id: str
    available_slots: int


class ScreenEconomics(BaseModel):
    screen_id: str

    availability: list[TimeSlotAvailability]

    demand_forecast: DemandForecastSummary

    pricing: PricingRecommendation

    expected_impressions: float

    expected_revenue: float

    confidence: float
```

Example:

``` json
{
  "screen_id": "LH-SCR-001928",

  "availability": [
    {
      "date": "2026-09-01",
      "time_block_id": "5",
      "available_slots": 4
    }
  ],

  "demand_forecast": {
    "expected_impressions": 18400,
    "demand_index": 0.87
  },

  "pricing": {
    "floor": 61.0,
    "target": 74.0,
    "cap": 91.0,
    "recommended_price": 76.0,
    "booking_probability": 0.72,
    "confidence": 0.84
  },

  "expected_revenue": 54.72,
  "confidence": 0.84
}
```

------------------------------------------------------------------------

# 10. Step 5 --- Inventory Optimization

The optimization layer must be mathematical, not LLM-based.

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

# 11. Optimization Decision Variables

For example:

\[ x\_{s,t} `\in `{=tex}{0,1,2,3,`\ldots`{=tex}} \]

where:

``` text
x[s,t] = number of slots purchased
         on screen s
         during time block t
```

Additional variables can be introduced for:

-   screen selection
-   campaign duration
-   geography coverage
-   bundle selection
-   frequency
-   reach approximations

Keep the formulation as simple as possible while preserving business
value.

------------------------------------------------------------------------

# 12. Optimization Objectives and Constraints

Objective should depend on `CampaignSpec.optimization_goal`.

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

# 13. OptimizedPackage

``` python
class Allocation(BaseModel):
    screen_id: str
    time_block_id: str

    slots_per_day: int
    duration_days: int

    price_per_slot_per_day: float

    expected_impressions: float
    expected_revenue: float


class OptimizedPackage(BaseModel):
    allocations: list[Allocation]

    total_cost: float
    expected_impressions: float
    expected_reach: float
    expected_frequency: float

    budget_utilization: float

    constraint_status: dict[str, bool]

    objective_value: float

    optimization_method: str
```

Example:

``` json
{
  "total_cost": 48750,
  "expected_impressions": 1250000,
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
      "expected_impressions": 552000,
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

------------------------------------------------------------------------

# 14. Step 6 --- Recommendation Generator

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

------------------------------------------------------------------------

# 16. Data Intelligence Agent

## Responsibilities

The Data Agent handles:

-   campaign data interpretation
-   database exploration
-   table joins
-   filtering
-   feature engineering
-   audience analysis
-   screen profiling
-   relevance scoring
-   data quality checks

## Tools

Conceptually:

``` python
query_database()
get_table_schema()
build_screen_profiles()
filter_inventory()
score_screens()
summarize_data()
```

## Input

``` text
CampaignSpec
```

plus access to the data layer.

## Output

``` text
ScreenCandidate[]
ScreenProfile summary
Data quality diagnostics
```

The Data Agent should return concise structured results rather than raw
DataFrames.

------------------------------------------------------------------------

# 17. ML / Forecasting Agent

## Responsibilities

The ML Agent handles:

-   feature inspection
-   target selection
-   model selection
-   training
-   validation
-   model comparison
-   demand prediction
-   pricing prediction
-   booking probability prediction
-   confidence estimation
-   model failure/retry logic

## Tools

``` python
run_python()
train_forecaster()
evaluate_forecaster()
predict_demand()
train_price_model()
predict_price()
train_booking_probability_model()
evaluate_model()
```

The ML Agent should reason over model performance.

Example internal flow:

``` text
inspect data
    ↓
define target
    ↓
build baseline
    ↓
train candidate models
    ↓
cross-validate
    ↓
compare metrics
    ↓
reject poor models
    ↓
select model
    ↓
generate predictions
```

Do not force the LLM to perform numerical calculations itself.

------------------------------------------------------------------------

# 18. OR Agent + Master Agent + Shared Artifact Architecture

## OR Agent

Responsibilities:

-   interpret optimization objective
-   formulate decision variables
-   construct constraints
-   solve optimization
-   diagnose infeasibility
-   generate alternatives
-   explain tradeoffs

Tools:

``` python
build_optimization_model()
solve_optimization()
check_feasibility()
analyze_solution()
generate_alternative_solution()
```

The OR solver should be deterministic.

Preferred implementation can use a suitable MILP/CP-SAT/optimization
library depending on project constraints.

------------------------------------------------------------------------

## Master Agent

The Master Agent is the only component responsible for end-to-end
orchestration.

Workflow:

``` text
USER
 │
 ▼
MASTER
 │
 ├── Understand Campaign
 │
 ├── Ask Data Agent
 │      │
 │      └── ScreenCandidates
 │
 ├── Ask ML Agent
 │      │
 │      └── ScreenEconomics
 │
 ├── Ask OR Agent
 │      │
 │      └── OptimizedPackage
 │
 ├── VERIFY
 │      │
 │      ├── Budget ✓
 │      ├── Inventory ✓
 │      ├── Dates ✓
 │      ├── Geography ✓
 │      ├── Model confidence ✓
 │      └── Optimization ✓
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

------------------------------------------------------------------------

# 19. Shared Artifact Architecture

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

# 20. Data Layer

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

  Optimization            `CampaignSpec` +        `OptimizedPackage`
                          economics               

  Recommendation          All previous artifacts  `CampaignRecommendation`
  --------------------------------------------------------------------------

These are the interfaces that should remain stable even if internal
models are replaced.

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

The LLM agents should decide **when and why** to call these tools.

The tools should decide **how** to perform the calculations.

------------------------------------------------------------------------

# 24. Error Handling and Infeasibility

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

## Optimization

Test:

``` text
Budget constraint
Inventory constraint
Geography constraint
Date constraint
Requested screen count
Objective improvement
Solution feasibility
```

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
18. Implement deterministic optimizer
19. Add campaign objectives
20. Add hard constraints
21. Add soft preferences
22. Add infeasibility handling
```

## Phase 5 --- Agents

``` text
23. Implement Data Agent
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
     │ DATA AGENT     │ │ ML AGENT      │ │ OR AGENT       │
     │                │ │               │ │                │
     │ Joins          │ │ Demand        │ │ Formulation    │
     │ Features       │ │ Pricing       │ │ Constraints    │
     │ Audience       │ │ Prediction    │ │ Optimization   │
     │ Screening      │ │ Evaluation    │ │ Alternatives   │
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
                     │ Economics       │
                     │ OptimizedPackage │
                     └────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ VALIDATION LAYER    │
                    │                     │
                    │ Budget              │
                    │ Inventory           │
                    │ Geography           │
                    │ Dates               │
                    │ Model confidence    │
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
                    │ Reach               │
                    │ Impressions         │
                    │ Reasons             │
                    │ Risks               │
                    │ Alternatives        │
                    └─────────────────────┘
```

------------------------------------------------------------------------

# 31. Non-Negotiable Design Principles

The coding agent should follow these rules throughout implementation.

### 1. Do not over-agentify

Use three specialist agents plus one manager.

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

Store supporting factors alongside scores.

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

The final user experience should be:

``` text
User:
"I have $50K to promote a new product to young commuters
in the eastern part of the city for 30 days."

                    ↓

Master Agent

                    ↓

CampaignSpec

                    ↓

Data Agent
"These 214 screens are relevant."

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

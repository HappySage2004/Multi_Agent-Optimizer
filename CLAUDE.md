# CLAUDE.md

Transit Media Campaign Recommendation System — hackathon build. Natural-language
campaign brief in, explainable sales-ready media package out.

Full specs live in the repo root and are the source of truth:
- [SOLUTION.md](SOLUTION.md) — architecture, 6-stage pipeline, agent design, Pydantic contracts
- [DATASETS.md](DATASETS.md) — the 14 source CSVs, join map, data-quality traps
- [UI.md](UI.md) — UI spec (design system, 3-panel layout, component tree). The static
  `UI-handoff.html` mockup this was written against has been deleted — UI.md is now the
  only UI source of truth, so do not go looking for the HTML.

Read the relevant spec before implementing a stage. Do not re-derive them here.

## Repository structure

```
backend/          Python 3.11+ / FastAPI / LangGraph Deep Agents
  app/
    main.py               FastAPI entrypoint (/health, sessions, uploads, campaign)
    config.py             Settings; all paths resolve from the repo root
    api/                  schemas.py, sessions.py, messages.py, uploads.py, campaign.py,
                          providers.py (GET /models -- the model picker's catalogue)
    agents/
      master.py           build_master_agent() -- create_deep_agent + 2 subagents
      providers.py        model provider registry: catalogue, clients, rate limiters
      subagents.py        aggregator: assembles the two specs in pipeline order
      ml_agent.py         ML/Pricing specialist: name + description + prompt + build()
      or_agent.py         OR specialist          (one file per agent)
      prompts.py          MASTER_SYSTEM_PROMPT only
      validation.py       deterministic package verification (master-owned)
    tools/
      master_tools.py     intake, geography, read_campaign_document, verify, inspect
      relevance_tools.py  AUDIENCE RELEVANCE ENGINE + its tools, one file.
                          Master-owned, no subagent. Stage 2.
      ml_agent_tools.py   ML Agent surface -- thin wrapper over app/ml/
      or_agent_tools.py   OR Agent surface -- thin wrapper over app/optimize/,
                          plus the reach accounting the validator re-derives
    models/               Pydantic contracts -- the stable inter-agent interfaces
    data/                 db.py (DuckDB + views), reference.py (validation lookups)
    services/             artifact_store.py, local_db.py, run_state.py, session_titles.py,
                          transcripts.py (the persisted chat log),
                          documents.py (brief parsing -- the only place upload bytes
                          are interpreted)
    ml/                 pricing engine (see below). occupancy, price_band,
                        booking_probability, seasonality, impressions, price_optimizer,
                        demand_value (merit vs realized price -> mispricing premium),
                        client_profile (negotiation history -> advisory flag, no price),
                        levers (the agent-tunable parameter surface),
                        engine (singleton), loaders (DuckDB)
    optimize/           the MILP. config (constants, provenance-tagged), exposure
                        (people passing -> viewed exposures, ONE implementation),
                        contract (input validation), pooled (pool population +
                        the diagnostic curve), solver (the formulation)
    features/           empty. The relevance engine lives in tools/relevance_tools.py
                        by design decision, not here.
  tests/                  conftest.py isolates storage; test_pipeline_smoke.py,
                          test_relevance_engine.py, test_pricing_engine.py,
                          test_pricing_levers.py, test_demand_value.py,
                          test_client_profile.py, test_optimizer.py,
                          test_transcripts.py, test_artifact_store.py,
                          test_session_titles.py, test_followup_turns.py,
                          test_documents.py, test_providers.py
  pyproject.toml, .env.example
frontend/         Next.js (App Router) + TypeScript + Tailwind
  src/app/            routes
  src/components/     layout/ chat/ inspector/  (see UI.md component tree)
                      layout/SettingsDialog.tsx is the model picker
  src/lib/            API client, types mirroring backend Pydantic models
datasets/         Read-only source CSVs. Never write here.
localDB/          JSON files = the app database (sessions, chat transcripts, runs, uploads)
stage/            User-uploaded campaign documents, one subfolder per session
.gitignore        Root only — do not add nested .gitignore files
```

Monorepo: all code lives under `backend/` or `frontend/`. Nothing at root except
docs, `datasets/`, `localDB/`, `stage/`, and config.

## Three distinct data layers — do not conflate

- `datasets/*.csv` — the 166 MB analytical source. Loaded once into DuckDB
  (`backend/app/data/`), queried through SQL views (`v_screen_profile`,
  `v_screen_availability`, `v_screen_demand_history`, `v_historical_pricing`).
  `datasets/ridership_actuals.csv` is 124 MB / 2.05 M rows, gitignored, and must be
  provisioned separately. Never `cat` it or `read_csv` it without `usecols`/`dtype`.
- `localDB/*.json` — application persistence: chat sessions, **chat transcripts**,
  campaign specs, run history, cached recommendations. Plain JSON files, read/written via
  `backend/app/services/`. One file per collection (`sessions.json`, `messages.json`,
  `runs.json`, `uploads.json`).
  Application data only — no analytical data, no uploaded file bytes.
  **Record order is insertion order and callers may rely on it** — `local_db._load`
  preserves it and `update`/`delete` rewrite in place. That is what makes a transcript
  replayable without a sort key: `created_at` has second granularity, so two messages in
  one turn tie and sorting on it would be unstable.
  **Ignored by default, with the chat data as a deliberate exception.**
  `messages.json` and `sessions.json` are committed — the transcripts are the
  demonstration of what this system does, and a clone with no sessions opens on an empty
  rail; `sessions.json` follows because a transcript is keyed to a session and cannot be
  rendered without one. `runs.json` and `uploads.json` are **not**, and committing them
  caused a real bug — see the artifact-portability note below. The cost of the split is
  known and accepted: a fresh clone replays every conversation, but the inspector's D1-D4
  tabs are empty, because the run records those tabs read are absent.
  Large intermediate artifacts (screen candidates, economics) go to
  `backend/artifacts/` as parquet, referenced by ID — not into localDB or agent context.
- `stage/{session_id}/` — documents the user uploads alongside a query (briefs, RFPs,
  client decks), plus a `<file>.extracted.txt` sidecar per readable one. The upload
  endpoint writes the file, parses it once, and records only metadata plus an extraction
  summary in localDB — never document text. Treat contents as untrusted input: validate
  type and size, never execute, never pass a whole document into agent context. See
  "Reading uploaded briefs" below.

## Current implementation status

**No stage is a stub.** Every artifact carries `provenance="computed"`, and
`run_state.stub_stages()` returns `[]` on a normal run. The stub scaffolding
(`_stub_support.py`) is deleted; the `stub_stages` plumbing stays, because it is the
mechanism that would surface a regression.

| Component | State |
|---|---|
| Master Agent, prompts, orchestration | real |
| Brief intake + geography resolution | real |
| Audience relevance engine (`tools/relevance_tools.py`) | real |
| Audience volume — impressions, reach, frequency | real |
| Pricing engine (`app/ml/`) — bands, booking probability, occupancy, seasonality | real |
| Validation layer (`agents/validation.py`) — 19 checks | real |
| DuckDB reference layer, artifact store, localDB, run state | real |
| FastAPI + SSE endpoints | real |
| Inventory optimization | real — MILP (HiGHS via scipy), `app/optimize/` |

**No stage is a heuristic any more either.** The greedy fill was replaced by a MILP ported
from an OR handoff bundle. `optimization_method` reports the solver, the objective, the gap
and whether optimality was proven, e.g.
`milp_highs_pooled_reach_min[reach,gap<=1%,optimal]`; a `feasible` status means a valid plan
inside the gap, not an error.

What the port changed, and why, is documented at each site in `app/optimize/solver.py`. The
four that matter: reach is bounded by the exact `min(E, P)` rather than by tangent lines on
an exponential curve with an assumed constant (the bundle's version left measured audience
on the table and turned a 0.1s solve into a 30s timeout); the screen-count constraint counts
screens rather than (screen x block) cells; no 90% spend floor is invented, since no spec
declares one; and the wear-out cap is a multiple of the flight's unavoidable exposure floor
rather than an absolute number, because an absolute one is satisfiable by no plan.

### Pricing

`app/ml/` is a faithful port of the pricing handoff package — verified output-identical to
the original on 180 rows x 15 fields, with a bit-identical booking-probability training
report (`price_coefficient = -1.1803182351670876`). Only I/O and two signature fixes
changed; see each module's "Port note".

What it computes, all from real data:
- **Price band** — p25/p50/p90 of comparable `bookings`, segmented
  size x type x position x **zone** x daypart with a bounded 5-rung fallback ladder
  (zone+daypart → zone → city+daypart → city → attributes only).
  Zone sits above city on measurement: holding city, size, type and position fixed, the
  median contracted price still varies **1.87x–2.52x across zones of one city**, and
  segmenting on city alone quoted all of them from a single blended band. Coverage
  supports the depth — 95.6% of bookings sit in a zone+daypart cell with n≥30, 99.7% in a
  zone cell. Zone is NULL for all 2,615 mobile screens, so those rungs self-skip and
  mobile falls through to city with no branch on inventory class.
  Each location rung is **also split by deal shape** (`is_bundle`) and tried split-first.
  A non-bundle deal holds exactly one screen (max 1); a bundled one a median of 20 — so a
  package this system builds is commercially a bundle and a single-screen quote is not.
  How big the effect is depends on the measurement, and the two readings disagree: by mean
  price index at city grain it looks inert (1.0617 vs 1.0459), but by the **actual band
  quantiles at zone grain** — the cells the ladder now resolves at — single-screen
  comparables sit at floor ×1.090, target ×1.079, cap ×1.065 across the 302 cells where
  both shapes clear n≥30. The quantile reading governs, because the price formula consumes
  quantiles and a mean of per-booking ratios against a blended median is a different
  statistic. End to end this moves a single-screen quote ~3.8% above a bundled one.
  Shape is the **first** dimension surrendered when a cell is thin (worth 6.5–9%, against
  zone's 87–152%); `is_bundle=None` means "don't split", never a guess.
  One dimension was **tested and rejected**, documented in `price_band.py` so nobody
  re-adds it on intuition: `duration_days` (~2% between the buckets most campaigns fall
  in, once bundle is controlled, and non-monotone on thin non-bundle cells).
- **Recommended price** — `floor + occupancy_rate x (cap - floor)`. Occupancy drives it.
  These are **seller-side** prices. The `price x P(booked)` argmax from SOLUTION.md 8 was
  tried and rejected: it degenerates to the cap for every screen, because within one
  segment's band predicted `P(booked)` moves only ~0.1-0.2%.
- **Booking probability** — calibrated logistic model, `bookings` vs price-driven
  `lost_leads` (191,109 vs 393). Reported as a diagnostic, never as the price driver.
  Guarded by a price-coefficient sign check.
- **Availability** — day-by-day slot occupancy, 6 slots per screen per block per day.
  `max_slots_per_day` is the **tightest single day** across the flight.

What it does **not** compute: audience volume. That belongs to the relevance engine
below. `pricing_internal_reach_proxy` is still **quarantined** and still not mapped onto any
exposure field: its multipliers are hand-set with no ground truth, and its fixed
and mobile paths are on different units (`est_daily_footfall` is per day; corridor
`estimated_ridership` is per departure) — a ~36x gap. It was never reach; the relevance
engine's ridership figures are.

Two upstream behaviours preserved as-is and flagged in code: the day-of-week multiplier
scales **price** rather than demand (mean 0.913 over a full week, so every campaign takes a
~9% haircut off a band already built from contracted prices), and the holiday multiplier is
inert (`ridership_actuals` spans 2026-02-19..2026-08-19 and holds exactly 2 holiday dates).
Now that a real demand signal exists, the day-of-week factor arguably belongs on demand
rather than on price — revisit it there.

The engine is a **process singleton** — `get_pricing_engine()`. `build()` costs ~9s.
Never construct it per request.

#### Demand value — the one signal that can disagree with history

`app/ml/demand_value.py` (M7). Everything else in `app/ml/` answers "what did screens like
this sell for?", which makes the engine a mirror: a screen always sold cheap is quoted
cheap forever. This is the second opinion, and it **never sees a price**.

`merit` scores a screen on what it physically delivers — riders (0.50), zone income (0.20),
daytime activity (0.15), POI draw (0.15) — as a percentile **within its own screen_type x
city**, matching the segment the realized price index is normalized against. Compare merit
against what the screen has actually transacted at; the gap is the finding.

**Why it must not be fitted to price**, since the obvious alternative fails quietly: a
model predicting price from location and audience treats the historical average as correct,
so a systematically underpriced *category* is learned as correct and reports nothing. It
finds deviation from a norm, never a wrong norm. Measured: top-quintile bus stops carry
6.2x the riders of the bottom quintile and 1.59x the price.

Four gates, and each withholds real screens (385 of 6,690 eligible end up flagged):
- **fixed inventory only** — merit/absorption correlation is +0.37 bus_stop and +0.31
  metro_station but **−0.28 bus and −0.20 metro_rail_coach**. Mobile also has no zone
  demographics, so 3 of 4 merit components are structurally zero, which is why their
  merit/price-rank correlation is 0.02–0.14 against ~0.5 for the fixed types.
- **≥10 bookings** or there is no reliable read on what the screen sells for.
- **merit ≥ 0.50** — a weak screen priced weakly is the market being right. The residual
  alone flags 191 below-median-merit screens.
- **absorption ≥ 0.30** — withholds 281 screens averaging **283k riders/day, the highest of
  any bucket**, on an absorption rank of 0.166. This gate separates "undervalued" from
  "unwanted" and is the reason the model is defensible.

**A ranking trap this hit once and must not hit again.** `pr_price` is a percentile among
screens that *have* a price index. Ranking the NULL-price screens in the same partition
squeezed 181 priced bus_stop/ACS screens into ranks 0.000–0.257 instead of 0.000–1.000 —
every price rank depressed, every residual inflated, and screens transacting *above* their
comparables flagged as underpriced (811 premiums instead of 385). Hence the extra
`(price_index IS NULL)` partition key in `DEMAND_VALUE_SQL`.

The premium ramps linearly from residual 0.20 (+0%) to 0.60 (**+15%, the cap**) so the gate
is not a cliff. Capped conservatively on purpose: at the observed mean of ×1.056 the flagged
set moves from an average transacted 64.51 to 68.14, still well **below** the 88.31 of
screens it leaves alone and far below the 118.23 it flags as overpriced — it narrows a gap
rather than closing it. Risk is measured, not assumed — deals lost to price wanted a third
off (`price_gap_pct` 0.328), deals lost to competitors 0.025.

**This is the only adjustment that may carry a quote above the band cap**, deliberately: an
underpriced screen's own comparables are what understate it, so clamping back inside the
band would delete the correction. A test asserts nothing *else* can exceed the cap.

Auto-applied at `demand_premium_weight=1.0`. Self-correcting: if a raised price stops a
screen selling, occupancy falls and `floor + occupancy x (cap - floor)` pulls it back down
next run. No held-out metric exists and never will from this data — validate forward by
tracking whether flagged screens keep their occupancy.

#### Client negotiation profile — a flag for the rep, not a price input

`app/ml/client_profile.py` (M8). Wired to **no engine, no lever default and no artifact**.
It answers one question for the salesperson: *how has this client behaved on price before?*
Surfaced by the Master-owned, read-only `master_tools.get_client_negotiation_profile`, which
resolves by client_id or company name and returns `ambiguous` rather than guessing.

**Why advisory is structural, not squeamish.** Price-driven loss rates are **flat** across
leverage tiers (34.2% / 32.5% / 34.8%), so posture says nothing about *whether* you lose the
deal — only what you settle at. Applying it automatically would be pricing off the half of
the finding that doesn't hold. `suggested_commercial_multiplier` exists, capped to
[0.90, 1.15], and only the rep may act on it via `set_pricing_levers`.

**`negotiation_leverage` is context, never a forecast** — and the module says so in three
places because it is genuinely misleading:

| weighting | high | medium | low | |
|---|---|---|---|---|
| per line item (median) | 0.965 | 1.017 | 1.019 | monotone, looks clean |
| per **client** (median) | 1.039 | **1.013** | 1.053 | ordering breaks |

The label tracks *account size*, not price behaviour: high-leverage clients carry a median
328 priced line items against 165–172. Weight by volume and the big accounts dominate;
give each client one vote and the ordering inverts. Since the question is about a client,
the per-client figure is the honest one.

**What it leads with instead**, both measured per client: their own realized price index
(mean of `price ÷ its own segment median`) judged against **its own standard error**, and
their own recorded price objections. Confidence is deliberately about the *departure*, not
the sample size — a 1.001 index on 500 line items is precisely measured and says nothing,
so it scores "weak" and suggests no change. Within-client spread (~0.214) is about as wide
as between-client spread (p10 0.956 → p90 1.239), which is exactly why the SE gate exists.

Two data traps handled: `price_gap_pct` is not populated on every price-driven lead, so a
missing gap reads "not recorded" rather than "0% off" (which would tell a rep the opposite
of the truth); and the segment medians include the client's own bookings, which pulls their
index toward 1.0 — conservative, not wrong, and not currently corrected.

#### Pricing levers — the parameters an agent turn may move

`app/ml/levers.py`. Every multiplier used to be derived once and applied on every run, so a
second turn could not act on anything the sales rep said. `PricingLevers` is that surface:
`seasonality_weight`, `event_weight`, `industry_weight`, `occupancy_gamma`, `band_position`,
`commercial_multiplier`, `respect_band_floor`, plus a free-text `note`.

Four rules hold it together, and each is pinned by a test in `tests/test_pricing_levers.py`:

1. **All defaults are identity**, where identity means "use the model's derived value" —
   `seasonality_weight=1.0` applies the derived seasonality multiplier, and
   `demand_premium_weight=1.0` applies the derived demand premium. Only that last one
   *does* something at its default, because the demand premium is auto-applied by design;
   set it to 0.0 to price purely off historical comparables.
2. **Clamped in code, not in the prompt.** `clamp()` bounds and *reports* rather than
   raising — a rejected tool call in an agent loop becomes a retry against a per-minute rate
   limit to reach the number clamping returns directly. `industry_weight` is re-clamped
   after weighting, so the band module's -15/+20% guarantee survives any weight.
3. **No lever reaches the feasibility gate.** Availability, occupancy and the band's
   comparables are inventory truth. A sold-out screen stays sold out.
4. **Weights dial INFLUENCE, not magnitude.** `effective_multiplier(m, w) = 1 + w(m-1)`, so
   `w=0` means exactly "term off" whether the multiplier sits above or below 1.0. Scaling
   would turn a 1.15 event premium into 0.0.

Levers live on the **run**, not in a delegation message: the Master sets them with
`master_tools.set_pricing_levers`, and `estimate_screen_economics` reads them off run state.
A float that has to survive an LLM paraphrase is a float that will eventually arrive wrong —
same "run handles, not payloads" rule the artifacts follow. They appear in
`get_active_run`'s `campaign_inputs` (so moving one is a REBUILD, not a question), on the
`screen_economics` artifact summary, and in `inspect_package`, because **a quote a human
moved is a different claim from a quote the model derived** and the recommendation has to
say which one it is.

`seasonality_weight=0.0` is the documented answer to the day-of-week double-count below.
The two seasonality terms are weighted *separately* and only then multiplied, so a rep can
keep the event premium while dropping the weekday haircut;
`SeasonalityAdjustment.combined_multiplier` is the unweighted product and is therefore not
what the engine applies any more — `ScreenEconomics.seasonality_multiplier` reports the
figure the price was actually computed from.

When the client-elasticity and screen-demand factors land, they arrive as further fields on
`PricingLevers` with identity defaults. `effective_multiplier` stays the one place a weight
becomes a number.

### Audience relevance and volume

`app/tools/relevance_tools.py` is the whole stage-2 capability in one file: the feature
layer, the scoring model and the three tools the Master calls. It is **deterministic** — no
LLM, no subagent. It began as a port of the audience relevance notebook, verified
output-identical at the time on all 12 impression columns, POI footfall and POI counts
across 11,163 screens, and has since diverged deliberately — every departure is flagged at
its site in the module docstring.

Feature layer in DuckDB (`app/data/db.py`): `v_screen_profile` (demographics + POI +
`pool_key`) and `v_screen_demand_history` (avg daily riders per screen x block x day type),
built on `v_schedule_block`, `v_route_stop_weight`, `v_route_block_demand`,
`v_corridor_block_demand`, `v_location_site`, `v_screen_poi`.

**A route's riders are shared between its stops, not multiplied by them.** This is the
single most consequential number in the system and it was wrong. `v_screen_demand_history`
used to credit every stop on a route with that route's WHOLE ridership, so one rider was
counted once per stop they passed. Measured: summing the modelled audience of the stops
along a corridor came to **20.4x (median) that corridor's own ridership**. Bus routes
average 12.58 stops and metro routes 13.86, which is the scale of it — but do not hard-code
a divisor, because it varies per route.

`v_route_stop_weight` gives each stop `weight / sum(weight over its route)`, weighting
termini at `TERMINUS_WEIGHT = 1.5`. Shares sum to exactly 1.0 per route, so a corridor's
stops now sum to **exactly 1.000x** its ridership against the same source (~0.97x across
sources, which is the scheduled-vs-observed gap, not an accounting error). The terminus
weight is an ASSUMPTION and safely so — it only redistributes a route's riders between its
own stops, and the corridor total is identical at any value. Multiple routes at one stop
still SUM: that is genuinely more people. **The mobile path is untouched** — measured
identical to 0.1 riders — so the fixed:mobile comparison stays meaningful.

Correcting this cut fixed-screen volume ~13-16x (metro_station median 227,981 -> 14,873;
bus_stop 3,112 -> 250) and with it every reach figure the system reports. That is the
intended outcome, not a regression.

Relevance is a transparent weighted sum of five 0-1 components:
`0.40 audience + 0.20 geography + 0.15 context + 0.15 time_of_day + 0.10 booking history`.
`transit_score` is reported as a volume percentile but is **not** in that sum — volume is
the optimizer's objective quantity, not a measure of fit.

**Four units that are easy to confuse, and must not be:**
- A **time block** is a 4-hour window. A **slot** is one of 6 rotation positions cycling
  continuously through it, so slot POSITION is meaningless — `slots_booked_per_day` is share
  of voice. Holding k of 6 slots puts the creative on k of every 6 loop passes, so viewed
  exposures are **linear** in k.
- `ScreenCandidate.impressions_by_block` is whole-block daily **people passing** the pool.
  No viewability discount is applied there.
- `ScreenEconomics.reachable_daily_audience` is the **reach ceiling**: people who *look*
  (`x viewability`). Distinct people, so it does not scale with slots or days.
- `ScreenEconomics.viewed_exposures_per_slot_per_day` is what ONE slot earns on ONE day:
  `people passing x LOOP_PASSES_PER_TRIP / 6 x viewability`. Scales with slots x days.

`daily_unique_audience` is retained as the upstream people-passing figure for traceability
and is **not** the ceiling. The conversion lives in exactly one module,
`app/optimize/exposure.py`, called from exactly one place — the constants are ASSUMED, so
the single call site matters more than the values.

**A pool's reach ceiling is `pool_reachable_daily_audience`, and every screen in a pool must
agree on it.** The per-screen `reachable_daily_audience` is NOT the pool's: a vehicle's figure
is its share of the corridor (`v_corridor_block_demand` divides by vehicle count), so capping
a corridor's reach against it understates the pool by up to ~9x. The solver always
reconstructed the corridor total while `_package_metrics` and the validator capped at one
vehicle's share, so `curve_reach_bounded` failed on **every** package containing mobile
inventory — 132,724 against 14,682 on one brief. The pool figure is now published once on
`ScreenEconomics` and all three read it. Three independent implementations of one definition
is the goal; three implementations of three definitions is the bug.

The same class of defect hit fixed inventory through `TERMINUS_WEIGHT`. It was 1.5, applied
per `location_id`, while `pool_key` merges location rows into a site — and a route that
terminates on one side of a road runs mid-route on the other, so one physical stop carried
two crowd figures 1.5x apart on **196 (site x block) cells**. It is now **1.0**: the
disagreement drops to zero, verification passes, and an ASSUMED constant stops reaching a
validated client-facing number (reported reach moved 16,637 -> 15,940 on an otherwise
identical package; that 4.2% was the assumption). If the terminus effect is wanted back,
weight the SITE, not the location.

**A `pool_key` is a SITE, not a `location_id`.** One physical station is modelled as
several location rows — opposite platforms, separate entrances — and screens on them see the
same crowd, so a raw `location_id` let reach count that crowd twice. `v_location_site`
groups on `(city_id, name, the set of corridors serving it)`, and all three keys are
load-bearing:

```
910  raw location_id               under-merges: splits one station across platform rows
626  (city_id, name)               OVER-merges by ~31%. Station names are a low-cardinality
                                   template that unrelated real stations coincidentally share
878  + the corridor set            correct on both counts
```

Total pools: **972** = 878 sites + 94 corridors (was 1,004). 22 sites absorbed 54 locations.
Merging makes reported reach slightly *smaller* and more honest. `pool_key` is now a
synthetic `SITE-<city>-<n>` id, so `location_name` travels alongside it — a reason string
should name the site, never the key.

**Reach is not the sum of exposures.** Screens sharing a `pool_key` (one site, or one
corridor) see the same people; on the canonical 250-screen pool the naive sum over-counts by
**~25x** (1,656,829 against 65,801).

```
reach = SUM over (pool_key, block) of
            min(gross viewed exposures bought, that pool's reachable daily audience)
```

Both sides are in viewed units on purpose: capping viewed exposures at the undiscounted
crowd would let a saturated plan claim every passer-by when only ~35% of them look.

Three implementations, one definition. `or_agent_tools._package_metrics` computes it,
`validation._reach_checks` recomputes it independently, and the solver maximizes it directly
(`R <= min(E, P)` is two exact linear constraints — `min()` of two linear functions is
concave). Keep them in step and do **not** import one into the other — the point is that
independent implementations agree.

**A consequence to state rather than bury.** At `LOOP_PASSES_PER_TRIP = 8`, one slot on a
saturated pool delivers 8/6 viewed exposures per person per day, so a 30-day flight cannot
deliver fewer than ~40 exposures per person reached whatever the optimizer picks. Duration
is the brief's, the minimum purchase is one slot, and there is no flighting. The wear-out cap
therefore constrains **stacking** (a multiple of that floor), not total exposure, and the OR
tool discloses the figure rather than implying it was tuned.

**Audience scores are graded, not binary.** `dominant_occupation` has five values across
30 zones and `mixed` is the most common (14 of 30), so the old white-collar flag scored
`mixed` identically to `student`. Three affinity maps grade all five. Two audience terms
also pointed at the wrong column: `young_professionals` averaged `professional_score` with
**`student_score`** (a different audience), and `high_income` used `professional_score`,
which is 40% occupation. Both now have their own column. Every weight set sums to 1.0 over
bounded inputs, so `family_score` no longer needs a renormalization constant.

**`industry_vertical` is a closed vocabulary, and it was not.** It was a bare `str | None`
with no validator, and every value the Master actually wrote to a run missed:
`'AUTOMOTIVE / ELECTRIC VEHICLES'`, `'Consumer Tech'`, `'Fintech'`,
`'Beauty / Skincare'`. Two sub-scores key on that one string — `context_fit` (0.15) and
`historical_performance` (0.10) — so an unmatched value pinned **25% of every relevance
score to a constant 0.5** while the tool reported a normal-looking ranking. Worse, the MILP
weights `conv_fit` (= `contextual_score`) at 0.40 on a `conversion` objective. Now
`INDUSTRY_VERTICALS` holds the 13 real `bookings.industry_vertical` values, a validator
rejects anything else (normalizing case and separators first, so `'Real Estate'` is
accepted as `real_estate`), and a test pins
`set(INDUSTRY_VERTICALS) == set(INDUSTRY_TO_POI_CONTEXT)` so they cannot drift.
`constant_subscores` on the artifact summary now reports **any** sub-score that is identical
pool-wide, because that state is indistinguishable from success from the outside.

**Mobile screens are excluded from the POI judgement, not scored 0.2 on it.**
`v_screen_poi` joins POIs on `anchor_location_id = location_id`, and a vehicle has no
`location_id` — so all 2,615 mobile screens had an empty POI set for an ARCHITECTURAL
reason and took the 0.2 mismatch penalty for it. `poi_applicable` is checked first and they
score a neutral 0.5, recorded in `defaults_applied` rather than silently.

**Ties are broken at sort time, never in the score.** Screens at one site genuinely tie —
same zone, same POIs, same traffic — and the order among them used to be decided by
`screen_id` alone. The chain is `relevance desc, screen_size desc, ambient footfall desc,
screen_id asc`. `screen_id` MUST stay last: it is what makes the artifact reproducible.
Neither tiebreak is in `WEIGHTS`, because neither measures audience fit, and `position` is
deliberately absent — nothing in this data says any mounting position outperforms another.

**`nearby_ambient_footfall` is quarantined**, the same discipline as
`pricing_internal_reach_proxy`. It correlates only weakly with transit ridership (~0.12-0.26)
and can disagree ~20x at one location, so it is carried on `ScreenCandidate`, used ONLY as a
tiebreak, and added into no impressions, reach or price figure. Blending it in would undo
the stop-share correction.

Known limitations, all surfaced by `describe_relevance_model` and pinned by tests:
- Volume is **schedule-derived only**, with no ambient/pedestrian term. No scheduled service
  starts between 00:00 and 04:00, so block 1's **measured** volume is zero for every
  screen — while 8,544 of 191,110 bookings (4.5%) sit in block 1, so the inventory
  demonstrably sells. That is a **modelling gap, not a finding about block 1**; zero means
  "not modelled", never "nobody there". `impressions_block_1_estimated` publishes an
  8%-of-block-6 **assumption** per day type, kept out of `total_impressions`,
  `peak_impressions`, `offpeak_impressions` and `commuter_score` — nothing the validator
  checks may move with an assumed constant, and `commuter_score`'s denominator is the total.
- Fixed inventory carries ~**2.7x** the median daily volume of mobile (12,775 vs 4,789) and
  `metro_station` ~**59x** `bus_stop` (14,873 vs 250). Before stop shares the fixed:mobile
  gap was 40.9x. Volume-per-dollar still leans fixed. The mobile figure is one vehicle's
  share of its corridor — a stated judgement, since `route_schedules` has no `vehicle_id`.
- **The corridor and stop paths use different sources.** A corridor's pool population comes
  from `route_schedules.estimated_ridership` (scheduled); a stop's audience from
  `ridership_actuals` (observed). `corridor_pool_sanity()` asserts the invariant that a
  corridor's pool population is never smaller than the largest single-station audience drawn
  from its own routes — 0 violations. Restricting to *its own routes* is the whole subtlety:
  compare a stop's TOTAL against one corridor and 40 busy interchanges look like violations.
- **A mobile screen's audience makes a round trip across two modules.**
  `v_corridor_block_demand` in `app/data/db.py` DIVIDES a corridor's riders by `n_vehicles`
  to get one vehicle's share, and `app/optimize/contract.py` MULTIPLIES back by
  `pool_partition_count` (the same count, published on `v_screen_profile`) to recover the
  corridor's whole crowd for the reach ceiling. The two halves must keep using the same
  count or pool populations silently disagree with the per-screen figures — which is why
  `v_corridor_vehicle_count` is defined once and consumed by both, with a warning at the
  definition site. Publishing the corridor total directly on `ScreenCandidate` would remove
  the round trip, but it changes the OR contract, so it is a decision rather than a cleanup.
- **Mobile screens have no demographics.** Zone is undefined for a vehicle, so all 2,615 of
  them score 0 on every demographic component and only `commuter_score` carries signal.
  Averaging the zones a corridor touches is the obvious next improvement, and is
  deliberately kept out of the occupation map rather than smuggled in as a side effect.
- `historical_performance` is a booking **completion rate**, not campaign performance.
- `commuter_score` spans a narrow range, so a commuter-only brief's ordering is effectively
  decided by the other four components.
- **The pool is a truncation, and truncation can be categorical.** See the next section.
- No held-out accuracy metric exists for the audience model, which is why no stage emits a
  per-screen confidence and the validator still skips that check.

#### A mixed-inventory brief needs `screen_type_mix`, not `allowed_screen_types`

`hard_constraints["allowed_screen_types"]` is a **filter, not a mix**, and that made a
mixed brief unservable. Measured: permitting `["metro_station", "bus"]` let 806 eligible
screens through and the single global relevance cut returned **250 metro_station and 0
bus**. Zero bus screens reached pricing, let alone the optimizer.

Bus screens are not bad — exclusion is **categorical rather than marginal**. `bus_stop`
averages **0.5891** against `metro_station`'s **0.6066**, a 2.9% gap, and still landed 0 of
250. A 3% scoring difference produced a 100%/0% outcome.

`CampaignSpec.screen_type_mix` is a real typed field, deliberately not a `hard_constraints`
key: that vocabulary is closed and an unrecognized key is persisted, echoed back and
silently ignored, which is exactly how a slot-depth constraint was lost end to end.
`_stratified_head` allocates `top_n` per named type, capped at availability, redistributing
any shortfall — so a scarce type gets its whole supply (75 of 75 `bus_stop`) and the pool
never shrinks. `pool_composition_by_screen_type`, `eligible_by_screen_type` and
`screen_type_mix_unfilled` are reported on **every** run, mix or not: a pool that is 100%
one screen type is the most consequential fact about it, and the Master cannot state a
composition it was never told.

This is not a new decision column — `head(top_n)` was *already* a decision about which
screens are handed on, and only how that truncation is made has changed. No screen is
picked for the optimizer.

**Intake and the solver are now both wired.** `create_campaign_spec` takes
`screen_type_mix`, and `or_agent_tools._solve` builds one **elastic** coverage group per
requested type (`screen_type:<t>`, min 1 cell). `COVERAGE_PENALTY` is a tenth of the total
reachable population per unit of shortfall — far above what one screen's marginal reach can
earn — so the mix is honoured unless a HARD constraint leaves no room, and then the plan
yields and reports instead of failing.

**Elastic, not hard, and that is the decision.** A mix is a media judgement, so the package
always ships. What is not optional is saying so: `validation.screen_type_mix_disclosed`
validates the **disclosure**, not the constraint — a requested type missing from the package
passes only if `unmet_coverage` names it, and fails otherwise. That inversion is the point.
The original failure was not that bus screens were absent; it was that nothing said so while
every layer reported success.

**The two costs are different quantities and conflating them overstates the smaller one.**
- The **coverage rule** costs ~0. Measured on several briefs: dropping the coverage rows from
  the same candidate pool returns the same reach, because reach saturates per `pool_key` and a
  cheap bus_stop is a *fresh* pool — good value, not a concession. The optimizer buys them
  anyway; on one 60k brief it chose 13 bus_stop against 6 metro_station unprompted.
- The **pool stratification** upstream is where the real cost sits: a stratified 120-screen
  pool reached 8,190 where a metro-only pool of the same size reached 15,940. That belongs to
  candidate selection. `reach_cost_of_the_coverage_rule` reports only the first, and the OR
  prompt forbids attributing the second to it.

**Why the pool cut excludes a type at all, measured — it is not price and not impressions.**
Relevance carries no price term and no impressions term (`transit_score` is reported at
weight 0.0). Permitting `metro_station` + `bus_stop` and taking the top 120 of 6,304 returned
**120 metro_station, 0 bus_stop** — not because bus stops score badly, but because there are
**4,224 metro_station eligible against 735 bus_stop** and the score ranges overlap heavily
(bus_stop 0.4432-0.7692, metro_station 0.4673-0.7831; means 0.5618 vs 0.6114). The scarcer
type loses every slot to a truncation artifact. Do not describe this as a quality judgement.

Still deliberately NOT done: **no per-type floor when no mix is requested.** That would change
what a candidate pool means on every brief ever run.

Why it is worth building rather than just disclosing: reach saturates per `pool_key`. Once a
geography's metro sites are bought out, the marginal metro screen is worth ~0 and a bus
screen is a *fresh* pool — the only remaining way to buy incremental reach. The system
otherwise reports "audience saturation is binding" and leaves budget unspent while a real
diversification option was filtered out at stage 2.

Also a **process singleton** — `get_relevance_engine()`, ~15s to build (the
`ridership_actuals` scan dominates). Degrades to `route_schedules.estimated_ridership` when
that file is not provisioned, and reports which source is live in `demand_source`.

### Working on the optimizer

`app/optimize/` holds the model; `tools/or_agent_tools.py` is a thin wrapper that maps run
state in and contracts out. Same shape as `app/ml/` + `tools/ml_agent_tools.py`.
`tools/relevance_tools.py` is the deliberate exception — engine and tools in one file.

Rules that survive any further change to the formulation:

1. **`_package_metrics` is the reach definition, not part of the model.** The validator
   re-derives it independently. Change the formulation freely; leave the accounting alone.
2. Keep `optimize_package`'s name — the Master's prompt and the validation layer both
   depend on it. Adding a keyword argument is fine; `slots_per_day_cap` went from a
   fabricating default of 3 to `None` deliberately (rule 5), and that is the only kind of
   signature change worth making: one that removes an invented constraint.
3. Anything that turns an audience figure into an exposure figure goes through
   `app/optimize/exposure.py`. One implementation, one call site.
4. A quantity the validator enforces may not depend on a fitted or assumed constant. That is
   why `REACH_LAMBDA` reaches only `curve_reach_diagnostic`, and why `curve_reach_bounded`
   asserts a lambda-free bound instead of re-deriving the curve.
5. **A brief-declared constraint is read off the run, never off a tool argument**, and it
   gets a validator check. `optimize_package(run_id, slots_per_day_cap=None)` resolves the
   slot ceiling from `spec.hard_constraints`; the argument is an exploration override that
   may tighten a declared cap and never widen it. See the slot-structure note below.
6. `backend/tests/` must still pass. `test_optimizer.py` pins the constraint semantics, the
   wear-out arithmetic and a MILP-beats-greedy comparison; `test_pipeline_smoke.py` runs
   every stage with no LLM and asserts the validator accepts the result;
   `test_relevance_engine.py` pins the audience units and the pooling behaviour.

#### The brief's slot structure — and why it binds per screen, not per cell

`hard_constraints["max_slots_per_day"]` is how a brief states the leasing structure ("1
rotating slot on digital screens only"). It had **no channel at all**: `CampaignSpec` has no
slot field, every consumer reads `hard_constraints` against its own hardcoded key list, and
the only real control was `optimize_package`'s `slots_per_day_cap=3` default — mentioned in
neither agent prompt, so no model ever set it. A brief asking for one slot shipped three,
and `verify_package` passed it clean while the constraint sat visible in
`normalized_spec.hard_constraints` and `campaign_inputs` the whole time.

**Semantics: per SCREEN per day, summed across time blocks.** The per-cell reading — which
is all an `available` clip can express — lets a screen bought in Block 3 and Block 5 take
1 + 1 = 2 slots that day and pass. Measured on a 45-day brief, a per-cell cap of 1 returned a
plan whose busiest screen carried 2. The constraint is a hard per-screen row in the MILP
(`solver.py`), and the applied cap, its `source` and the semantics string are reported in
the payload so the figure says which reading produced it.

**Honouring it costs no reach** — measured across four briefs, four cap levels and all
four objectives, not assumed. Reach is identical at every cap (45,328 / 44,717 / 173,603 /
169,439 / 95,355), because reach is capped at each pool's *daily* audience while exposures
accumulate over a 30-day-plus flight, so one slot already over-saturates the pool and extra
depth buys frequency rather than people. Lead with this: it holds on every brief measured.

**Whether it also cuts repetition is brief-dependent.** On a 2-zone 30-day reach brief at a
150k budget, exposures per person fall 111.8 → **40.0, the flight's unavoidable floor**, at
34,334 spend against 92,417 for the same 173,603 people. On a 2-zone 45-day brief at 50k,
caps of 1, 3 and 6 return identical plans on every objective — the solver was already buying
one slot per screen, so the cap never bound. And on `awareness` / `frequency` / `conversion`
the cap does not bound frequency at all: those profiles reward exposures, so the solver
stacks extra *screens* into the same pool once it cannot stack slots (169.6 exposures per
person at every cap). **A slot cap is not a frequency cap** — the wear-out cap is the
frequency instrument, and it is elastic.

**`MAX_CELLS_PER_POOL = 4` must not be scaled by the cap.** The obvious fix — hold pool slot
depth constant, so 4 cells become 12 at a 1-slot cap — was implemented and measured: reach
identical, spend 34,334 → 115,628, exposures per person 40.0 → 112.2. The extra cells let the
solver stack *screens* into an already-saturated pool, the same mechanism that makes the cap
inert on exposure-weighted objectives, so the cap would have relocated repetition rather than
reducing it.

**An unrecognized `hard_constraints` key now FAILS verification.** That is the
generalization: `ENFORCED_HARD_CONSTRAINTS` in `app/models/campaign.py` is the closed
vocabulary of keys some stage enforces, and a spec carrying anything else cannot be blessed.
A false fail costs one turn; a silent pass ships a package that breaches a written brief.
Adding a key there without its enforcing code re-creates the original bug.

**"On digital screens only" is a no-op here.** `datasets/screens.csv` has no digital/static
attribute at all — verified; its descriptive columns are `screen_type`, `position`,
`screen_size`. The 6-slot rotating loop implies digital anyway, so the constraint cannot be
filtered and every screen already satisfies it. `describe_inventory` says so in
`no_digital_flag`. Do not invent a filter.

## Agent wiring

`create_deep_agent(model, tools=[*master_tools.TOOLS, *relevance_tools.TOOLS],
system_prompt=MASTER_SYSTEM_PROMPT, subagents=[ml_agent, or_agent])`.

**Two specialists, not three.** The Master owns stages 1, 2, 5 and 6 (intake, relevance,
verification, recommendation) through its own tools, and delegates 3 and 4 via the built-in
`task` tool. There is no Data Intelligence subagent: stage 2 is a fixed calculation, and an
LLM shell around it bought only latency and a chance to paraphrase numbers wrongly. This is
a deliberate departure from SOLUTION.md 31.1's "exactly three specialists" in favour of
31.2's "LLMs reason; tools calculate" — delegation is reserved for the stages where a
specialist genuinely reasons about its own output.

### Model providers — two, and the rep picks

`app/agents/providers.py` is the registry: the catalogue, the client construction, and the
rate limiters. `master.py` only consumes a resolved `ModelSelection`. Nothing else builds a
chat client.

| provider id | client | models | cap |
|---|---|---|---|
| `gemini` | `ChatGoogleGenerativeAI` | `gemini-3.5-flash-lite` | `MODEL_REQUESTS_PER_MINUTE`, 12/min |
| `azure_openai` | `AzureChatOpenAI` | `gpt-5.4-nano`, `gpt-5.4-mini` | `AZURE_REQUESTS_PER_MINUTE`, 60/min |

Keys live in the **repo-root `.env`** (gitignored): `GEMINI_API_KEY`, and
`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_VERSION`.
`backend/.env` is read second and overrides it.

**The choice rides on the request, not on server config.** `CampaignQuery.provider` /
`.model`, set from the sidebar's Settings dialog and remembered in `localStorage`. It is a
per-turn choice because the two providers fail differently: Gemini's free tier allows ~20
requests/day/model and one orchestration costs 15-20 of them, so a demo runs out of Gemini
before it runs out of questions, and editing `.env` + restarting uvicorn mid-demo is the
thing this avoids. `MODEL_PROVIDER` is only the default for a request naming neither, and
`default_provider()` falls forward to a configured provider rather than 503-ing every
endpoint because a stale env var points at a key nobody set.

`GET /models` serves the catalogue the dialog renders. The frontend must not hold its own
copy: whether a provider has credentials is a backend fact, and a stale list offers the rep
an option that 503s. Unconfigured providers are still listed, with `unconfigured_reason`
naming the missing env var — more actionable than an option that quietly vanishes.

`api/campaign.py` caches one compiled graph **per selection** (`_agents`), because
compiling is not free and the provider changes between turns. All selections share one
`InMemorySaver`, so switching model mid-session keeps the conversation — LangChain
normalizes message history, so a Gemini turn replays fine into an OpenAI one.

Provider notes:
- `deepagents` also accepts the string form `"google_genai:gemini-3.5-flash-lite"`, which it
  resolves through `init_chat_model`. We build clients explicitly so credentials come from
  settings rather than from provider-specific env vars, and so the provider is selectable.
- **Azure: the deployment name is not the model name.**
  `Team7-GPT-5.4-nano-39a7f0abb4d54f9c265d` is what the API wants; `gpt-5.4-nano` is what it
  404s on. The mapping is `AZURE_OPENAI_DEPLOYMENTS` in `config.py`, applied in exactly one
  place, because the failure is an opaque 404 rather than an error naming the cause. The
  **model name** is the selectable id everywhere else — the deployment is a per-tenant
  opaque string that would be meaningless in the UI.
- **Azure: `max_completion_tokens`, not `max_tokens`,** on the GPT-5 family.
  `AzureChatOpenAI` renames the field on the wire, so passing `max_tokens=` is correct —
  but do not go back to a raw `openai` client thinking they are interchangeable. Reasoning
  tokens bill against that cap, hence `MAX_OUTPUT_TOKENS` is per provider (8,192 Gemini /
  16,384 Azure): a cap sized for Gemini's visible output starves the answer, and the failure
  is a truncated tool call rather than an error.
- **Azure: leave `temperature` unset.** These deployments accept only the default, and
  langchain's historical 0.7 is a 400.
- Do **not** set `thinking_level` on the Gemini client version — it stalls the request.
  `gemini-3.5-flash-lite` already reasons by default (`output_token_details.reasoning` is
  non-zero).
- Gemini responses come back as **content blocks**, not a bare string; OpenAI's come back as
  a string. Anything reading assistant text must handle both (see `_message_text` in
  `api/campaign.py`).
- **Rate limiting is mandatory, not optional.** Free-tier Gemini enforces a
  **per-minute** request cap (`gemini-3.5-flash-lite`: 15/min) on top of the daily one. A
  single orchestration blows straight through it, so `providers._rate_limiter` installs one
  **shared** `InMemoryRateLimiter` across the Master and both specialists. It must stay
  shared *within* a provider — the cap is per project+model, so separate limiters would
  each assume the full budget — and must **not** be shared *across* providers, because the
  two caps differ 5x. Dropping the Data subagent removed a few calls per run: stage 2 now
  costs zero model calls.
- Provider failures map to honest HTTP statuses via the provider-agnostic
  `langchain_core.exceptions` bases, so the mapping survives a provider swap: 429 quota, 503
  upstream 5xx/overload, 502 bad credentials or unknown model id. The selection is threaded
  into `_provider_error` so the credential message names the right env var. A model id this
  build does not offer is a **400** rather than a 503 — a stale `localStorage` value is the
  caller's problem, and collapsing it into 503 made it look like a broken backend.
- Iterate against `tests/` (no model calls) rather than burning quota on live runs.
  `tests/test_providers.py` pins the catalogue, the resolution rules and the deployment
  mapping without touching the network.

**The Master owns `set_pricing_levers` and `get_client_negotiation_profile`.** Both serve
the conversation with the sales rep rather than a pipeline stage, so both are deliberately
absent from `TOOL_STAGES` in `frontend/src/lib/stages.ts` — an unmapped tool leaves the
stage rail where it is, whereas mapping them to `economics` would advance the rail past
stages that have not run. Neither creates a run, so neither belongs in
`PIPELINE_ENTRY_TOOL` either; a test pins that for the client profile.

**The OR agent has two tools**, `optimize_package` and `compare_objectives`. The second
exists because the reach/awareness/frequency trade-off is the most commercially useful thing
this system produces and it belongs to the human, not to a silent default. It withholds any
profile whose exposures per person exceed the wear-out cap, reporting the measured figures
and why — but it never withholds or substitutes the campaign's own stated objective, which
`optimize_package` serves and discloses. The bundle's other four tools
(`budget_sensitivity`, `diagnose`, `scenario_reoptimize`, `explain_last_plan`) are not wired
in: each costs a solve plus model calls against a per-minute rate limit.

**Stages are strictly sequential.** Each consumes the previous stage's artifact. The
supervisor can still try to delegate two stages in one turn, so the dependent tools check
their input via `run_state.missing_prerequisite()` and return a recoverable
`prerequisite_missing` result naming the producing stage — they never crash. The Master's
prompt also forbids more than one `task` call per turn.

**Not every turn runs the pipeline.** A question about an existing package is answered
from it — the pipeline only re-runs when a campaign *input* changes. The Master calls
`get_active_run(session_id)` first on any non-opening turn; it returns the session's
latest run plus a `campaign_inputs` dict of exactly what the optimizer consumed, and the
prompt's triage rule compares that against the new message (ANSWER vs REBUILD). The API
reports `pipeline_ran` (a run id that changed across the turn), which is how the UI knows
to skip the stage rail and not repeat the metrics deck. Read-only tools
(`inspect_package`, `describe_relevance_model`, …) must never create a run — if you add
one that does, add it to `PIPELINE_ENTRY_TOOL` in `frontend/src/lib/stages.ts`.

**Logging.** `app/logging_utils.py` prints `[INFO]` milestones, `[DEBUG]` per-call detail
and `[ERROR]` failures (set `LOG_DEBUG=0` to quiet the debug lines).
`app/agents/tracing.py` holds `AgentRunLogger`, a callback handler registered once in the
run config — callbacks propagate, so one instance traces the Master *and* every subagent,
and totals tokens. Every run ends with a `TOKENS`/`CALLS` summary line, and the same
figures come back in the API response as `token_usage`.

**Run handles, not payloads.** Every tool takes a `run_id`. The spec, artifact references,
optimization result and validation live in `run_state` (localDB) and the artifact store,
so candidate lists and price tables never enter an LLM's context.

## Reading uploaded briefs

`app/services/documents.py` is the only place uploaded bytes are interpreted. Three
formats, because those are the three a rep forwards: **`.pdf`, `.docx`, `.txt`**. Both
parsers are pure Python (`pypdf`, `python-docx`) so the upload path needs no build
toolchain.

Uploads were accepted for months and never read. The prompt told the agent to use "the
filesystem tools", but deepagents' `read_file` addresses a **virtual state filesystem, not
the disk** — so a staged file was never reachable, and raw PDF bytes would not have helped
if it had been. A rep could attach an RFP carrying the budget, the dates and the market
list, and the package came back built from the one-line chat message.

How it works now:

1. **Parsed once, at upload.** `documents.extract_and_store` writes the text to a
   `<file>.extracted.txt` sidecar beside the staged file and records only a summary
   (status, char/page count, truncation, a 200-char preview) in localDB. Not per agent
   turn: re-parsing a 50-page PDF on every tool call is wasted work against a per-minute
   rate limit, and a rep needs to know *at upload* that their scan contributed nothing.
2. **`UploadOut` carries the extraction result**, and the attachment chip shows it. A
   scanned PDF is otherwise indistinguishable from a readable one until the recommendation
   comes back missing the constraints the rep thought they had supplied.
3. **The agent reads through one tool**, `master_tools.read_campaign_document(upload_id)`,
   which returns a **bounded excerpt** — `AGENT_EXCERPT_CHARS`, ~8 pages — never the file.
   `_build_prompt` lists each document with its `upload_id` and marks the unreadable ones
   so the agent neither ignores an attachment silently nor spends a rate-limited call
   discovering it is empty.

Decisions worth not re-litigating:

- **Only the three parseable formats are accepted.** `.md`, `.csv`, `.xlsx` and `.pptx`
  used to upload cleanly and reach no parser — the file attached, the agent saw nothing,
  and nobody was told. Rejecting them with a 415 is the honest version. Widening the list
  means writing the parser first.
- **Two different caps.** `MAX_EXTRACT_CHARS` (200k) is what gets stored; the cap is
  applied *while* accumulating, because a 20 MB `.docx` is a zip that decompresses to far
  more and building the whole string first turns an upload limit into a memory problem.
  `AGENT_EXCERPT_CHARS` (20k) is what reaches a model, per SOLUTION.md 31's rule that a
  whole document never enters agent context.
- **`.docx` tables are read, not just paragraphs.** Briefs put the budget, the flight dates
  and the market list in a table about as often as in a sentence; a paragraph-only reader
  drops exactly the fields intake needs.
- **A scanned PDF and a locked PDF get different messages.** Both yield no text, but the
  rep's next action is opposite — retype the numbers, versus resend it unlocked. An
  "encrypted" PDF is retried with an empty user password first, since most in circulation
  carry only an owner password and open fine in any viewer.
- **Nothing raises into the request.** Every failure mode returns a status, and an
  unparseable upload still succeeds as an upload — the rep can still describe the brief in
  the chat, and losing their file because we could not read it is the worse outcome.
- **Documents are data, never instructions.** Both the tool docstring and the prompt say
  so, because the content is untrusted third-party input. A directive found inside a
  client deck is text to summarise, not a request to follow.
- Extraction status is recorded, so a known-unreadable document short-circuits in the tool
  rather than being re-parsed. `documents.load_text` falls back to re-parsing the original
  when the sidecar is missing, which is every document staged before this existed.

## Persistence and the read path

Everything a session accumulates is durable. Three fixes landed together here, and each
had a symptom that looked like the other two.

**Transcripts are stored server-side.** `services/transcripts.py` + `api/messages.py`
(`GET/POST/DELETE /sessions/{id}/messages`, `GET/PATCH/DELETE /messages/{id}`). The
campaign endpoints write both halves of every turn themselves — the **user message before
the agent is invoked**, so a turn that dies on quota still records what was asked, and the
assistant message on completion. The browser therefore never has to POST its own messages
for them to survive a reload; the CRUD routes are a read/amend surface.

An assistant message carries the turn's metadata as well as its prose: `run_id`,
`pipeline_ran`, `tool_trail`, `token_usage`. `pipeline_ran` is what keeps a restored
transcript honest — a follow-up answered off an existing package must not re-render the
metrics deck, exactly as in the live stream.

Sessions predating this have no messages, so the UI falls back to rebuilding the brief from
`campaign_spec.original_query` and labels that turn `restored`. That is the **only** path
that shows a package with no answer, and it says so on screen.

**Session names are the campaign objective, capped at five words.**
`services/session_titles.py`. Naming happens in two steps because the two useful names
arrive at different times, and the sidebar used to show the wrong one of them:

1. `_ensure_session` names a session provisionally from the rep's raw sentence, so the rail
   is not a placeholder for the 45-90s a run takes.
2. `_final_title` then **replaces** it with the run's resolved `campaign_objective` — which
   is what the centre-panel header shows. Showing the brief in the rail and the objective in
   the header is the same campaign under two names, and the rail read like a chat log.

`title_source` on the session record is what makes step 2 safe: only `PATCH /sessions/{id}`
sets it to `"user"`, and a user-typed title is never replaced. Records predating the field
are treated as auto-named, which is the conservative reading — the only ways they could
have got a title were automatic.

The five-word cap (`MAX_WORDS`, with `MAX_CHARS` as the guard for five long words or one
unbroken string) is applied in `title_from_text`, and `backfill_from_runs` **re-shortens
titles written before the cap existed** — capping the function alone would leave every row
already on disk breaking the rule. It runs on every `GET /sessions` and is idempotent.

The stream emits a **`session` event before the first model call**, carrying the
provisional name. Without it the client learned the title only on `done`, so the rail spent
the whole run showing a placeholder; the frontend used to paper over that by displaying the
raw user message, which is the behaviour that made this look broken. There is deliberately
no client-side copy of the naming heuristic.

**Deleting a session cascades** to its messages, runs and uploads (`local_db.delete_where`,
one atomic write per collection). Orphaning them left `latest_run_for_session` resolving
runs for a session the user believed was gone. Files on disk — staged uploads and artifact
parquet — are deliberately left alone; they are unreachable once the records go, and
deleting bytes is heavier and irreversible. Clearing a transcript
(`DELETE /sessions/{id}/messages`) is a *different* action and leaves the runs intact.

**`artifact_id` is the portable handle; `ArtifactReference.path` is not.**
`artifact_store.resolve_path` looks in the configured `artifacts_dir` first and treats the
recorded path as a fallback. The reverse order was a live bug: `runs.json` was committed
while `backend/artifacts/` is gitignored, so a run cloned from another checkout arrived
carrying `/Users/someone/projects/.../screen_candidates-abc123.parquet` and was unreadable
even after the file was regenerated locally under the same id. An artifact that is
genuinely absent still raises, and the endpoint returns **410** naming the id — a missing
*reference* means the stage never ran, which is a different thing.

The UI must not swallow that 410. `RunData.artifactErrors` carries it and the inspector
renders it, because the old `.catch(() => null)` turned it into an empty row array while
the reference still reported 250 rows — a tab claiming data it was not showing.

**Call the API on `127.0.0.1`, never `localhost`.** uvicorn binds IPv4 only, and on Windows
a SYN to `[::1]:8000` is *dropped* rather than refused, so a client resolving `localhost`
waits out the full connect timeout before falling back. Measured on the same four-request
session-open path, host as the only variable: **8.18s via `localhost` against 0.053s via
`127.0.0.1`.** This was the whole of the "the UI takes 5-6 seconds" complaint. `cors_origins`
allows both loopback spellings for the *page* origin.

Two request-pattern fixes went with it: `GET /runs` uses `run_state.snapshot_of` on records
already in hand (it was N+1 on file reads — 21 reads for 20 runs, now 1), and the
session-open path no longer fetches the same run record twice.

**Run data is cached per session in the UI.** `runDataBySession`, not a single slot. It used
to be one slot that `selectSession` cleared while `hydratedRef` suppressed the refetch, so
the first visit to a session filled the inspector and **every visit after it showed empty
D1-D4 tabs**. The cache is what makes the "already hydrated" guard correct. Hydration
un-marks itself on failure, so a session that failed to load retries on the next visit.

## Agent architecture rules (from SOLUTION.md §31 — non-negotiable)

- One Master Deep Agent + **two** specialists: ML and OR. Do not add agents. §31.1 asked
  for three; the Data specialist was removed because stage 2 is a deterministic engine and
  the LLM shell around it added nothing. Prefer a Master-owned tool over a subagent
  whenever a stage does not actually reason.
- LLMs reason and orchestrate; deterministic tools calculate. No LLM arithmetic,
  no LLM optimization.
- All inter-agent contracts are Pydantic models. Pass artifact references +
  schema + summary, never DataFrames, through agent context.
- Hard constraints (budget, availability, dates, geography) are enforced in code.
  An LLM may never reason a violation away.
- Infeasibility is explicit: return status + reason codes + relaxation options.
  Never fabricate a plausible-looking package.
- Master Agent validates every specialist output before answering.
- Build the deterministic pipeline first; wrap it in agents after it works.
- **Reach is never the sum of exposures.** Dedupe on `pool_key` first. Any new consumer of
  audience numbers has to respect this or it will overstate the audience ~25x. Dedupe on
  the SITE, not on `location_id` — one station is several location rows.
- **A validated number may not depend on an assumed constant.** If a figure the validator
  checks moves with a constant nobody can measure, the check validates nothing about the one
  thing that is actually uncertain.
- **A constraint the brief declares gets a validator check, or it is not enforced.** A
  stated input qualifies (unlike `REACH_LAMBDA`), and without an independent re-derivation
  nothing surfaces a silent miss — which is exactly how a slot-depth constraint was dropped
  end to end while every layer reported success. `hard_constraints` keys live in the closed
  `ENFORCED_HARD_CONSTRAINTS` vocabulary and an unrecognized one fails verification.

## Build order

Deterministic pipeline before the agent layer:
data/DuckDB + views → features → relevance scoring → demand + pricing models →
MILP optimizer → recommendation → agents → API → UI.

Everything on that line is built, MILP included.

## Conventions

- Backend: `snake_case`, full type hints, Pydantic v2, `uv` or `pip` with
  `pyproject.toml`. Secrets in the repo-root `.env` (`GEMINI_API_KEY`), never committed.
- Frontend: `PascalCase` components, colocated per `UI.md`'s tree, server components
  by default. Talks to FastAPI only through `src/lib/api`.
- Frontend types mirror backend Pydantic models — change both together.
- Every recommendation carries traceable reasons referencing real feature values or
  model outputs. No generic "this screen is highly relevant".

## Commands

```bash
cd backend
uv venv --python 3.13 .venv && uv pip install -e ".[dev]"   # first time
# GEMINI_API_KEY / AZURE_OPENAI_* come from the repo-root .env; backend/.env only overrides

.venv/bin/uvicorn app.main:app --reload --port 8000
.venv/bin/python -m pytest tests/ -q     # no API key needed
.venv/bin/ruff check app tests && .venv/bin/ruff format app tests

# frontend
cd frontend && npm run dev        # :3000
npx tsc --noEmit && npx next lint
```

The repo sits in OneDrive, whose synced paths make Next's `readlink` calls on `.next`
fail intermittently with `EINVAL` — so `predev`/`prebuild` delete it via
`frontend/scripts/clean-next-cache.mjs`; a `.next` that never persists is never swept.
Read that script's header before trying anything cleverer: the `ReparsePoint` attribute is
not the signal (all of `node_modules` has it), `attrib +P` does not help, and relocating
`distDir` breaks typechecking. Only moving the repo out of OneDrive actually fixes it.

`GET /health` reports table count, whether `ridership_actuals.csv` was provisioned, and
which model providers have credentials (`model_providers_configured`, per provider —
either one alone is enough to run). `GET /models` is the picker's catalogue. Hit it on `127.0.0.1`, not `localhost` — see
"Persistence and the read path" for the ~200ms-per-connection reason. The agent endpoints return 503 when the selected
provider has no key; the test suite does not need one.

Smoke-test the agent end to end:

```bash
curl -s -X POST 127.0.0.1:8000/campaign/run -H 'content-type: application/json' -d '{
  "provider": "azure_openai", "model": "gpt-5.4-mini",
  "query": "I have $50,000 for a 30-day campaign starting 2026-10-01 targeting commuters
            aged 18-34 in the Downtown Core zone of Las Hackland. Optimize for reach."
}' | jq '{run_id, provenance, stub_stages, answer}'
```

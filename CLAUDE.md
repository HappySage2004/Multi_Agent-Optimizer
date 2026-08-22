# CLAUDE.md

Transit Media Campaign Recommendation System — hackathon build. Natural-language
campaign brief in, explainable sales-ready media package out.

Full specs live in the repo root and are the source of truth:
- [SOLUTION.md](SOLUTION.md) — architecture, 6-stage pipeline, agent design, Pydantic contracts
- [DATASETS.md](DATASETS.md) — the 14 source CSVs, join map, data-quality traps
- [UI.md](UI.md) — UI spec (design system, 3-panel layout, component tree)
- [UI-handoff.html](UI-handoff.html) — static HTML mockup of the target UI

Read the relevant spec before implementing a stage. Do not re-derive them here.

## Repository structure

```
backend/          Python 3.11+ / FastAPI / LangGraph Deep Agents
  app/
    main.py               FastAPI entrypoint (/health, sessions, uploads, campaign)
    config.py             Settings; all paths resolve from the repo root
    api/                  schemas.py, sessions.py, uploads.py, campaign.py
    agents/
      master.py           build_master_agent() -- create_deep_agent + 2 subagents
      subagents.py        aggregator: assembles the two specs in pipeline order
      ml_agent.py         ML/Pricing specialist: name + description + prompt + build()
      or_agent.py         OR specialist          (one file per agent)
      prompts.py          MASTER_SYSTEM_PROMPT only
      validation.py       deterministic package verification (master-owned)
    tools/
      master_tools.py     intake, geography, verify, inspect
      relevance_tools.py  AUDIENCE RELEVANCE ENGINE + its tools, one file.
                          Master-owned, no subagent. Stage 2.
      ml_agent_tools.py   ML Agent surface -- thin wrapper over app/ml/
      or_agent_tools.py   OR Agent surface -- thin wrapper over app/optimize/,
                          plus the reach accounting the validator re-derives
    models/               Pydantic contracts -- the stable inter-agent interfaces
    data/                 db.py (DuckDB + views), reference.py (validation lookups)
    services/             artifact_store.py, local_db.py, run_state.py, session_titles.py
    ml/                 pricing engine (see below). occupancy, price_band,
                        booking_probability, seasonality, impressions, price_optimizer,
                        engine (singleton), loaders (DuckDB)
    optimize/           the MILP. config (constants, provenance-tagged), exposure
                        (people passing -> viewed exposures, ONE implementation),
                        contract (input validation), pooled (pool population +
                        the diagnostic curve), solver (the formulation)
    features/           empty. The relevance engine lives in tools/relevance_tools.py
                        by design decision, not here.
  tests/                  conftest.py isolates storage; test_pipeline_smoke.py,
                          test_relevance_engine.py, test_pricing_engine.py,
                          test_optimizer.py
  pyproject.toml, .env.example
frontend/         Next.js (App Router) + TypeScript + Tailwind
  src/app/            routes
  src/components/     layout/ chat/ inspector/  (see UI.md component tree)
  src/lib/            API client, types mirroring backend Pydantic models
datasets/         Read-only source CSVs. Never write here.
localDB/          JSON files = the app database (sessions, saved campaigns, runs)
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
- `localDB/*.json` — application persistence: chat sessions, campaign specs, run
  history, cached recommendations. Plain JSON files, read/written via
  `backend/app/services/`. One file per collection (e.g. `sessions.json`).
  Application data only — no analytical data, no uploaded file bytes.
  Large intermediate artifacts (screen candidates, economics) go to
  `backend/artifacts/` as parquet, referenced by ID — not into localDB or agent context.
- `stage/{session_id}/` — documents the user uploads alongside a query (briefs, RFPs,
  client decks). The upload endpoint writes the file here and records only its
  metadata (path, filename, mime type, size, session) in localDB. Brief intake reads
  the staged file to enrich `CampaignSpec` (SOLUTION.md §3). Treat contents as
  untrusted input: validate type and size, never execute, never pass a whole document
  into agent context — extract and summarize first.

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
| Validation layer (`agents/validation.py`) — 17 checks | real |
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
  size x type x position x city x daypart with a bounded A/B/C fallback ladder.
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

### Audience relevance and volume

`app/tools/relevance_tools.py` is the whole stage-2 capability in one file: the feature
layer, the scoring model and the three tools the Master calls. It is **deterministic** — no
LLM, no subagent. Ported from the audience relevance notebook and verified
output-identical on all 12 impression columns, POI footfall, POI counts and all 1,004 pool
keys across 11,163 screens.

Feature layer in DuckDB (`app/data/db.py`): `v_screen_profile` (demographics + POI +
`pool_key`) and `v_screen_demand_history` (avg daily riders per screen x block x day type),
built on `v_schedule_block`, `v_route_block_demand`, `v_corridor_block_demand`,
`v_screen_poi`.

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

**Reach is not the sum of exposures.** Screens sharing a `pool_key` (one stop, or one
corridor) see the same people; on a 250-screen pool the naive sum over-counts by ~23x.

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

Known limitations, all surfaced by `describe_relevance_model` and pinned by tests:
- Volume is **schedule-derived only**, with no ambient/pedestrian term. Block 1
  (00:00-04:00) is zero for every screen despite 8,544 real block-1 bookings. Zero means
  "not modelled", never "nobody there".
- `metro_station` median daily volume is ~380x `bus`, so impressions-per-dollar favours
  fixed inventory almost exclusively. The mobile figure is one vehicle's share of its
  corridor — a stated judgement, since `route_schedules` has no `vehicle_id`.
- **Mobile screens have no demographics.** Zone is undefined for a vehicle, so all 2,615 of
  them score 0 on the demographic components. Averaging the zones a corridor touches is the
  obvious next improvement.
- `historical_performance` is a booking **completion rate**, not campaign performance.
- `commuter_score` spans only 0.34-0.51, so a commuter brief's ordering is effectively
  decided by the other four components.
- No held-out accuracy metric exists for the audience model, which is why no stage emits a
  per-screen confidence and the validator still skips that check.

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
2. Keep `optimize_package`'s name and argument signature — the Master's prompt and the
   validation layer both depend on them.
3. Anything that turns an audience figure into an exposure figure goes through
   `app/optimize/exposure.py`. One implementation, one call site.
4. A quantity the validator enforces may not depend on a fitted or assumed constant. That is
   why `REACH_LAMBDA` reaches only `curve_reach_diagnostic`, and why `curve_reach_bounded`
   asserts a lambda-free bound instead of re-deriving the curve.
5. `backend/tests/` must still pass. `test_optimizer.py` pins the constraint semantics, the
   wear-out arithmetic and a MILP-beats-greedy comparison; `test_pipeline_smoke.py` runs
   every stage with no LLM and asserts the validator accepts the result;
   `test_relevance_engine.py` pins the audience units and the pooling behaviour.

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

**Model: Google Gemini `gemini-3.5-flash-lite`** via `langchain-google-genai`
(`ChatGoogleGenerativeAI`), same model for both tiers. The key lives in the **repo-root
`.env`** as `GEMINI_API_KEY` (gitignored); `backend/.env` is read second and overrides it.
`GEMINI_MODEL` sets the model, with optional `MASTER_MODEL` / `SPECIALIST_MODEL` overrides.

Provider notes:
- `deepagents` also accepts the string form `"google_genai:gemini-3.5-flash-lite"`. We build the
  client explicitly so the key comes from settings rather than a `GOOGLE_API_KEY` env var.
- Do **not** set `thinking_level` on this client version — it stalls the request.
  `gemini-3.5-flash-lite` already reasons by default (`output_token_details.reasoning` is
  non-zero).
- Responses come back as **content blocks**, not a bare string. Anything reading assistant
  text must handle `list[dict]` and keep only `type == "text"` blocks (see `_final_text` in
  `api/campaign.py`).
- **Rate limiting is mandatory, not optional.** Free-tier Gemini enforces a
  **per-minute** request cap (`gemini-3.5-flash-lite`: 15/min) on top of the daily one. A
  single orchestration blows straight through it, so `agents/master.py` installs one
  **shared** `InMemoryRateLimiter` across the Master and both specialists
  (`MODEL_REQUESTS_PER_MINUTE`, default 12). It must stay shared — the cap is per
  project+model, so separate limiters would each assume the full budget. Dropping the Data
  subagent removed a few calls per run: stage 2 now costs zero model calls.
- Provider failures map to honest HTTP statuses: 429 quota, 503 upstream overload, 502 bad
  credentials or unknown model id, via the provider-agnostic `langchain_core.exceptions`
  bases. `POST /campaign/run` maps provider
  failures honestly: 429 on quota, 503 on upstream 5xx/overload, 502 on bad
  credentials or an unavailable model id. Error mapping uses the provider-agnostic
  `langchain_core.exceptions` bases, so it survives a provider swap.
- Iterate against `tests/` (no model calls) rather than burning quota on live runs.

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
  audience numbers has to respect this or it will overstate the audience ~23x.
- **A validated number may not depend on an assumed constant.** If a figure the validator
  checks moves with a constant nobody can measure, the check validates nothing about the one
  thing that is actually uncertain.

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
# GEMINI_API_KEY comes from the repo-root .env; no backend/.env needed unless overriding

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
whether `GEMINI_API_KEY` is configured. The agent endpoints return 503 without a key; the
test suite does not need one.

Smoke-test the agent end to end:

```bash
curl -s -X POST localhost:8000/campaign/run -H 'content-type: application/json' -d '{
  "query": "I have $50,000 for a 30-day campaign starting 2026-10-01 targeting commuters
            aged 18-34 in the Downtown Core zone of Las Hackland. Optimize for reach."
}' | jq '{run_id, provenance, stub_stages, answer}'
```

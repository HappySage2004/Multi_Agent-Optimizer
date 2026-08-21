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
      master.py           build_master_agent() -- create_deep_agent + 3 subagents
      subagents.py        SubAgent specs for data / ml / or
      prompts.py          system prompts
      validation.py       deterministic package verification (master-owned)
    tools/
      master_tools.py     intake, geography, verify, inspect  (REAL)
      data_agent_tools.py Data Agent surface                  (STUB)
      ml_agent_tools.py   ML Agent surface                     (STUB)
      or_agent_tools.py   OR Agent surface                     (STUB)
    models/               Pydantic contracts -- the stable inter-agent interfaces
    data/                 db.py (DuckDB + views), reference.py (validation lookups)
    services/             artifact_store.py, local_db.py, run_state.py
    features/ ml/ optimize/   empty -- for the specialist owners' real pipelines
  tests/                  conftest.py isolates storage; test_pipeline_smoke.py
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

The **Master Agent is complete**. The **three specialists are stubs** — teammates are
building the real Data, ML and OR pipelines separately and will be integrated in.

| Component | Owner | State |
|---|---|---|
| Master Agent, prompts, orchestration | this repo | real |
| Brief intake + geography resolution | this repo | real |
| Validation layer (`agents/validation.py`) | this repo | real |
| DuckDB reference layer, artifact store, localDB, run state | this repo | real |
| FastAPI + SSE endpoints | this repo | real |
| Data Intelligence Agent | teammate | stub |
| ML / Forecasting Agent | teammate | stub |
| OR / Optimization Agent | teammate | stub |

Stubs return **deterministic, contract-shaped placeholders** built on real screen IDs, so
the pipeline runs end to end and the validation layer exercises real code paths. They are
never presented as analysis: every stub artifact carries `provenance="stub"`,
`run_state.stub_stages()` reports which stages are placeholders, and the Master Agent's
prompt requires it to say so in its answer.

### Integrating a real specialist

1. Replace the `_stub_*` function body in the matching `app/tools/*_agent_tools.py`. Each
   file opens with an INTEGRATION POINT block naming what must stay fixed.
2. Keep the tool names, argument signatures, and the returned contract identical — the
   Master Agent's prompt and the validation layer both depend on them.
3. Flip `provenance="stub"` to `"computed"` on the artifact write. That is the single
   switch that stops the "illustrative only" warning.
4. `backend/tests/test_pipeline_smoke.py` must still pass. It runs every stage with no
   LLM in the loop and asserts the validator accepts the result.

Put real feature/model/solver code in `app/features/`, `app/ml/`, `app/optimize/` and
call it from the tool module — tools stay thin wrappers.

## Agent wiring

`create_deep_agent(model, tools=master_tools.TOOLS, system_prompt=MASTER_SYSTEM_PROMPT,
subagents=[data_agent, ml_agent, or_agent])`. The Master delegates via the built-in `task`
tool and owns stages 1, 5 and 6 (intake, verification, recommendation) itself.

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
  single orchestration makes ~17 model calls in ~20s and blows straight through it, so
  `agents/master.py` installs one **shared** `InMemoryRateLimiter` across the Master and
  all three specialists (`MODEL_REQUESTS_PER_MINUTE`, default 12). It must stay shared —
  the cap is per project+model, so three separate limiters would each assume the full
  budget. A paced run takes ~90s.
- Provider failures map to honest HTTP statuses: 429 quota, 503 upstream overload, 502 bad
  credentials or unknown model id, via the provider-agnostic `langchain_core.exceptions`
  bases. `POST /campaign/run` maps provider
  failures honestly: 429 on quota, 503 on upstream 5xx/overload, 502 on bad
  credentials or an unavailable model id. Error mapping uses the provider-agnostic
  `langchain_core.exceptions` bases, so it survives a provider swap.
- Iterate against `tests/` (no model calls) rather than burning quota on live runs.

**Stages are strictly sequential.** Each consumes the previous stage's artifact. The
supervisor can still try to delegate two stages in one turn, so the dependent tools check
their input via `run_state.missing_prerequisite()` and return a recoverable
`prerequisite_missing` result naming the producing stage — they never crash. The Master's
prompt also forbids more than one `task` call per turn.

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

- One Master Deep Agent + exactly three specialists: Data, ML, OR. Do not add agents.
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

## Build order

Deterministic pipeline before the agent layer:
data/DuckDB + views → features → relevance scoring → demand + pricing models →
MILP optimizer → recommendation → agents → API → UI.

## Conventions

- Backend: `snake_case`, full type hints, Pydantic v2, `uv` or `pip` with
  `pyproject.toml`. Secrets in `backend/.env` (`ANTHROPIC_API_KEY`), never committed.
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

# frontend (not scaffolded yet)
cd frontend && npm run dev        # :3000
```

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

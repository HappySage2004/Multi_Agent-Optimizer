# Transit Media Campaign Intelligence Platform

A campaign brief in plain language goes in — "$50,000, 30 days, commuters aged 18–34 in the
Downtown Core, optimise for reach" — and a priced, optimised, fully verified media package
comes out, with the reasoning a sales rep can say out loud on a call.

The system replaces the spreadsheet-and-instinct workflow it was built against. A Master agent
reads the brief (including any attached PDF/DOCX RFP), then orchestrates three specialists:
an **audience relevance engine** that scores all 11,163 screens against the brief, a
**pricing engine** that quotes each screen × time block off 191,110 historical bookings, and a
**MILP optimizer** that picks the best buyable combination under budget, availability and
inventory constraints. The Master then re-derives every headline number through 19 independent
verification checks before it answers — and ends each answer with costed suggestions the rep
can act on.

Design principle throughout: **LLMs reason and orchestrate; deterministic tools calculate.**
No LLM arithmetic, no LLM optimization, and no number stated that a tool did not return.

- Architecture, pipeline stages and agent contracts → [SOLUTION.md](SOLUTION.md)
- Source datasets, join map and data-quality traps → [DATASETS.md](DATASETS.md)
- UI spec → [UI.md](UI.md)
- Working notes and non-negotiable rules for contributors → [CLAUDE.md](CLAUDE.md)
- C4 system-design document → `SYSTEM-DESIGN-C4.html`

---

## Technical specifications

### Backend

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (developed on 3.13) |
| API | FastAPI + uvicorn, REST and Server-Sent Events |
| Agent framework | LangGraph **Deep Agents** (`deepagents`) — one Master agent, two specialist subagents |
| Model providers | Google Gemini (`langchain-google-genai`) and Azure OpenAI (`langchain-openai`); the rep picks per request |
| Analytical store | **DuckDB** (in-process) — the 14 source CSVs registered as views, plus 11 derived analytical views |
| Optimizer | **HiGHS** MILP via `scipy.optimize.milp` |
| ML / pricing | scikit-learn (calibrated logistic booking-probability model), pandas, LightGBM |
| Contracts | Pydantic v2 at every seam, `pydantic-settings` for config |
| Artifacts | Parquet via PyArrow, referenced by id (large tables never enter agent context) |
| App persistence | `localDB/` — plain JSON files, one per collection |
| Document parsing | `pypdf` + `python-docx` (pure Python, no build toolchain needed) |
| Lint / format | `ruff` (line length 100) |
| Tests | `pytest` — 362 tests, no API key and no network required |

### Frontend

| Concern | Choice |
|---|---|
| Framework | Next.js 15 (App Router), React 19 |
| Language | TypeScript 5.7 (types mirror the backend Pydantic models) |
| Styling | Tailwind CSS 4 |
| Markdown | `react-markdown` + `remark-gfm` |
| Transport | REST + SSE against the FastAPI backend only |

### Three data layers — deliberately never conflated

| Layer | Path | Contents |
|---|---|---|
| Analytical | `datasets/*.csv` | 165 MB, 14 tables, **read-only**. Loaded once into DuckDB and queried through SQL views. |
| Application | `localDB/*.json` | Sessions, chat transcripts, runs, uploads, campaign specs. |
| Artifacts | `backend/artifacts/*.parquet` | Screen candidates and screen economics — intermediate results, by id. |
| Staging | `stage/{session_id}/` | Uploaded campaign documents plus one extracted-text sidecar each. |

---

## Local setup

### Prerequisites

- Python **3.11+** (3.13 recommended) and [`uv`](https://docs.astral.sh/uv/) (or plain `pip`)
- Node.js **20+** and npm
- API credentials for **at least one** model provider — Google Gemini or Azure OpenAI.
  Either one alone is enough to run the whole system.

### 1. Provision the datasets

`datasets/` holds the 14 source CSVs. Thirteen are in the repo;
**`datasets/ridership_actuals.csv` (124 MB, 2.05 M rows) is gitignored and must be copied in
separately.** Without it the system still runs — the audience model degrades to scheduled
ridership estimates and reports which source is live — but reach figures are less accurate.

### 2. Credentials

Create a **repo-root `.env`** (gitignored). Copy `backend/.env.example` for the full annotated
list; the minimum is one provider:

```bash
# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite

# or Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_MODEL=gpt-5.4-mini

# Default provider for a request that names neither: gemini | azure_openai
MODEL_PROVIDER=gemini
```

`backend/.env` is read second and overrides the root file. On Azure note that the **deployment
name is not the model name** — the mapping lives in `AZURE_OPENAI_DEPLOYMENTS` /
`backend/app/config.py`, and sending a model name returns an opaque 404.

### 3. Backend

```bash
cd backend
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"

# Windows
.venv/Scripts/uvicorn app.main:app --reload --port 8000
# macOS / Linux
.venv/bin/uvicorn app.main:app --reload --port 8000
```

First request is slow on purpose: the pricing engine (~9 s) and the audience engine (~15 s)
are process singletons built once, never per request.

Check it came up:

```bash
curl -s 127.0.0.1:8000/health
```

That reports the table count, whether `ridership_actuals.csv` was provisioned, and which model
providers have credentials. `GET /models` returns the catalogue the UI's model picker renders.

> **Use `127.0.0.1`, never `localhost`.** uvicorn binds IPv4 only, and on Windows a connection
> to the IPv6 loopback is dropped rather than refused — measured 8.18 s versus 0.053 s on the
> same request path, host as the only variable.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

The API base defaults to `http://127.0.0.1:8000`; override with `NEXT_PUBLIC_API_BASE_URL`
(see `frontend/.env.example`) only if uvicorn is bound elsewhere.

### 5. Lint and typecheck

```bash
cd backend
.venv/Scripts/ruff check app tests && .venv/Scripts/ruff format app tests

cd ../frontend
npx tsc --noEmit && npx next lint
```

Iterate against the test suite rather than live runs — a single orchestration costs 15–20 model
calls, and Gemini's free tier allows roughly 20 per day per model.

### 6. Smoke-test the agent end to end

```bash
curl -s -X POST 127.0.0.1:8000/campaign/run -H 'content-type: application/json' -d '{
  "provider": "azure_openai", "model": "gpt-5.4-mini",
  "query": "I have $50,000 for a 30-day campaign starting 2026-10-01 targeting commuters
            aged 18-34 in the Downtown Core zone of Las Hackland. Optimize for reach."
}' | jq '{run_id, provenance, stub_stages, answer}'
```

A healthy run returns `provenance: "computed"` and an empty `stub_stages` — no stage in this
system is a stub or a heuristic.

---

## Repository layout

```
backend/            FastAPI + LangGraph Deep Agents
  app/agents/         master, ml_agent, or_agent, prompts, providers, validation, tracing
  app/tools/          master, relevance (the stage-2 engine), ml_agent, or_agent tool surfaces
  app/ml/             pricing engine — bands, occupancy, booking probability, seasonality,
                      demand value, client profile, levers
  app/optimize/       the MILP — config, exposure, contract, pooled reach, solver
  app/data/           DuckDB connection and the analytical view layer
  app/api/            routers; app/models/ Pydantic contracts; app/services/ persistence
  tests/              362 tests, no network required
frontend/           Next.js app — src/components/{chat,inspector,layout,proposal}
datasets/           read-only source CSVs — never write here
localDB/            JSON application database
stage/              uploaded campaign documents, one folder per session
```

Monorepo: all code lives under `backend/` or `frontend/`. Nothing at the root but docs,
data directories and config — and only one `.gitignore`, at the root.

### A note on OneDrive

If the repo sits in a OneDrive-synced folder, Next's `readlink` calls on `.next` fail
intermittently with `EINVAL`. `predev`/`prebuild` delete that cache via
`frontend/scripts/clean-next-cache.mjs`; read that script's header before trying anything
cleverer. Moving the repo out of OneDrive is the only real fix.

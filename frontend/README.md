# AgentIQ Frontend

Next.js (App Router) + TypeScript + Tailwind v4. Implements the 3-panel workspace in
[`../UI.md`](../UI.md), styled to match [`../UI-handoff.html`](../UI-handoff.html).

## Run it

```bash
npm install
cp .env.example .env.local        # NEXT_PUBLIC_API_BASE_URL, defaults to :8000
npm run dev                       # :3000
```

The backend must be running, since every panel reads from it:

```bash
cd ../backend && .venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
npm run typecheck    # tsc --noEmit
npm run lint         # eslint
npm run build        # production build
```

## Layout

```
src/
  app/                    layout.tsx, page.tsx, globals.css
  components/
    layout/               ResizableLayout, Sidebar, TopHeader, Workspace
    chat/                 ChatFeed, ImpactMetricsDeck, StrategySummaryCard,
                          PromptInputBar, StageProgress
    inspector/            InspectorPanel + TabAudienceD1 / TabRelevanceD2 /
                          TabPricingD3 / TabOptimizerD4, InspectorShell
    ui/                   Icon.tsx — the whole 16px monochrome SVG set
  hooks/useCampaignRun.ts the run state machine
  lib/
    api.ts                the only module that talks to FastAPI
    types.ts              mirrors the backend Pydantic contracts
    derive.ts             run record -> displayed values
    stages.ts             pipeline stages + SSE event mapping
    format.ts             display formatters
```

## Rules worth keeping

**`lib/types.ts` mirrors the backend Pydantic models — change both together.** Each block
names the module it mirrors (`app/models/campaign.py`, `app/models/optimization.py`, …).

**Only `lib/api.ts` fetches.** Components take data as props.

**Nothing invents a number.** `lib/derive.ts` restates what the optimizer computed; the
one derived ratio is effective CPM (cost over impressions), and it is labelled. When data
is absent a helper returns `null` and the panel shows an `AwaitingStage` empty state
naming the stage that produces it — never a zero or a placeholder.

**Stub provenance is always visible.** A run whose `stub_stages` is non-empty renders an
"Illustrative" badge in the header, a warning above the metrics deck, and a `StubNotice`
in each affected inspector tab. Do not suppress these — they are what stops placeholder
output being read as analysis.

**Infeasibility is shown, not hidden.** When the optimizer returns no package, the chat
renders the reason codes and relaxation options instead of a deck.

## How a run flows

1. `POST /campaign/stream` — SSE. `update` events drive the stage rail in `StageProgress`;
   the terminal `done` event carries the answer text, `run_id` and run snapshot.
   A paced run takes ~90s because of the shared Gemini rate limiter.
2. `GET /runs/{run_id}` — spec, artifact references, optimization result, validation.
3. `GET /runs/{run_id}/artifacts/{kind}` — artifact rows for the inspector. Two slices are
   fetched: the top of the ranked pool for D2, and the rows for the screens actually
   bought (`?screen_ids=…`) for the deck, D3 and D4. The bought screens sit anywhere in a
   250- or 750-row artifact, so a plain top-N slice misses most of them.

Chat transcripts are **not** persisted — localDB stores sessions, runs and uploads, but no
message log. Reopening an older session rehydrates the brief and package from its latest
run and says the written answer is not recoverable.

## Known gaps

- **Export Proposal PDF** calls `window.print()`. A real generated proposal needs a
  backend endpoint rendering the run record.
- **Settings** is a placeholder button with no panel behind it.
- **D1 proximity clusters** show real `screen_type` counts from the candidate pool. The
  mockup's POI categories (business hubs, retail corridors) and the 24-hour footfall curve
  need POI and ridership features the Data Agent owns and no endpoint exposes yet.
- **Session titles** stay "New Campaign" — the backend never renames a session from its
  brief.

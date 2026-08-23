# UI Specification: AgentIQ DOOH Media Planner

## 1. Design System & Philosophy

*   **Aesthetic**: High-density executive AI Copilot interface. Clean, fast, and ultra-minimalist with zero decorative emojis. All visual cues use 16px monochrome inline SVG icons.
*   **Color Palette**:
    *   **Primary Accent**: Deep Violet (`bg-violet-950` / `#3b0764`) for primary CTAs, active tab headers, and user chat bubbles.
    *   **Backgrounds**: Ultra-light zinc slate (`bg-zinc-50/50`, `bg-white`) for clean contrast and zero visual bloat.
    *   **Borders & Dividers**: Subtile zinc tones (`border-zinc-200/60`, `border-zinc-100`).
    *   **Status Indicators**: Soft Emerald (`bg-emerald-50`, `text-emerald-700`) for active, optimal, or online system signals.
*   **Performance & Interaction**:
    *   3-column resizable layout with draggable divider handles (`#resizer-left`, `#resizer-right`).
    *   Instantaneous client-side tab switching (<10ms) and floating bottom prompt bar.

---

## 2. Layout & Panel Breakdown

### Panel 1: Left Sidebar (Navigation & Sessions)
*   **Width**: Default `240px` (min `190px`, max `360px`).
*   **Top Header**: Brand logo icon (`AQ` in `bg-violet-950`), title **AgentIQ**, subtitle "Urban Media Engine".
*   **Primary CTA**: Single `+ New Campaign` full-width button (`bg-violet-950`, text white, rounded `xl`).
*   **Session History**: Scrollable list under header "PREVIOUS CAMPAIGNS".
    *   Row labels are the session's **campaign objective**, capped at five words by
        the backend — the same string the centre header shows. The rail never renders
        the raw user query; the stream's `session` event supplies a real name before
        the first model call.
    *   Active Session item: `bg-zinc-100/70`, `text-zinc-800`, border indicator.
    *   Inactive Session items: `text-zinc-500`, hover highlight `bg-zinc-50`.
*   **Bottom Section (Fixed)**:
    *   Active Inventory status card (`11,240` screens online with animated pulse dot).
    *   `Settings` button fixed at the very bottom with gear SVG icon. Opens the Settings modal (below); the currently selected model id is shown right-aligned in the same row, so the running model is visible without opening it.

### Panel 2: Center Workspace (Main Chat & Impact Deck)
*   **Width**: Dynamic flex fill (widest column, min `380px`).
*   **Top Bar**:
    *   Title: **Eastern Metro Campaign** (bold uppercase text).
    *   Status badge: "Orchestrated" with `bg-violet-950` status dot.
    *   Actions: "Reset" button & "Export Proposal PDF" button (`bg-zinc-700`).
*   **Chat Stream**:
    *   **User Bubble**: Right-aligned, `bg-violet-950`, rounded `2xl` with top-right `xs` corner.
    *   **AI Response Block**: Left-aligned with square `AI` avatar badge (`bg-zinc-700`).
    *   **Impact Metrics Deck (4 Cards)**:
        1.  *Est. Reach*: Highlighted card (`bg-zinc-700`, text white), value `4.25 M` (non-linear deduplicated).
        2.  *Target Spend*: `$48,600` (Under $50k Budget).
        3.  *Screen Count*: `38 Units` (Metro + Bus Corridor).
        4.  *Effective CPM*: `$11.43` (-14% vs Market Avg).
    *   **Strategy Summary Card**: High-level synthesis card outlining screen distribution (18 Metro Platforms + 20 Bus Stops) and rotation slot choices.
*   **Floating Prompt Input**:
    *   Positioned fixed at bottom center (`bottom-4 left-6 right-6`).
    *   Contains paperclip attachment icon, text input (`Refine plan...`), and a `Refine` button (`bg-violet-950`).

### Panel 3: Right Inspector Workspace (Step Reasoning Engine)
*   **Width**: Default `420px` (min `300px`, max `600px`).
*   **Tab Navigation (D1–D4 Engines)**:
    *   `D1: Audience`: Audience Profiling Engine.
    *   `D2: Relevance`: Campaign-Screen Relevance Scorer.
    *   `D3: Pricing`: Demand Forecasting & Dynamic Pricing Guardrails.
    *   `D4: Optimizer`: Impressions & Rotation Loop Optimizer.
*   **Tab Content Views**:
    *   **D1 Tab**: Proximity tags (Business Hubs <100m, Transit Interchanges, Retail Corridors) + footfall density sparkline chart across 24-hour time blocks.
    *   **D2 Tab**: Ranked candidate screen list with percentage fit pills (e.g., `96.4% Fit` in `bg-violet-950`).
    *   **D3 Tab**: Price range guardrail horizontal gauge (Floor: $28, Target: $42, Cap: $65) + 6-block time occupancy grid (`dim_slot`).
    *   **D4 Tab**: 6-slot rotation loop allocation matrix showing active partial rotation choices (Slot 1 & Slot 2 selected).

### Settings Modal (Panel 1 → gear)

*   **Trigger**: the sidebar's `Settings` button. Rendered as a sibling of
    `ResizableLayout`, not inside a panel — the overlay is `position: fixed` and a
    resizable column with `overflow-hidden` would clip it.
*   **Chrome**: centred card, `max-w-lg`, `rounded-2xl`, `bg-white`, over a
    `bg-zinc-900/25` backdrop. Closes on backdrop click, on `Escape`, and on `Close`.
*   **Only setting: Orchestration Model.** One block per provider (Google Gemini, Azure
    OpenAI), each listing its selectable models. The active one is `bg-violet-950` with a
    check; the rest are white rows with a hover border.
    *   Each model row shows the id as the title and, on Azure, the deployment it resolves
        to as a `text-[10px]` subtitle — the deployment is never the selectable value.
    *   Each provider header carries its request-per-minute cap, which is why a Gemini run
        takes ~90s and an Azure one ~45s, plus a `server default` marker.
    *   A provider with no credentials is **shown disabled with the reason** (the missing
        env var), not hidden. The rep was told they had the key; a silently absent option
        is not actionable.
*   **Persistence**: `localStorage` under `agentiq.model-selection`, re-validated against
    `GET /models` on load. Applies from the next turn — a package already built is not
    re-priced by switching.

---

## 3. Component Architecture & Props Matrix

```text
src/
├── components/
│   ├── layout/
│   │   ├── ResizableLayout.tsx       # Manages 3-column split & dragging handles
│   │   ├── SettingsDialog.tsx        # Settings modal — the model picker
│   │   ├── Sidebar.tsx               # Left panel (New Campaign, Sessions, Settings)
│   │   └── TopHeader.tsx             # Main header with title & export action
│   ├── chat/
│   │   ├── ChatFeed.tsx              # Renders user prompts & AI responses
│   │   ├── ImpactMetricsDeck.tsx     # 4-card metric grid (Reach, Spend, Screens, CPM)
│   │   ├── PromptInputBar.tsx        # Floating input with attachments & submit button
│   │   └── StrategySummaryCard.tsx   # Executive pitch summary
│   └── inspector/
│       ├── InspectorPanel.tsx        # Container for right sidebar tabs
│       ├── TabAudienceD1.tsx         # POI clusters & footfall sparklines
│       ├── TabRelevanceD2.tsx        # Screen ranking & affinity scores
│       ├── TabPricingD3.tsx          # Price guardrails & 6-block occupancy grid
│       └── TabOptimizerD4.tsx        # 6-slot rotation loop allocation matrix
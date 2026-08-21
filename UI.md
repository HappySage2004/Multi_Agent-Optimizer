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
*   **Session History**: Scrollable list under header "PREVIOUS SESSIONS".
    *   Active Session item: `bg-zinc-100/70`, `text-zinc-800`, border indicator.
    *   Inactive Session items: `text-zinc-500`, hover highlight `bg-zinc-50`.
*   **Bottom Section (Fixed)**:
    *   Active Inventory status card (`11,240` screens online with animated pulse dot).
    *   `Settings` button fixed at the very bottom with gear SVG icon.

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

---

## 3. Component Architecture & Props Matrix

```text
src/
├── components/
│   ├── layout/
│   │   ├── ResizableLayout.tsx       # Manages 3-column split & dragging handles
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
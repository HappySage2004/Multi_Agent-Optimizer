# AGENT_ARCHITECTURE.md

# Agent Architecture

## Overview

This document describes the architecture for our hackathon solution.

The primary design goal is to maximize reasoning capability while keeping the implementation small enough to build, debug and iterate within two days.

After evaluating several agent architectures, we intentionally avoid a large multi-agent system consisting of many independent agents. Instead, we adopt a **hierarchical planner-worker architecture**:

- One central **Manager Deep Agent**
- Two domain-specialist subagents
    - Forecasting Agent
    - Operations Research Agent
- Domain-specific execution tools

The manager is the only agent that interacts with the user.

The specialist agents function as **intelligent tools** that perform deep reasoning within their own domains.

---

# High-Level Architecture

```text
                          User
                            │
                            ▼
                 Manager Deep Agent
         (Planning + Orchestration + Verification)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Forecast Specialist              OR Specialist
        Deep Agent                    Deep Agent
            │                               │
      ┌─────┼──────┐                 ┌──────┼──────┐
      ▼     ▼      ▼                 ▼      ▼      ▼
   Python  Bash   RAG             Python   Bash   RAG
   Files   Cache  Search           Files  Solvers Search
            │                               │
            └──────────────┬────────────────┘
                           ▼
                Manager Verification
                           ▼
                     Final Response
```

---

# Why This Architecture?

The architecture is based on one simple principle:

> Agents should own reasoning.
>
> Tools should own execution.

The Manager Agent is responsible for:

- understanding the user's request
- breaking the problem into tasks
- deciding which specialist should solve each task
- combining outputs
- validating that the final response satisfies the user's objective

The specialist agents are responsible for:

- domain reasoning
- iterative experimentation
- deciding which tools to invoke
- producing structured outputs

The tools perform deterministic execution:

- running Python
- reading files
- calling optimization libraries
- plotting graphs
- loading datasets

---

# Why Not Many Agents?

Instead of building:

```text
Planner

↓

Research Agent

↓

Forecast Agent

↓

Optimization Agent

↓

Visualization Agent

↓

Reviewer Agent

↓

Writer Agent
```

we intentionally keep the architecture small.

Every additional LLM agent introduces:

- more latency
- more prompt engineering
- more debugging
- more context passing
- higher token usage
- additional failure modes

A manager with two specialist agents is expected to handle over 90% of the intended workload while remaining understandable and maintainable.

---

# Manager Deep Agent

## Responsibilities

The Manager Agent performs only high-level reasoning.

It should **never** perform forecasting or optimization itself.

Its workflow is:

```text
Receive Query

↓

Understand Objective

↓

Create Plan

↓

Delegate Tasks

↓

Collect Results

↓

Verify Completeness

↓

Request Additional Work (if required)

↓

Generate Final Response
```

---

## Responsibilities in Detail

### 1. Planning

The manager determines:

- What is the user asking?
- Which domains are involved?
- What dependencies exist?

Example:

User:

> Forecast next year's demand and optimize warehouse allocation.

Manager plan:

```
1. Forecast demand.

2. Optimize warehouse allocation.

3. Explain decisions.

4. Produce report.
```

---

### 2. Delegation

Instead of solving the task itself:

```
Forecast demand
```

the manager delegates:

```
Forecast Agent
```

Similarly:

```
Optimize production
```

becomes

```
OR Agent
```

---

### 3. Verification

After receiving outputs from all specialists, the manager verifies:

- Was every user requirement addressed?
- Is any information missing?
- Should another specialist be invoked?
- Should additional analysis be performed?

This replaces the need for a dedicated Reviewer Agent.

---

# Forecast Specialist Agent

## Responsibilities

The Forecast Agent owns **all forecasting-related reasoning**.

It decides:

- preprocessing strategy
- feature engineering
- model selection
- evaluation
- uncertainty estimation
- diagnostics

The manager never performs these decisions.

---

## Internal Workflow

```text
Receive Task

↓

Inspect Dataset

↓

Reason

↓

Execute Python

↓

Evaluate Results

↓

Reason Again

↓

Repeat if Needed

↓

Return Structured Output
```

Example reasoning:

```
Dataset has weekly seasonality.

↓

Perform stationarity test.

↓

Try Prophet.

↓

Performance poor.

↓

Try TFT.

↓

Improved accuracy.

↓

Return best model.
```

---

## Tools

The Forecast Agent receives:

- Python execution
- Bash
- File system
- Dataset loader
- Plotting
- Documentation search
- RAG
- Model registry (optional)

Notice that forecasting models are **not** individual tools.

The Forecast Agent dynamically writes Python to train whichever models it believes are appropriate.

---

# Operations Research Specialist

## Responsibilities

The OR Agent owns optimization reasoning.

It decides:

- problem formulation
- solver selection
- constraint relaxation
- sensitivity analysis
- feasibility checking

The manager never reasons about optimization algorithms.

---

## Internal Workflow

```text
Receive Task

↓

Understand Constraints

↓

Select Optimization Method

↓

Generate Model

↓

Run Solver

↓

Analyze Solution

↓

Retry if Needed

↓

Return Results
```

---

## Tools

The OR Agent receives:

- Python
- Bash
- OR-Tools
- Pyomo
- Simulation libraries
- Visualization
- Documentation search

Again, optimization algorithms are not individual tools.

The OR Agent writes Python dynamically.

---

# Why Python Is Not an Agent

Python is an execution environment.

It does not own reasoning.

Instead:

```
Forecast Agent

↓

Python Tool
```

and

```
OR Agent

↓

Python Tool
```

Both specialists execute Python independently.

Creating a dedicated Python Agent would unnecessarily separate reasoning from execution.

---

# Agent Communication

Communication is intentionally minimal.

```
Manager

↓

Task

↓

Specialist

↓

Structured Result

↓

Manager
```

The manager never receives:

- intermediate chain-of-thought
- internal experiments
- failed attempts

Only the final structured output is returned.

This prevents context bloat.

---

# Structured Outputs

Forecast Agent returns:

```python
class ForecastResult(BaseModel):
    model_used: str
    forecast: dict
    confidence_intervals: dict
    metrics: dict
    assumptions: list[str]
    recommendations: list[str]
```

OR Agent returns:

```python
class OptimizationResult(BaseModel):
    solver: str
    objective_value: float
    solution: dict
    feasibility: bool
    assumptions: list[str]
    recommendations: list[str]
```

Using structured outputs significantly simplifies downstream synthesis.

---

# Technology Stack

## Agent Framework

- LangGraph Deep Agents

Reason:

- planning-first architecture
- built-in subagent support
- isolated contexts
- checkpointing
- streaming
- middleware
- memory
- provider agnostic

---

## LLM

Initially:

- GPT-5.5

Future:

- Provider configurable

---

## Backend

- FastAPI

---

## Execution

- Python sandbox
- Bash
- File system
- Pandas
- NumPy
- Scikit-Learn
- PyTorch
- OR-Tools
- Pyomo
- Matplotlib

---

# FastAPI Project Structure

```
backend/

│
├── app.py
│
├── api/
│   ├── routes.py
│   └── schemas.py
│
├── agents/
│   │
│   ├── manager.py
│   ├── forecast_agent.py
│   ├── or_agent.py
│   ├── prompts.py
│   └── state.py
│
├── tools/
│   ├── python_tool.py
│   ├── filesystem.py
│   ├── plotting.py
│   ├── search.py
│   └── optimization.py
│
├── services/
│   ├── planner.py
│   ├── execution.py
│   └── synthesis.py
│
├── models/
│
├── config/
│
└── requirements.txt
```

---

# FastAPI Flow

```
POST /query

↓

FastAPI

↓

Manager Agent

↓

Planning

↓

Subagent Calls

↓

Verification

↓

Response

↓

Return JSON
```

---

# Manager Pseudocode

```python
def handle_query(query):

    plan = manager.plan(query)

    forecast_result = None
    optimization_result = None

    if plan.requires_forecast:
        forecast_result = forecast_agent.run(plan.forecast_task)

    if plan.requires_optimization:
        optimization_result = or_agent.run(plan.optimization_task)

    verified = manager.verify(
        query,
        forecast_result,
        optimization_result
    )

    if not verified.complete:
        # Manager may re-invoke one or more specialists
        ...

    return manager.summarize(
        query,
        forecast_result,
        optimization_result
    )
```

---

# Why LangGraph Deep Agents?

Deep Agents already provide:

- planning
- subagent delegation
- isolated execution contexts
- middleware
- memory
- filesystem support
- retries
- streaming
- checkpointing

The framework acts as an opinionated harness on top of LangChain and LangGraph rather than introducing a new runtime, making it a good fit for long-running, tool-heavy workflows with specialist subagents. :contentReference[oaicite:0]{index=0}

This allows us to focus on building high-quality specialist agents rather than orchestration infrastructure.

---

# Future Extensions

The architecture is intentionally extensible.

Additional specialists can be introduced without changing the manager's workflow.

Examples:

- Supply Chain Agent
- Finance Agent
- Scheduling Agent
- Simulation Agent
- Data Cleaning Agent

Each would simply become another callable subagent with its own tools and prompt.

---

# Design Principles

1. One manager owns orchestration.
2. Specialist agents own domain reasoning.
3. Tools execute deterministic actions.
4. Specialists return structured outputs.
5. Manager verifies against the original objective.
6. Keep the number of agents minimal.
7. Separate reasoning from execution.
8. Prefer reusable tools over prompt complexity.
9. Build specialist expertise through prompts, not orchestration.
10. Optimize for simplicity, reliability, and extensibility.
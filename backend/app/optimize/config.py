"""Every optimizer constant in one place, tagged by provenance.

    STRUCTURAL  validated against the source data; safe to trust
    ASSUMED     no data support in the 14 CSVs; ours to defend
    SOLVER      search-control knobs, no business meaning

Ported from the OR handoff bundle (`or_engine/config.py`), keeping its provenance
discipline. Dropped on the way in: `RATE_CARD_KEYS`, `SHRINK_PRIOR_N`, `STOP_PASS_*` and
`HOLIDAY_AS`, which belonged to the bundle's own rate-card and ridership modules. Those
capabilities already exist here as `app/ml/` and `app/tools/relevance_tools.py`, and a
second implementation of either would be a drift risk, not a fallback.
"""

from __future__ import annotations

# =============================================================================
# STRUCTURAL
# =============================================================================

# 6 slots per screen per time block per day. Confirmed empirically against `bookings`
# (the handoff reports 0 violations across 8.87M slot-days) and matches
# `relevance_tools.SLOTS_PER_BLOCK`.
SLOTS_PER_CELL = 6

# =============================================================================
# ASSUMED — the exposure model. See app/optimize/exposure.py for the derivation.
# =============================================================================

# Loop passes one person gets while in range of a screen, set by DWELL TIME. Holding k of
# 6 slots puts the creative on k of every 6 passes, so viewed exposures are LINEAR in slot
# count: buying more slots does not make the loop run faster.
LOOP_PASSES_PER_TRIP = 8

# Upstream audience volume is PEOPLE PASSING, not people who look at the screen. The
# attention discount lives here, applied exactly once, in `exposure.py`.
VIEWABILITY_IN_VEHICLE = 0.65  # captive for the whole ride
VIEWABILITY_STATIC = 0.35  # brief dwell, more distraction

# Pooled-reach saturation constant, used for a REPORTED DIAGNOSTIC ONLY. It is no longer in
# the solver's objective either: `R <= min(E, P)` is exact, lambda-free, and is the same
# quantity `or_agent_tools._package_metrics` reports and `validation._reach_checks`
# recomputes. This constant never reaches a client-facing number.
REACH_LAMBDA = 0.9

# Wear-out is expressed as a MULTIPLE of the flight's unavoidable exposure floor, not as an
# absolute number of exposures, and the distinction is the whole point.
#
# The floor is `LOOP_PASSES_PER_TRIP / SLOTS_PER_CELL * duration_days` — about 40 on a
# 30-day flight — and no allocation can go below it: duration is set by the brief, the
# minimum purchase is one slot, and daily allocation is constant (no flighting). An absolute
# cap of, say, 10 exposures is therefore unsatisfiable by any plan, and gating on one
# withholds every package rather than the bad ones. Measured on the canonical brief before
# this was relative: reach 152, awareness 361, frequency 361 exposures per person, all
# withheld, comparison empty.
#
# A multiple of the floor separates the two causes. 3.0 lets a plan buy up to three slots
# of depth beyond saturation, which is what makes the awareness and frequency profiles
# genuinely different plans rather than relabelled copies of the reach one; past that the
# frequency is a CHOICE the solver made — stacking slots and screens into a pool it had
# already saturated — and that is what gets penalized. Measured on the canonical brief, a
# multiple of 1.5 forbade all stacking and every profile collapsed onto the same plan.
WEAR_OUT_STACKING_MULTIPLE = 3.0

# Effective frequency, used only to normalize the objective's exposure term against its
# reach term. Not a target: the floor above already exceeds it on any real flight, which is
# a consequence of LOOP_PASSES_PER_TRIP and is fixed by dwell data, not by allocation.
EFFECTIVE_FREQUENCY_FLOOR = 3

# =============================================================================
# SOLVER
# =============================================================================

# Pooled reach is symmetric — many packages are near-identical in value — so proving
# optimality outright is not worth the wall clock. A solve that stops inside the gap is
# reported honestly as "feasible" rather than "optimal".
#
# 1% rather than the handoff's 5%, because the exact min(E, P) bound made the problem cheap
# enough to afford it. Measured on the canonical brief with slots up to 6: at a 5% gap reach
# ranged 247,595-255,625 across otherwise identical runs and one solve took 40s; at 1% it is
# 260,519-261,329 in 6-8s. The 5% gap was wide enough to swallow a 3.7% swing, which is
# larger than most effects worth reasoning about.
MIP_REL_GAP = 0.01

# No spend floor by default. This is a deliberate reversal of the handoff's
# MIN_SPEND_FRAC = 0.90, and the reason is principle rather than measured harm.
#
# A minimum budget utilisation is not a constraint any campaign spec declares, and
# SOLUTION.md section 2 is explicit that missing hard constraints are not to be invented.
# Where it does bind it forces the solver to convert leftover budget into depth, and on a
# reach brief those rupees buy repetition rather than people.
#
# Measured honestly: on the canonical brief with the current formulation the floor is INERT
# — floor 0.0 and floor 0.90 return byte-identical plans (49,708 spent, 25 screens, 261,329
# reached), because the reach-optimal plan already uses 99.4% of budget. The 12% reach cost
# observed earlier belonged to the tangent objective plus the reach profile's exposure
# weight, both since removed; it was not the floor's doing. Removing the floor is still
# right, but it is not what recovered that audience.
#
# A brief that genuinely requires utilisation can say so via
# `hard_constraints["min_budget_utilization"]`, and it is then enforced as hard and reported
# as a conflict if it cannot be met — never silently relaxed.
MIN_SPEND_FRACTION_DEFAULT = 0.0

# Cost enters the objective only as a tie-breaker, never as a trade-off against audience:
# at equal reach, prefer the cheaper package, and stop buying once there is nobody left to
# reach rather than spending the balance on repetition.
#
# The magnitude matters and was measured. At 1e-3 a line had to add roughly 24 people to be
# worth buying, which sounds negligible and was not: it cost 2,042 people (0.9% of reach) on
# an 80-candidate brief, because the solver reported "optimal" against an objective that was
# no longer pure reach. At 1e-6 the threshold is ~0.02 people — a true tie-break.
COST_TIE_BREAKER = 1e-6

# Reach saturates, so beyond a few screens in one audience pool the extra cells add
# symmetry for the solver to thrash on and frequency the client did not ask for.
MAX_CELLS_PER_POOL = 4

# Weight on the coverage-shortfall slack, on the normalized objective scale: one unit of
# shortfall costs this fraction of total reachable population.
COVERAGE_PENALTY = 0.10

# Wear-out slack is penalized ABOVE the largest exposure reward any profile offers
# (w_freq = 0.80 for the frequency profile), so breaching the cap always costs more than the
# exposures it buys. The cap therefore behaves as hard unless another hard constraint —
# an exact screen count in few pools, a zone-coverage minimum — forces a breach, in which
# case the plan is still returned and the breach is reported. At the handoff's 0.05 the
# penalty was an order of magnitude cheaper than the reward, so it never bound at all.
WEAR_OUT_PENALTY = 2.0

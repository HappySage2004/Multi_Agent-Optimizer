# DATASETS.md — Transit Media Dataset Reference

Reference for the 14 CSV files in [datasets/](datasets/) that back the Transit Media
Campaign Recommendation System described in [SOLUTION.md](SOLUTION.md).

All figures below were measured directly from the files (single-pass streaming scans),
not estimated.

---

## 1. At a Glance

| File | Size | Data rows | Grain / primary key | Domain |
|---|---:|---:|---|---|
| [ridership_actuals.csv](datasets/ridership_actuals.csv) | 124 MB | 2,049,632 | `(schedule_id, date)` | Transit network |
| [bookings.csv](datasets/bookings.csv) | 39 MB | 191,109 | `booking_id` | Commercial |
| [route_schedules.csv](datasets/route_schedules.csv) | 1.3 MB | 19,838 | `schedule_id` | Transit network |
| [screens.csv](datasets/screens.csv) | 592 KB | 11,163 | `screen_id` | Inventory |
| [route_stops.csv](datasets/route_stops.csv) | 204 KB | 2,436 | `(route_id, stop_sequence)` | Transit network |
| [lost_leads.csv](datasets/lost_leads.csv) | 392 KB | 1,450 | `lead_id` | Commercial |
| [points_of_interest.csv](datasets/points_of_interest.csv) | 180 KB | 1,375 | `poi_id` | Contextual |
| [locations.csv](datasets/locations.csv) | 72 KB | 910 | `location_id` | Geography |
| [vehicles.csv](datasets/vehicles.csv) | 36 KB | 854 | `vehicle_id` | Inventory |
| [client_facts.csv](datasets/client_facts.csv) | 88 KB | 520 | `client_id` | Commercial |
| [events.csv](datasets/events.csv) | 56 KB | 367 | `event_id` | Contextual |
| [zone_demographics.csv](datasets/zone_demographics.csv) | 4 KB | 30 | `zone_id` | Geography |
| [dim_slot.csv](datasets/dim_slot.csv) | 1 KB | 6 | `time_block_id` | Commercial |
| [cities.csv](datasets/cities.csv) | 1 KB | 3 | `city_id` | Geography |

**Total: ~2.28 M data rows across ~166 MB.** Two files (`ridership_actuals`, `bookings`)
hold 98.3% of all rows; the other twelve are small dimension/reference tables.

### Format conventions

- UTF-8, comma-delimited, newline-terminated, single header row.
- **No quoted fields anywhere** — no embedded commas, quotes, or newlines. Verified across
  all 14 files, so naive comma-splitting / `awk -F,` parsing is safe.
- Field counts are uniform per file (e.g. `ridership_actuals` is NF=7 on all 2,049,632 rows;
  `bookings` is NF=21 on all 191,109).
- Dates are `YYYY-MM-DD`. Times are `HH:MM` 24-hour. Booleans are the strings `True` / `False`
  (Python-style capitalization, **not** `true`/`1`).
- Nulls are empty strings, never `NULL` or `NaN`.

---

## 2. Handling the Large Files

> [!WARNING]
> **`datasets/ridership_actuals.csv` is 124 MB / 2.05 M rows.** Do not `cat` it, open it in an
> editor, load it into a chat context, or read it whole without a reason. `bookings.csv`
> (39 MB / 191 K rows) deserves the same care.

`ridership_actuals.csv` is listed in [.gitignore](.gitignore) and is **not** tracked in git —
it must be provisioned separately when cloning the repo. Every other CSV is committed.

**Safe inspection patterns:**

```bash
# peek at schema
head -3 datasets/ridership_actuals.csv

# count rows (streams, constant memory)
wc -l < datasets/ridership_actuals.csv

# aggregate in a single pass instead of loading
awk -F, 'NR>1 {s += $7; n++} END {print s/n}' datasets/ridership_actuals.csv

# slice a specific range of rows
sed -n '1000,1010p' datasets/ridership_actuals.csv
```

**Avoid:** `cat`, `sort` on the whole file (spills to disk), `pandas.read_csv()` without
`usecols`/`dtype`/`chunksize`, and any glob that sweeps the file into an LLM context.

**Recommended pandas load** — the naive read costs well over 1 GB of RAM from object-dtype
strings; this brings it to a few hundred MB:

```python
import pandas as pd

ridership = pd.read_csv(
    "datasets/ridership_actuals.csv",
    usecols=["schedule_id", "date", "day_of_week", "is_holiday", "actual_ridership"],
    dtype={
        "schedule_id": "category",
        "day_of_week": "category",
        "is_holiday": "bool",
        "actual_ridership": "int16",
    },
    parse_dates=["date"],
)
```

For repeated analysis, convert once to Parquet (`ridership.to_parquet(...)`) — it reloads
roughly an order of magnitude faster and preserves dtypes. Better still, pre-aggregate to
`(schedule_id, day_type)` means, which is the grain the demand-forecasting step in
SOLUTION.md actually consumes.

---

## 3. Join Map

```
Commercial                    Inventory                 Geography
──────────                    ─────────                 ─────────
client_facts <--client_id--+
                           |
dim_slot <--time_block_id--+
                           |
                       bookings --screen_id--> screens --location_id--> locations
                                                   |     (fixed only)      |
lost_leads --anchor_screen_id----------------------+                       +--zone_id--> zone_demographics
     |                                             |                       |
     +---client_id (44% null)--> client_facts      +--vehicle_id--> vehicles  +--city_id--> cities
                                                      (mobile only)   |
Contextual                                                            | corridor_id
──────────                                                            v
events --poi_id (23% null)--> points_of_interest               route_stops --location_id--> locations
   |                                  |                             ^
   +--anchor_location_id--------------+--anchor_location_id--> locations
                                                               route_id
                               route_schedules --route_id----------+
                                      ^
                                      +--schedule_id-- ridership_actuals
```

**Conditional joins** (the two easiest things to get wrong):

1. `screens` splits by `screen_type` into **fixed** and **mobile** inventory. Fixed screens
   (`bus_stop`, `metro_station` — 8,548 rows) have `location_id` set and `vehicle_id` empty.
   Mobile screens (`bus`, `metro_rail_coach` — 2,615 rows) have `vehicle_id` set and
   `location_id` empty. Neither join is total; always filter on the non-null side first.
2. Mobile screens reach geography only indirectly:
   `screens → vehicles.corridor_id → route_stops → locations`, which fans out to every stop
   on the corridor. There is no single location for a mobile screen by construction.

### ID conventions

Most keys are prefixed with the 2–3 letter `city_id`, so IDs are globally unique and the
owning city is readable from the key itself:

| Pattern | Example | Table |
|---|---|---|
| `{CITY}-LOC-####` | `LH-LOC-0120` | locations |
| `{CITY}-SCR-######` | `LH-SCR-000001` | screens |
| `{CITY}-VEH-#####` | `LH-VEH-00001` | vehicles |
| `{CITY}-RT-B###` / `-M###` | `LH-RT-B001` | corridor (`corridor_id`) |
| `{CITY}-RT-B###-OUT` / `-IN` | `LH-RT-B001-OUT` | route (`route_id`) |
| `{CITY}-SCH-######` | `LH-SCH-000001` | route_schedules |
| `{CITY}-POI-####` | `LH-POI-0001` | points_of_interest |
| `{CITY}-EVT-#####` | `ACS-EVT-00062` | events |
| `{CITY}-ZONE-###` | `LH-ZONE-001` | zone_demographics |
| `{CITY}-BKG-#######` | `DAT-BKG-0000001` | bookings |
| `CLI-#####` | `CLI-00001` | client_facts *(no city prefix)* |
| `LEAD-######` | `LEAD-000001` | lost_leads *(no city prefix)* |
| `DEAL-######` | `DEAL-000001` | bookings.deal_id *(no city prefix)* |

Note `route_id` = `corridor_id` + direction suffix, so a corridor always has exactly two
routes (94 corridors → 188 routes).

---

## 4. Geography

### cities.csv — 3 rows

The entire universe is three synthetic cities. Every other table partitions by `city_id`.

| Column | Type | Notes |
|---|---|---|
| `city_id` | str | PK. `LH`, `ACS`, `DAT` |
| `city_name` | str | Las Hackland, Accordionshire, DA Town |
| `population` | int | 3.2 M / 850 K / 1.45 M |
| `transit_density` | enum | `dense`, `sprawling`, `mixed` |
| `market_tier` | enum | `premium`, `value`, `standard` |
| `timezone` | str | IANA — `America/New_York`, `America/Chicago`, `America/Denver` |

| city_id | Name | Population | Tier | Zones | Locations | Screens | Vehicles | Bookings |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `LH` | Las Hackland | 3,200,000 | premium | 10 | 350 | 6,304 | 370 | 119,967 |
| `DAT` | DA Town | 1,450,000 | standard | 10 | 300 | 3,123 | 284 | 53,969 |
| `ACS` | Accordionshire | 850,000 | value | 10 | 260 | 1,736 | 200 | 17,173 |

> [!NOTE]
> The three cities are **very** unequal in the commercial tables — LH holds 62.8% of bookings
> and ACS only 9.0%. Any model trained pooled across cities will be LH-dominated. City-level
> normalization or stratification is worth considering.

### locations.csv — 910 rows

Physical fixed sites: bus stops and metro stations.

| Column | Type | Notes |
|---|---|---|
| `location_id` | str | PK |
| `city_id` | str | → cities |
| `name` | str | Intersection or station name, e.g. `Grant Rd & Kingsley Rd` |
| `city_zone` | str | Human-readable zone name; duplicates `zone_demographics.zone_name` |
| `zone_id` | str | → zone_demographics |
| `location_type` | enum | `bus_stop` (719), `metro_station` (191) |

### zone_demographics.csv — 30 rows

10 zones per city. The audience-targeting substrate for the relevance-scoring step.

| Column | Type | Notes |
|---|---|---|
| `zone_id` | str | PK |
| `city_id` | str | → cities |
| `zone_name` | str | e.g. `Downtown Core`, `Harborfront` |
| `resident_population` | int | |
| `population_density_per_sqkm` | int | |
| `median_age` | float | |
| `pct_age_under_18`, `pct_age_18_34`, `pct_age_35_54`, `pct_age_55_plus` | float | Percentages, sum ≈ 100 |
| `median_household_income` | int | USD |
| `income_index` | float | 100 = national baseline |
| `pct_bachelor_or_higher` | float | |
| `dominant_occupation` | enum | `mixed` (14), `white_collar` (7), `blue_collar` (3), `retail_service` (3), `student` (3) |
| `daytime_population_multiplier` | float | Daytime ÷ resident population. >1 = commuter inflow (e.g. Downtown Core 3.39); <1 = bedroom community |

`daytime_population_multiplier` is the key exposure lever — a dense downtown zone can carry
3× its residential population during working hours.

---

## 5. Inventory

### screens.csv — 11,163 rows

The sellable ad units. This is the central table of the whole dataset: bookings, leads, and
all optimization decision variables key off `screen_id`.

| Column | Type | Notes |
|---|---|---|
| `screen_id` | str | PK |
| `city_id` | str | → cities |
| `screen_type` | enum | `metro_station` (6,391), `bus_stop` (2,157), `metro_rail_coach` (1,400), `bus` (1,215) |
| `location_id` | str | → locations. **Set only for fixed screens** (8,548); empty for mobile |
| `vehicle_id` | str | → vehicles. **Set only for mobile screens** (2,615); empty for fixed |
| `position` | enum | Mounting position, e.g. `top`, `left` |
| `screen_size` | enum | `M` (4,528), `L` (3,428), `S` (3,207) |

Fixed/mobile split: **76.6% fixed** (8,548) / **23.4% mobile** (2,615). Exactly one of
`location_id` / `vehicle_id` is populated on every row — verified, no rows with both or neither.

Screens per city: LH 6,304 / DAT 3,123 / ACS 1,736.

### vehicles.csv — 854 rows

| Column | Type | Notes |
|---|---|---|
| `vehicle_id` | str | PK |
| `city_id` | str | → cities |
| `vehicle_type` | enum | `metro_train` (449), `bus` (405) |
| `corridor_id` | str | → route_stops.corridor_id. The corridor this vehicle runs |
| `screen_count` | int | Screens on board; matches the count in `screens` |

> [!IMPORTANT]
> `vehicles.vehicle_type` and `screens.screen_type` use **different vocabularies** for the same
> concept: a `metro_train` vehicle carries screens typed `metro_rail_coach`. Don't join or
> compare these two columns directly — map through `vehicle_id`.

---

## 6. Transit Network

### route_stops.csv — 2,436 rows

Ordered stop sequences. Grain is `(route_id, stop_sequence)`.

| Column | Type | Notes |
|---|---|---|
| `route_id` | str | PK part 1. 188 distinct |
| `corridor_id` | str | 94 distinct — one corridor = inbound + outbound route pair |
| `city_id` | str | → cities |
| `route_name` | str | e.g. `Route B1` |
| `mode` | enum | `bus` (1,660), `metro` (776) |
| `direction` | enum | `outbound` (1,218), `inbound` (1,218) — perfectly balanced |
| `stop_sequence` | int | PK part 2. 1-based |
| `location_id` | str | → locations |
| `is_first_stop`, `is_last_stop` | bool | `True`/`False` strings |
| `num_stops` | int | Total stops on the route (denormalized) |

### route_schedules.csv — 19,838 rows

One row per scheduled trip departure. The join bridge to ridership.

| Column | Type | Notes |
|---|---|---|
| `schedule_id` | str | PK |
| `route_id` | str | → route_stops.route_id |
| `corridor_id` | str | Denormalized from route_stops |
| `direction` | enum | `inbound` (9,943), `outbound` (9,895) |
| `day_type` | enum | `weekday` (13,052), `weekend` (6,786) |
| `start_time` | str | `HH:MM` departure time |
| `estimated_ridership` | int | **Planned** figure — contrast with `ridership_actuals.actual_ridership` |

`estimated_ridership` vs. actuals is the natural forecast-error baseline: any demand model
should beat simply reusing the planned number.

### ridership_actuals.csv — 2,049,632 rows ⚠️ 124 MB

Daily observed ridership per scheduled trip. **See §2 before reading this file.**

| Column | Type | Notes |
|---|---|---|
| `schedule_id` | str | PK part 1 → route_schedules. All 19,838 present |
| `route_id` | str | Denormalized |
| `city_id` | str | Denormalized |
| `date` | date | PK part 2. `2026-02-19` … `2026-08-19` |
| `day_of_week` | str | `Monday` … `Sunday` |
| `is_holiday` | bool | `True` on 2 dates only |
| `actual_ridership` | int | Observed boardings. mean 179.6, max 734 |

**Coverage is exactly complete, with no gaps:** 182 consecutive dates (26 weeks = 130 weekdays
+ 52 weekend days). Weekday schedules appear on all 130 weekdays, weekend schedules on all 52
weekend days: `13,052 × 130 + 6,786 × 52 = 2,049,632`. Every row reconciles — no missing
service days to impute.

| Day | Rows | Mean ridership |
|---|---:|---:|
| Friday | 339,352 | 216.5 |
| Thursday | 339,352 | 206.4 |
| Wednesday | 339,352 | 204.7 |
| Tuesday | 339,352 | 200.8 |
| Monday | 339,352 | 187.2 |
| Saturday | 176,436 | 74.4 |
| Sunday | 176,436 | 58.1 |

The weekday/weekend gap is severe — weekend ridership runs **~65% below** weekday. Weekend
schedules are also a thinner set (6,786 vs 13,052), so the drop is per-trip, not just fewer
trips. `day_type` is mandatory as a model feature; a pooled daily average is meaningless.

Holidays: `2026-05-25` (Memorial Day, Monday — runs the 13,052 weekday schedules) and
`2026-07-04` (Independence Day, Saturday — runs the 6,786 weekend schedules). Only 2 of 182
dates are flagged, so `is_holiday` is a ~1% positive-rate feature — too sparse to fit a
holiday effect on its own.

Rows per city: LH 845,754 / DAT 715,078 / ACS 488,800.

---

## 7. Commercial

### bookings.csv — 191,109 rows ⚠️ 39 MB

Historical sold inventory — the pricing and booking-probability training set.

| Column | Type | Notes |
|---|---|---|
| `booking_id` | str | PK |
| `deal_id` | str | 56,762 distinct — a deal spans multiple screens/line items (avg 3.4 rows) |
| `client_id` | str | → client_facts. All 520 clients appear |
| `city_id` | str | → cities |
| `screen_id` | str | → screens. 9,939 distinct = **89.0% of the 11,163 screens** have booking history |
| `ad_type` | str | Creative label, e.g. `Concert Tour Announcement (Frequency)` |
| `industry_vertical` | str | Denormalized from client_facts |
| `campaign_objective` | enum | `awareness` (75,534), `reach` (46,104), `conversion` (36,482), `frequency` (32,989) |
| `time_block_id` | int | → dim_slot |
| `daypart` | enum | `morning` (59,475), `evening` (47,142), `midday` (41,340), `afternoon` (26,082), `night` (17,070) |
| `slots_booked_per_day` | int | mean 2.65 |
| `rotation_type` | enum | `partial_rotation` (93,532), `single_rotation` (64,728), `full_exclusivity` (32,849) |
| `start_date`, `end_date` | date | `2025-08-19` … `2027-02-21` |
| `duration_days` | int | mean 73.0 |
| `booked_date` | date | Booking creation — gives lead time vs `start_date` |
| `contracted_price_per_slot_per_day` | float | **Target variable for the price model.** min 9.89, mean 80.26, max 221.01 |
| `line_item_value` | float | This row's value |
| `deal_total_value` | float | Whole-deal value; **repeats across every row of a deal — do not SUM** |
| `is_bundle` | bool | `True` (135,624) / `False` (55,485) — 71% bundled |
| `booking_status` | enum | `completed` (111,727), `upcoming` (49,428), `active` (29,954) |

> [!CAUTION]
> `deal_total_value` is denormalized to the line-item grain. Summing it over rows
> double-counts by ~3.4×. Use `SUM(line_item_value)` for revenue (total: $2.447 B), or
> deduplicate on `deal_id` first.

`booking_status` encodes time relative to a "today" inside the data (~mid-2026, consistent
with the ridership window ending `2026-08-19`). For backtesting, filter to `completed` — the
`active` and `upcoming` rows have not finished delivering.

### lost_leads.csv — 1,450 rows

Lost deals — the negative class. Without these, any booking-probability model trained on
`bookings` alone sees only wins.

| Column | Type | Notes |
|---|---|---|
| `lead_id` | str | PK |
| `client_id` | str | → client_facts. **44.3% null** (643 rows) — unidentified prospects |
| `company_name_raw` | str | **55.7% null** (807) — free-text, unnormalized |
| `industry_vertical` | str | |
| `city_id` | str | → cities |
| `requested_geography` | str | `CITY:Zone Name`, e.g. `LH:East Commons` — **composite string, needs parsing** |
| `anchor_screen_id` | str | → screens. Fully populated, 0 orphans |
| `lead_source` | enum | `repeat_client_inquiry` (500), `website_form` (250), `inbound_call` (195), `cold_outreach` (189), `referral` (185), `trade_show` (131) |
| `lead_date` | date | `2025-08-19` … `2026-08-16` |
| `sales_stage_reached` | enum | `initial_inquiry` (531), `quote_sent` (415), `negotiating` (281), `verbal_agreement` (141), `contract_sent` (82) |
| `lost_date` | date | |
| `requested_start_date` | date | |
| `requested_duration_days` | int | |
| `requested_num_screens` | int | |
| `indicated_budget` | float | Client-stated budget |
| `quoted_price_per_slot_per_day` | float | Our ask |
| `client_target_price_per_slot_per_day` | float | Their counter |
| `price_gap_pct` | float | Relative gap; mean 0.0945 |
| `negotiation_rounds` | int | |
| `competitor_mentioned` | bool | `True` on 257 (17.7%) |
| `loss_reason` | enum | `price_too_high` (305), `no_response_ghosted` (262), `budget_mismatch` (186), `went_with_competitor` (153), `timing_conflict` (125), `contract_terms_disagreement` (118), `inventory_unavailable` (107), `campaign_cancelled_internally` (96), `targeting_mismatch` (87), `creative_not_ready` (11) |
| `loss_reason_detail` | str | Free-text elaboration |
| `campaign_objective` | enum | Same vocabulary as bookings |
| `ad_type` | str | Same style as bookings |

> [!WARNING]
> **The class balance is extreme: 1,450 losses vs 191,109 bookings (0.75%).** These are not
> comparable samples — bookings are line items, leads are whole opportunities. Aggregate
> bookings to `deal_id` (56,762) before pairing, and even then expect ~2.5% positive rate.
> Class weighting or calibration is required.

Price-sensitivity signal is unusually clean here: `quoted_price_per_slot_per_day`,
`client_target_price_per_slot_per_day`, and `price_gap_pct` are all present on every lost lead,
and 34% of losses are explicitly price-driven (`price_too_high` + `budget_mismatch`).

### client_facts.csv — 520 rows

Advertiser dimension.

| Column | Type | Notes |
|---|---|---|
| `client_id` | str | PK |
| `company_name` | str | |
| `industry` | str | 13 distinct verticals |
| `client_tier` | enum | `local_business` (294), `regional_chain` (149), `national_chain` (77) |
| `home_city_id` | str | → cities |
| `active_cities` | str | **Composite** — one or more city codes |
| `preferred_geographies` | str | **Composite**, e.g. `DAT:Central Yard` — same format as `lost_leads.requested_geography` |
| `typical_campaign_budget` | float | USD |
| `budget_variance_pct` | float | |
| `campaign_frequency` | enum | `one_off` (178), `seasonal` (165), `quarterly` (129), `always_on` (48) |
| `avg_campaign_duration_days` | int | |
| `bundle_affinity` | enum | `single_screen` (277), `moderate_bundle` (160), `heavy_bundle` (83) |
| `negotiation_leverage` | enum | `low` (262), `medium` (180), `high` (78) |
| `relationship_start_date` | date | Tenure feature |
| `account_status` | enum | `active` (465), `lapsed` (55) |

### dim_slot.csv — 6 rows

Time-block dimension. Six contiguous 4-hour blocks covering the full day.

| `time_block_id` | `time_block_label` | `start_hour` | `end_hour` | `nearest_daypart` |
|---|---|---:|---:|---|
| 1 | 00:00-04:00 | 0 | 4 | night |
| 2 | 04:00-08:00 | 4 | 8 | morning |
| 3 | 08:00-12:00 | 8 | 12 | midday |
| 4 | 12:00-16:00 | 12 | 16 | afternoon |
| 5 | 16:00-20:00 | 16 | 20 | evening |
| 6 | 20:00-24:00 | 20 | 24 | night |

`nearest_daypart` is **not** 1:1 — `night` maps to both block 1 and block 6. Grouping by
daypart merges two non-adjacent blocks (late night and late evening), which behave very
differently. Prefer `time_block_id` for modeling.

---

## 8. Contextual

### points_of_interest.csv — 1,375 rows

Footfall generators near screens — the exposure-uplift signal.

| Column | Type | Notes |
|---|---|---|
| `poi_id` | str | PK |
| `city_id` | str | → cities |
| `city_zone` | str | Zone name |
| `name` | str | |
| `poi_type` | enum | 13 types. Largest: `shopping_mall` (225), `grocery_anchor` (202), `office_park` (175), `residential_tower` (154), `entertainment_district` (141) |
| `scale` | enum | `neighborhood` (507), `minor` (401), `major` (358), `flagship` (109) |
| `est_daily_footfall` | int | |
| `anchor_location_id` | str | → locations. Nearest transit location. 0 orphans |
| `distance_to_location_km`, `distance_to_location_mi` | float | **Redundant pair** — same value, two units. Keep one |
| `is_network_hub` | bool | `True` on 627 (45.6%) |
| `side_of_road` | enum | e.g. `far_side` — matters for physical visibility |
| `peak_daypart` | enum | Joins conceptually to `dim_slot.nearest_daypart` |

`stadium_arena` has only 3 rows — too rare for a per-type effect, worth folding into a
broader category.

### events.csv — 367 rows

Scheduled demand spikes.

| Column | Type | Notes |
|---|---|---|
| `event_id` | str | PK |
| `city_id` | str | → cities |
| `city_zone` | str | |
| `poi_id` | str | → points_of_interest. **23.4% null** (86 rows) |
| `anchor_location_id` | str | → locations. Fully populated, 0 orphans |
| `event_name` | str | |
| `event_type` | enum | `sports_game` (82), `concert` (70), `festival` (41), `community_fair` (33), `convention` / `trade_show` / `parade` (29 each), `holiday_event` / `political_rally` (21 each), `marathon_race` (12) |
| `recurrence` | enum | `one_time` (264), `weekly_season` (82), `annual` (21) |
| `start_date`, `end_date` | date | `2025-08-19` … `2027-02-19` |
| `expected_attendance` | int | |
| `attendance_tier` | enum | `large` (175), `medium` (175), `small` (17) |
| `primary_impact_daypart` | enum | |
| `impact_radius_km` | float | Geographic blast radius, e.g. 2.34 |

The 86 null `poi_id` values are **not** broken references — they are street-level events
(parades, political rallies, marathon races) that occur along routes rather than at a venue.
Always use `anchor_location_id` for spatial joins; it is populated on all 367 rows.

`recurrence = weekly_season` (82 rows) means one row represents **many** occurrences. Expanding
events to a daily grid requires materializing those repeats between `start_date` and `end_date`
— treating each row as a single day undercounts event exposure substantially.

---

## 9. Data Quality Summary

**Referential integrity is clean.** All foreign-key edges were checked; every non-null
value resolves, with zero true orphans:

| Edge | Result |
|---|---|
| `bookings.screen_id` → screens | 191,109 matched / 0 orphans |
| `bookings.client_id` → client_facts | 191,109 matched / 0 orphans |
| `bookings.time_block_id` → dim_slot | 191,109 matched / 0 orphans |
| `ridership_actuals.schedule_id` → route_schedules | 2,049,632 matched / 0 orphans |
| `screens.location_id` → locations (fixed) | 8,548 matched / 0 orphans |
| `screens.vehicle_id` → vehicles (mobile) | 2,615 matched / 0 orphans |
| `route_stops.location_id` → locations | 2,436 matched / 0 orphans |
| `route_schedules.route_id` → route_stops | 19,838 matched / 0 orphans |
| `vehicles.corridor_id` → route_stops | 854 matched / 0 orphans |
| `locations.zone_id` → zone_demographics | 910 matched / 0 orphans |
| `points_of_interest.anchor_location_id` → locations | 1,375 matched / 0 orphans |
| `events.anchor_location_id` → locations | 367 matched / 0 orphans |
| `events.poi_id` → points_of_interest | 281 matched, 86 null, 0 orphans |
| `lost_leads.anchor_screen_id` → screens | 1,450 matched / 0 orphans |
| `lost_leads.client_id` → client_facts | 807 matched, 643 null, 0 orphans |

**Known nulls** (all intentional, all structural):

| Table.column | Null rate | Meaning |
|---|---:|---|
| `screens.vehicle_id` | 76.6% | Fixed screens have no vehicle |
| `screens.location_id` | 23.4% | Mobile screens have no fixed location |
| `lost_leads.company_name_raw` | 55.7% | Unnormalized prospect not yet identified |
| `lost_leads.client_id` | 44.3% | Prospect never became a client |
| `events.poi_id` | 23.4% | Street-level event, no venue |

**Things to watch:**

1. `deal_total_value` in bookings is denormalized — summing it double-counts ~3.4×.
2. `distance_to_location_km` / `_mi` in POI are the same measurement twice.
3. `vehicles.vehicle_type` (`metro_train`) ≠ `screens.screen_type` (`metro_rail_coach`).
4. `nearest_daypart` in dim_slot is many-to-one (`night` covers blocks 1 and 6).
5. Composite string columns need parsing: `requested_geography`, `preferred_geographies`,
   and `active_cities` all use `CITY:Zone` / multi-code form.
6. Booleans are the strings `True`/`False` — most CSV readers will type them as `object`
   unless told otherwise.
7. `city_zone` and `industry_vertical` are denormalized onto several fact tables; they can
   drift from their dimension source. Prefer joining through the ID.

**Temporal windows** — the per-table date ranges, for reference:

| Table | Field | Range |
|---|---|---|
| ridership_actuals | `date` | 2026-02-19 → 2026-08-19 (182 days) |
| bookings | `start_date` → `end_date` | 2025-08-19 → 2027-02-21 (~18 months) |
| lost_leads | `lead_date` | 2025-08-19 → 2026-08-16 |
| events | `start_date` | 2025-08-19 → 2027-02-19 |

The ridership window covers only the middle 6 months of the commercial window, but **this is
not a limitation in practice** — the two sides are not joined on dates. Ridership serves the
audience/exposure side: it is aggregated to per-schedule and per-`day_type` averages, rolled up
through `route_stops` and `locations` into a screen-level exposure profile. The commercial
tables serve pricing and booking probability. They meet at `screen_id` and geography, not on a
shared calendar.

So treat the two windows as independent. A row-level join of a booking's flight dates against
observed daily ridership is not part of the design, and the non-overlap should not drive
train/test split decisions on either side.

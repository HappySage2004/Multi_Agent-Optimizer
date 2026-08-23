# EDA — Transit Media Campaign Datasets


---

## 1. Inventory is fixed-heavy, and concentrated in metro stations

11,163 screens: **8,548 stop-mounted (76.6%)** and **2,615 vehicle-mounted (23.4%)**. The
split is clean — zero screens have both a `location_id` and a `vehicle_id`, zero have
neither, so the fixed/mobile union never double-counts.

| screen_type | screens | share | mounting |
|---|---:|---:|---|
| metro_station | 6,391 | 57.3% | fixed |
| bus_stop | 2,157 | 19.3% | fixed |
| metro_rail_coach | 1,400 | 12.5% | mobile |
| bus | 1,215 | 10.9% | mobile |

Screens per location is **bimodal, not a distribution**: every one of the 719 bus stops has
exactly 3 screens, while the 191 metro stations carry **34 on average and up to 50**. There
is no middle. Any per-location average (the overall mean is 9.4) describes nothing real.

**Why this matters.** Buying "25 screens" means something completely different at a metro
station than across bus stops, and a single metro station can supply half a package on its
own — from one audience pool. Pooling is not an edge case here; it is the dominant effect,
since screens at the same stop see the same crowd and reach has to be deduplicated per pool
rather than summed per screen.

On the mobile side, corridors carry a **median of 6 vehicles (max 25)**. Since
`route_schedules` has no `vehicle_id`, corridor ridership has to be divided by that count
to get one vehicle's share — the single largest assumption in any mobile audience figure.

---

## 2. Ridership: metro dominates, and block 1 is structurally empty

| | departures | riders | riders/departure | share |
|---|---:|---:|---:|---:|
| metro | 12,756 | 2,730,462 | 214.1 | **91.6%** |
| bus | 7,082 | 248,927 | 35.1 | 8.4% |

Metro carries **6.1× the riders per departure** and 11× the total. Combined with metro
stations holding 57% of screens, transit volume and inventory point the same way.

**Weekday vs weekend is two effects, not one:**

- weekend/weekday **total** ridership: **0.211**
- weekend/weekday **per departure**: **0.405**
- weekend/weekday **departures scheduled**: **0.520**

**Why this matters.** A campaign priced per day needs the per-departure number; one sizing
a flight needs the total. Using the 0.21 figure where 0.41 belongs cuts weekend value in
half twice over.

**Block 1 (00:00–04:00) has zero scheduled departures — and 8,544 real bookings.** Any
schedule-derived audience figure is exactly 0.0 there. That is "not modelled", never
"nobody there", and it should never be shown to a buyer as a measured zero.

Finally, a route has **13.0 stops on average** (8–21). Riders board and alight along it, so
crediting a route's whole ridership to each of its stops overstates stop-level volume by
roughly that factor — a ~13× trap sitting in the most natural join in the schema.

---

## 3. Zone demographics: 30 zones, and one field does most of the work

| field | min | max | max/min |
|---|---:|---:|---:|
| population_density_per_sqkm | 1,277 | 14,946 | 11.7× |
| pct_age_18_34 | 10.0 | 65.8 | 6.6× |
| income_index | 73.5 | 171.7 | 2.3× |
| pct_age_35_54 | 17.2 | 38.9 | 2.3× |
| median_age | 23.9 | 48.7 | 2.0× |

`dominant_occupation` covers 5 categories over 30 zones, and it is **not** a proxy for
income — it is ordered by it:

| dominant_occupation | zones | mean income_index | mean pct_18_34 |
|---|---:|---:|---:|
| white_collar | 7 | 155.4 | 27.0 |
| mixed | 14 | 99.8 | 23.3 |
| retail_service | 3 | 91.1 | 29.1 |
| blue_collar | 3 | 81.5 | 21.8 |
| student | 3 | 79.5 | **65.4** |

**Why this matters.** A binary "is white_collar" flag scores `mixed`, `retail_service` and
`blue_collar` identically at 0, discarding an ordering the income column shows is real —
and `mixed` is 47% of all zones. Student zones are unmistakable on age (65% aged 18–34,
nearly 3× the mean) but sit at the bottom on income, so a "young professionals" brief and a
"students" brief must not resolve to the same screens.

The three cities are near-identical on averages (mean income_index 106.7 / 107.1 / 110.3)
and each holds exactly 10 zones with the same occupation mix. **City is not a targeting
signal here; zone is.**

---

## 4. Bookings: price is real, and it is mostly explained by inventory attributes

191,109 bookings across 9,939 screens and 520 clients. **1,224 screens have never been
booked** — a cold-start segment no history-based model can score.

Status splits **58.5% completed / 25.9% upcoming / 15.7% active**, with no cancelled or
failed category at all. That is worth saying plainly: the "completion rate" available here
is a *fulfilment* outcome, and it says nothing about whether an ad worked.

Price (`contracted_price_per_slot_per_day`) spans **$9.89 to $221.01**, mean $80.26, CV
0.45. It is far from random:

| segmentation | R² on price | segments |
|---|---:|---:|
| screen_type | 0.194 | 4 |
| + screen_size | 0.367 | 9 |
| + position | 0.367 | 15 |
| + city_id | 0.512 | 45 |
| + time_block_id | **0.691** | 268 |

**Why this matters.** Roughly **69% of price variance is explained by five inventory
attributes** — a defensible price band can be built from comparables alone, without a
black-box model. Note `position` adds *nothing* once type and size are known (it is nearly
collinear: a coach has no position, a bus stop is never a platform), while city and time
block carry the most. Segment counts also grow fast — 268 cells over 191 K bookings — so a
band needs a fallback ladder for thin cells.

Time-block pricing tracks the commute: **block 5 (16:00–20:00) $94.29** and **block 2
(04:00–08:00) $90.10** against **block 6 $38.20** and **block 1 $38.02**. Buyers already
pay ~2.4× for peak.

`slots_booked_per_day` runs 1–6 (mean 2.65). With 6 rotation slots cycling continuously
through each 4-hour block, slot *position* is meaningless — this field is **share of
voice**, and exposures are linear in it.

**The demand-side data is thin.** `lost_leads.csv` has 1,450 rows against 191,109 bookings,
and only 305 are `price_too_high`. Any booking-probability model trained on this is
learning from a ~500:1 class imbalance, which bounds how much it can honestly claim.

---

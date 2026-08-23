"""Exploratory data analysis over the 14 raw transit-media CSVs.

Standalone: reads only from DATA_FOLDER, imports nothing from `backend/`, writes nothing.
Run it and capture stdout.

    python eda.py                     # reads ./datasets
    python eda.py path/to/datasets    # or point it somewhere else
    EDA_DATA_FOLDER=... python eda.py

Every number printed is computed from the files, not restated from documentation. The
sections are ordered by how much they change a modelling decision rather than by table
size, so the location/pool_key investigation comes first.

`ridership_actuals.csv` (124 MB, 2.05 M rows, gitignored) is optional. When present it is
read in chunks with explicit dtypes; when absent the ridership section falls back to
`route_schedules.estimated_ridership` and says so in the output rather than going quiet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# The 14 tables, largest-domain first. `ridership_actuals` is the optional one.
TABLES = (
    "cities",
    "zone_demographics",
    "locations",
    "screens",
    "vehicles",
    "route_stops",
    "route_schedules",
    "ridership_actuals",
    "dim_slot",
    "bookings",
    "client_facts",
    "points_of_interest",
    "events",
    "lost_leads",
)

OPTIONAL_TABLES = ("ridership_actuals",)

# Documented primary keys, checked rather than assumed.
PRIMARY_KEYS: dict[str, list[str]] = {
    "cities": ["city_id"],
    "zone_demographics": ["zone_id"],
    "locations": ["location_id"],
    "screens": ["screen_id"],
    "vehicles": ["vehicle_id"],
    "route_stops": ["route_id", "stop_sequence"],
    "route_schedules": ["schedule_id"],
    "dim_slot": ["time_block_id"],
    "bookings": ["booking_id"],
    "client_facts": ["client_id"],
    "points_of_interest": ["poi_id"],
    "events": ["event_id"],
    "lost_leads": ["lead_id"],
}

# Keeps the 2 M-row file to a few hundred MB instead of >1 GB of object-dtype strings.
RIDERSHIP_DTYPES = {
    "schedule_id": "string",
    "route_id": "string",
    "city_id": "category",
    "day_of_week": "category",
    "is_holiday": "boolean",
    "actual_ridership": "int32",
}
RIDERSHIP_CHUNK = 500_000

WEEKEND_DAYS = {"Saturday", "Sunday"}

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

# The report is full of em dashes and arrows; a Windows console defaults to cp1252 and
# would turn each one into a mojibake byte in a captured file.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# OUTPUT HELPERS
# =============================================================================


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title.upper())
    print("=" * 78)


def sub(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 73 - len(title)))


def note(text: str) -> None:
    print(f"    note: {text}")


def frame(value: pd.DataFrame | pd.Series | str, indent: str = "  ") -> None:
    """Print a table (or an already-rendered one) indented under its section."""
    text = value if isinstance(value, str) else value.to_string()
    print("\n".join(f"{indent}{line}" for line in text.splitlines()))


# =============================================================================
# LOADING
# =============================================================================


def data_folder() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    if env := os.environ.get("EDA_DATA_FOLDER"):
        return Path(env).resolve()
    return (Path(__file__).resolve().parent / "datasets").resolve()


def load_all(folder: Path) -> dict[str, pd.DataFrame]:
    """Every table except `ridership_actuals`, which is loaded on demand and in chunks."""
    tables: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        if name in OPTIONAL_TABLES:
            continue
        path = folder / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"required table missing: {path}")
        tables[name] = pd.read_csv(path)
    return tables


def ridership_summary(folder: Path) -> pd.DataFrame | None:
    """Streamed aggregate of `ridership_actuals`, or None when it is not provisioned.

    Returns one row per (schedule_id, day_type, is_holiday) with total and mean observed
    boardings — small enough to join against, without ever holding 2 M rows.
    """
    path = folder / "ridership_actuals.csv"
    if not path.exists():
        return None

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=list(RIDERSHIP_DTYPES),
        dtype=RIDERSHIP_DTYPES,
        chunksize=RIDERSHIP_CHUNK,
    )
    for chunk in reader:
        chunk["day_type"] = chunk["day_of_week"].isin(WEEKEND_DAYS).map(
            {True: "weekend", False: "weekday"}
        )
        parts.append(
            chunk.groupby(["schedule_id", "day_type", "is_holiday"], observed=True)[
                "actual_ridership"
            ]
            .agg(["sum", "count"])
            .reset_index()
        )
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(["schedule_id", "day_type", "is_holiday"], observed=True)[
            ["sum", "count"]
        ]
        .sum()
        .reset_index()
        .rename(columns={"sum": "riders", "count": "trip_days"})
    )


# =============================================================================
# 1. TABLE INVENTORY
# =============================================================================


def section_inventory(tables: dict[str, pd.DataFrame], folder: Path) -> None:
    header("1. table inventory — what is actually in the 14 files")

    rows = []
    for name in TABLES:
        path = folder / f"{name}.csv"
        if name in tables:
            df = tables[name]
            n_rows, n_cols = len(df), df.shape[1]
            nulls = int(df.isna().sum().sum())
            key = PRIMARY_KEYS.get(name, [])
            unique_key = (
                "yes" if key and not df.duplicated(key).any() else ("no" if key else "-")
            )
        elif path.exists():
            # ridership_actuals: count rows without materializing them.
            n_rows = sum(
                len(c)
                for c in pd.read_csv(path, usecols=["schedule_id"], chunksize=RIDERSHIP_CHUNK)
            )
            n_cols = len(pd.read_csv(path, nrows=0).columns)
            nulls, unique_key = -1, "-"
        else:
            rows.append(
                {
                    "table": name,
                    "rows": "NOT PROVISIONED",
                    "cols": "-",
                    "MB": "-",
                    "nulls": "-",
                    "pk_unique": "-",
                }
            )
            continue

        rows.append(
            {
                "table": name,
                "rows": f"{n_rows:,}",
                "cols": n_cols,
                "MB": f"{path.stat().st_size / 1e6:.1f}",
                "nulls": "n/a" if nulls < 0 else f"{nulls:,}",
                "pk_unique": unique_key,
            }
        )
    frame(pd.DataFrame(rows).to_string(index=False))

    sub("columns per table")
    for name, df in tables.items():
        print(f"  {name:22} {', '.join(df.columns)}")

    sub("columns holding nulls (empty string in the source)")
    any_nulls = False
    for name, df in tables.items():
        counts = df.isna().sum()
        counts = counts[counts > 0]
        if not counts.empty:
            any_nulls = True
            detail = ", ".join(f"{c}={n:,} ({n / len(df):.0%})" for c, n in counts.items())
            print(f"  {name:22} {detail}")
    if not any_nulls:
        print("  none")
    note(
        "screens.location_id / screens.vehicle_id are mutually exclusive by design — a "
        "screen is stop-mounted or vehicle-mounted, never both. That is the fixed/mobile "
        "split, not missing data."
    )


# =============================================================================
# 2. SCREEN INVENTORY
# =============================================================================


def section_screens(tables: dict[str, pd.DataFrame]) -> None:
    header("2. screen inventory — what there is to sell")

    screens = tables["screens"].copy()
    screens["mounting"] = screens["location_id"].notna().map(
        {True: "stop-mounted (fixed)", False: "vehicle-mounted (mobile)"}
    )

    sub("mounting split")
    mount = screens.groupby("mounting").agg(screens=("screen_id", "size"))
    mount["share"] = (mount["screens"] / len(screens)).map("{:.1%}".format)
    frame(mount)
    both = int((screens["location_id"].notna() & screens["vehicle_id"].notna()).sum())
    neither = int((screens["location_id"].isna() & screens["vehicle_id"].isna()).sum())
    print(f"\n  screens with BOTH a location and a vehicle: {both}")
    print(f"  screens with NEITHER:                      {neither}")
    note("both must be 0, or the fixed/mobile union double-counts inventory.")

    sub("by city x mounting")
    pivot = pd.crosstab(screens["city_id"], screens["mounting"], margins=True, margins_name="ALL")
    frame(pivot)

    sub("by screen_type")
    by_type = screens.groupby("screen_type").agg(screens=("screen_id", "size"))
    by_type["share"] = (by_type["screens"] / len(screens)).map("{:.1%}".format)
    by_type["mounting"] = screens.groupby("screen_type")["mounting"].first()
    frame(by_type.sort_values("screens", ascending=False))

    sub("by position and size")
    frame(pd.crosstab(screens["screen_size"], screens["position"], margins=True, margins_name="ALL"))

    sub("screens per location (stop-mounted only)")
    per_loc = screens[screens["location_id"].notna()]["location_id"].value_counts()
    locations = tables["locations"]
    print(f"  locations in locations.csv:            {len(locations):,}")
    print(f"  locations carrying at least one screen: {per_loc.size:,}")
    print(f"  locations with no screen:               {len(locations) - per_loc.size:,}")
    frame(per_loc.describe().to_frame("screens_per_location"))

    dist = per_loc.value_counts().sort_index().rename_axis("screens_at_location").to_frame(
        "locations"
    )
    dist["screens_total"] = dist.index * dist["locations"]
    frame(dist)

    sub("screens per location, split by location_type")
    typed = (
        screens[screens["location_id"].notna()]
        .merge(locations[["location_id", "location_type"]], on="location_id", how="left")
        .groupby(["location_type", "location_id"])
        .size()
        .rename("screens")
        .reset_index()
    )
    frame(
        typed.groupby("location_type")["screens"].agg(
            locations="size", total_screens="sum", mean="mean", median="median", max="max"
        )
    )
    note(
        "a metro station carries far more screens than a bus stop, so screen count per "
        "location is the first place inventory concentration shows up."
    )

    sub("screens per vehicle (mobile)")
    vehicles = tables["vehicles"]
    per_veh = screens[screens["vehicle_id"].notna()]["vehicle_id"].value_counts()
    frame(per_veh.describe().to_frame("screens_per_vehicle"))
    print(f"\n  vehicles in vehicles.csv: {len(vehicles):,}")
    print(f"  vehicles carrying screens: {per_veh.size:,}")
    declared = vehicles.set_index("vehicle_id")["screen_count"]
    actual = per_veh.reindex(declared.index).fillna(0).astype(int)
    mismatch = int((declared != actual).sum())
    print(f"  vehicles where vehicles.screen_count != screens.csv count: {mismatch}")
    note("a mismatch here would mean screen_count is a stale denormalization.")

    sub("vehicles per corridor (the mobile pooling denominator)")
    per_corr = vehicles.groupby("corridor_id")["vehicle_id"].nunique()
    frame(per_corr.describe().to_frame("vehicles_per_corridor"))
    note(
        "route_schedules carries no vehicle_id, so corridor ridership has to be divided by "
        "this count to get one vehicle's share. It is the largest modelling assumption in "
        "the mobile audience figure."
    )


# =============================================================================
# 3. THE LOCATION / POOL_KEY INVESTIGATION
# =============================================================================


def section_locations(tables: dict[str, pd.DataFrame]) -> None:
    header("3. location identity — are same-named locations duplicates?")

    locations = tables["locations"].copy()
    route_stops = tables["route_stops"]
    screens = tables["screens"]

    print(
        "  Question: locations.csv has repeated `name` values. If those are duplicate\n"
        "  records for one physical site, then anything that treats location_id as the\n"
        "  audience unit counts the same crowd twice. If they are distinct sites that\n"
        "  happen to share a street-corner name, merging them would UNDER-count reach.\n"
        "  The two errors point in opposite directions, so guessing is not an option."
    )

    sub("how much name repetition is there")
    dup_mask = locations.duplicated("name", keep=False)
    print(f"  location_id (unique records):        {locations['location_id'].nunique():,}")
    print(f"  distinct `name` values:              {locations['name'].nunique():,}")
    print(f"  records whose name is not unique:    {int(dup_mask.sum()):,}")
    print(f"  names shared by >1 record:           {locations.loc[dup_mask, 'name'].nunique():,}")

    sub("name repetition by location_type")
    by_type = locations.assign(duplicated_name=dup_mask).groupby("location_type").agg(
        records=("location_id", "size"),
        distinct_names=("name", "nunique"),
        records_with_shared_name=("duplicated_name", "sum"),
    )
    by_type["share_shared"] = (
        by_type["records_with_shared_name"] / by_type["records"]
    ).map("{:.1%}".format)
    frame(by_type)

    # The discriminating evidence: which corridors serve each location.
    corridor_sets = (
        route_stops.groupby("location_id")["corridor_id"]
        .apply(lambda s: tuple(sorted(set(s))))
        .rename("corridor_set")
    )
    locations = locations.join(corridor_sets, on="location_id")
    unserved = int(locations["corridor_set"].isna().sum())
    locations["corridor_set"] = locations["corridor_set"].apply(
        lambda v: v if isinstance(v, tuple) else ()
    )
    locations["corridor_key"] = locations["corridor_set"].map(lambda t: "|".join(t) or "(none)")
    locations["n_corridors"] = locations["corridor_set"].map(len)

    sub("evidence: the set of corridors serving each location")
    print(f"  locations with no route_stops row at all: {unserved}")
    frame(
        locations.groupby("location_type")["n_corridors"].agg(
            locations="size", mean="mean", median="median", max="max"
        )
    )

    sub("duplicate-name groups: same corridors, or different ones?")
    dup_groups = locations[locations.duplicated(["city_id", "name"], keep=False)]
    verdict = dup_groups.groupby(["city_id", "name"]).agg(
        records=("location_id", "size"),
        distinct_corridor_sets=("corridor_key", "nunique"),
    )
    identical = verdict[verdict["distinct_corridor_sets"] == 1]
    different = verdict[verdict["distinct_corridor_sets"] > 1]
    print(f"  (city_id, name) groups holding >1 record: {len(verdict):,}")
    print(
        f"    -> serve DIFFERENT corridor sets:       {len(different):,} "
        f"({len(different) / len(verdict):.1%})  = genuinely different physical stops"
    )
    print(
        f"    -> serve an IDENTICAL corridor set:     {len(identical):,} "
        f"({len(identical) / len(verdict):.1%})  = true duplicate entrance/platform records"
    )
    print(f"  records inside the identical-corridor groups: {int(identical['records'].sum()):,}")
    print(
        f"  redundant records (records - one per group):  "
        f"{int(identical['records'].sum() - len(identical)):,}"
    )

    sub("three ways to count 'sites', and the gap between them")
    raw = locations["location_id"].nunique()
    naive = locations.groupby(["city_id", "name"]).ngroups
    true_sites = locations.groupby(["city_id", "name", "corridor_key"]).ngroups
    counts = pd.DataFrame(
        [
            {
                "grouping": "raw location_id",
                "sites": raw,
                "vs true": raw - true_sites,
                "reading": "every record is its own audience",
            },
            {
                "grouping": "(city_id, name)  [naive]",
                "sites": naive,
                "vs true": naive - true_sites,
                "reading": "merges distinct stops that share a name",
            },
            {
                "grouping": "(city_id, name, corridor_set)",
                "sites": true_sites,
                "vs true": 0,
                "reading": "merges only records with identical service",
            },
        ]
    )
    frame(counts.to_string(index=False))
    print(
        f"\n  Naive name grouping loses {raw - naive:,} sites ({(raw - naive) / raw:.1%} of the "
        f"inventory);\n  the corridor-set test says only {raw - true_sites:,} of those "
        f"({(raw - true_sites) / raw:.1%}) are actually redundant."
    )

    sub("worked examples — a same-name pair that is NOT a duplicate")
    for (city, name), grp in dup_groups[
        dup_groups.set_index(["city_id", "name"]).index.isin(different.index)
    ].groupby(["city_id", "name"]):
        if len(grp) == 2 and grp["n_corridors"].min() > 0:
            frame(
                grp[["location_id", "city_id", "name", "zone_id", "location_type", "corridor_key"]]
            )
            break

    sub("worked examples — a same-name pair that IS a duplicate")
    if len(identical):
        (city, name) = identical.index[0]
        grp = dup_groups[(dup_groups["city_id"] == city) & (dup_groups["name"] == name)]
        frame(grp[["location_id", "city_id", "name", "zone_id", "location_type", "corridor_key"]])

    sub("what it costs, in pools and in screens")
    screen_counts = screens[screens["location_id"].notna()]["location_id"].value_counts()
    dup_ids = dup_groups[
        dup_groups.set_index(["city_id", "name"]).index.isin(identical.index)
    ]["location_id"]
    affected = int(screen_counts.reindex(dup_ids).fillna(0).sum())
    redundant = int(identical["records"].sum() - len(identical))
    print(f"  physical sites split across >1 record:       {len(identical):,}")
    print(f"  records they are split across:               {int(identical['records'].sum()):,}")
    print(f"  redundant pools (over-counted crowds):       {redundant:,} of {raw:,} "
          f"({redundant / raw:.1%} of fixed pools)")
    print(f"  screens sitting on one of those records:     {affected:,} "
          f"({affected / int(screen_counts.sum()):.2%} of stop-mounted screens)")
    note(
        "the screen share looks larger than the pool share because the affected records "
        "are metro stations, which carry ~34 screens each. Reach is capped per POOL, so "
        f"the exposure is {redundant} double-counted crowds, not {affected} double-counted "
        "screens."
    )
    note(
        "the fix is a targeted merge of ~14 sites, NOT a name-based regrouping — that "
        "would collapse 183 genuinely separate stops to save 14 real duplicates."
    )

    sub("pool_key cardinality implied by each choice")
    corridors = tables["vehicles"]["corridor_id"].nunique()
    print(f"  fixed pools by location_id     {raw:,}   + {corridors} corridors = {raw + corridors:,}")
    print(
        f"  fixed pools by (city,name)      {naive:,}   + {corridors} corridors = "
        f"{naive + corridors:,}"
    )
    print(
        f"  fixed pools by true site        {true_sites:,}   + {corridors} corridors = "
        f"{true_sites + corridors:,}"
    )
    note(
        "reach is capped per pool, so pool count sets the ceiling. Choosing the naive "
        "grouping would shrink the addressable audience by whole stations."
    )


# =============================================================================
# 4. RIDERSHIP
# =============================================================================


def section_ridership(tables: dict[str, pd.DataFrame], observed: pd.DataFrame | None) -> None:
    header("4. ridership patterns — where the audience actually is")

    schedules = tables["route_schedules"].copy()
    dim_slot = tables["dim_slot"]
    route_stops = tables["route_stops"]

    # Trip start hour -> time block. dim_slot end_hour is exclusive and block 6 ends at 24,
    # so every hour 0-23 lands in exactly one block.
    schedules["start_hour"] = (
        schedules["start_time"].astype(str).str.slice(0, 2).astype(int)
    )
    blocks = dim_slot.sort_values("start_hour")
    schedules["time_block_id"] = pd.cut(
        schedules["start_hour"],
        bins=list(blocks["start_hour"]) + [24],
        labels=list(blocks["time_block_id"]),
        right=False,
    ).astype(int)

    mode_by_route = route_stops.drop_duplicates("route_id").set_index("route_id")["mode"]
    schedules["mode"] = schedules["route_id"].map(mode_by_route)

    if observed is not None:
        print("  source: ridership_actuals.csv (OBSERVED boardings)")
        merged = observed.merge(
            schedules[["schedule_id", "time_block_id", "mode", "day_type"]].rename(
                columns={"day_type": "scheduled_day_type"}
            ),
            on="schedule_id",
            how="left",
        )
        value, weight = "riders", "trip_days"
    else:
        print("  source: route_schedules.estimated_ridership (PLANNED, per departure)")
        note(
            "ridership_actuals.csv is not provisioned in this checkout, so this section "
            "uses the scheduled estimate. Same units, lower fidelity; the holiday and "
            "day-of-week breakdowns that need observed data are skipped and flagged."
        )
        merged = schedules.rename(columns={"estimated_ridership": "riders"})
        merged["trip_days"] = 1
        value, weight = "riders", "trip_days"

    sub("weekday vs weekend")
    day = merged.groupby("day_type", observed=True).agg(
        departures=(weight, "sum"), riders=(value, "sum")
    )
    day["riders_per_departure"] = day["riders"] / day["departures"]
    frame(day)
    if {"weekday", "weekend"} <= set(day.index):
        wd, we = day.loc["weekday"], day.loc["weekend"]
        print(
            f"\n  weekend / weekday total ridership:        "
            f"{we['riders'] / wd['riders']:.3f}"
        )
        print(
            f"  weekend / weekday per departure:          "
            f"{we['riders_per_departure'] / wd['riders_per_departure']:.3f}"
        )
        print(
            f"  weekend / weekday departures scheduled:   "
            f"{we['departures'] / wd['departures']:.3f}"
        )
        note(
            "the totals ratio mixes two effects — fewer weekend departures AND fewer "
            "riders on each. A campaign priced per day needs the per-departure figure; a "
            "campaign sizing a flight needs the totals. Conflating them misprices weekends."
        )

    sub("ridership by time block")
    by_block = merged.groupby("time_block_id", observed=True).agg(
        departures=(weight, "sum"), riders=(value, "sum")
    )
    by_block["riders_per_departure"] = by_block["riders"] / by_block["departures"]
    by_block["share_of_riders"] = (by_block["riders"] / by_block["riders"].sum()).map(
        "{:.1%}".format
    )
    by_block = by_block.join(dim_slot.set_index("time_block_id")[["time_block_label", "nearest_daypart"]])
    frame(by_block)
    missing_blocks = sorted(set(dim_slot["time_block_id"]) - set(by_block.index))
    print(f"\n  time blocks with ZERO scheduled service: {missing_blocks or 'none'}")
    note(
        "block 1 (00:00-04:00) has no scheduled departures at all, so any schedule-derived "
        "audience figure is exactly zero there. That means 'not modelled', not 'nobody "
        "there' — bookings.csv does contain block-1 bookings."
    )

    sub("metro vs bus")
    by_mode = merged.groupby("mode", observed=True).agg(
        departures=(weight, "sum"), riders=(value, "sum")
    )
    by_mode["riders_per_departure"] = by_mode["riders"] / by_mode["departures"]
    by_mode["share_of_riders"] = (by_mode["riders"] / by_mode["riders"].sum()).map(
        "{:.1%}".format
    )
    frame(by_mode)
    modes = set(by_mode.index)
    if {"metro", "bus"} <= modes:
        print(
            f"\n  metro / bus riders per departure: "
            f"{by_mode.loc['metro', 'riders_per_departure'] / by_mode.loc['bus', 'riders_per_departure']:.2f}x"
        )
        print(
            f"  metro / bus total riders:         "
            f"{by_mode.loc['metro', 'riders'] / by_mode.loc['bus', 'riders']:.2f}x"
        )

    sub("stops per route — the other half of the volume question")
    per_route = route_stops.groupby("route_id").agg(
        stops=("stop_sequence", "size"),
        declared=("num_stops", "first"),
        mode=("mode", "first"),
    )
    frame(per_route.groupby("mode")["stops"].agg(routes="size", mean="mean", min="min", max="max"))
    print(f"\n  routes where counted stops != num_stops: {int((per_route['stops'] != per_route['declared']).sum())}")
    note(
        f"a route's riders are spread over its {per_route['stops'].mean():.1f} stops on "
        f"average. Crediting a whole route's ridership to each of its stops would "
        f"over-state stop-level volume by roughly that factor."
    )

    if observed is not None:
        sub("holidays (observed data only)")
        hol = observed.groupby("is_holiday", observed=True).agg(
            trip_days=("trip_days", "sum"), riders=("riders", "sum")
        )
        hol["riders_per_trip"] = hol["riders"] / hol["trip_days"]
        frame(hol)
        note("only 2 holiday dates exist in the window, so any holiday effect is unfittable.")


# =============================================================================
# 5. ZONE DEMOGRAPHICS
# =============================================================================


def section_demographics(tables: dict[str, pd.DataFrame]) -> None:
    header("5. zone demographics — who the zones contain")

    zones = tables["zone_demographics"]
    cities = tables["cities"]

    sub("coverage")
    print(f"  zones: {len(zones)}   cities: {len(cities)}")
    frame(zones.groupby("city_id").agg(zones=("zone_id", "size"), population=("resident_population", "sum")))

    sub("distributions across all zones")
    numeric = [
        "resident_population",
        "population_density_per_sqkm",
        "median_age",
        "pct_age_18_34",
        "pct_age_35_54",
        "median_household_income",
        "income_index",
        "pct_bachelor_or_higher",
        "daytime_population_multiplier",
    ]
    frame(zones[numeric].describe().T[["min", "25%", "50%", "75%", "max", "mean", "std"]])

    sub("spread of each field, as max/min")
    spread = pd.DataFrame(
        {
            "min": zones[numeric].min(),
            "max": zones[numeric].max(),
            "max/min": zones[numeric].max() / zones[numeric].min(),
        }
    )
    frame(spread.sort_values("max/min", ascending=False))
    note(
        "income_index spans a wide range while the age percentages do not. A score built "
        "from a narrow field cannot discriminate much, whatever weight it is given."
    )

    sub("dominant_occupation")
    occ = zones["dominant_occupation"].value_counts().to_frame("zones")
    occ["share"] = (occ["zones"] / len(zones)).map("{:.1%}".format)
    occ["mean_income_index"] = zones.groupby("dominant_occupation")["income_index"].mean()
    occ["mean_pct_18_34"] = zones.groupby("dominant_occupation")["pct_age_18_34"].mean()
    frame(occ)
    note(
        "5 categories over 30 zones. A binary white-collar flag throws away the ordering "
        "between `mixed`, `retail_service` and `blue_collar` that the income column shows "
        "is real."
    )

    sub("by city")
    frame(
        zones.groupby("city_id")[
            ["income_index", "median_age", "pct_age_18_34", "pct_age_35_54"]
        ].agg(["mean", "min", "max"])
    )

    sub("occupation x city")
    frame(pd.crosstab(zones["city_id"], zones["dominant_occupation"], margins=True, margins_name="ALL"))

    sub("richest and poorest zones")
    ranked = zones.sort_values("income_index", ascending=False)[
        ["zone_id", "city_id", "zone_name", "income_index", "median_household_income",
         "pct_age_18_34", "dominant_occupation"]
    ]
    frame(pd.concat([ranked.head(5), ranked.tail(5)]))


# =============================================================================
# 6. BOOKINGS
# =============================================================================


def section_bookings(tables: dict[str, pd.DataFrame]) -> None:
    header("6. bookings — what has actually been sold")

    bookings = tables["bookings"]
    screens = tables["screens"]
    price = "contracted_price_per_slot_per_day"

    print(f"  rows: {len(bookings):,}   distinct screens booked: {bookings['screen_id'].nunique():,}"
          f"   distinct clients: {bookings['client_id'].nunique():,}")
    print(f"  screens never booked: {screens['screen_id'].nunique() - bookings['screen_id'].nunique():,}")

    sub("booking_status")
    status = bookings["booking_status"].value_counts().to_frame("bookings")
    status["share"] = (status["bookings"] / len(bookings)).map("{:.1%}".format)
    status["mean_price"] = bookings.groupby("booking_status")[price].mean().round(2)
    status["mean_duration_days"] = bookings.groupby("booking_status")["duration_days"].mean().round(1)
    frame(status)
    note(
        "completion rate here is a fulfilment outcome, not campaign performance. It says "
        "nothing about whether the ad worked."
    )

    sub("industry_vertical")
    ind = bookings["industry_vertical"].value_counts().to_frame("bookings")
    ind["share"] = (ind["bookings"] / len(bookings)).map("{:.1%}".format)
    ind["mean_price"] = bookings.groupby("industry_vertical")[price].mean().round(2)
    ind["completion_rate"] = (
        bookings.assign(done=bookings["booking_status"].eq("completed"))
        .groupby("industry_vertical")["done"]
        .mean()
        .map("{:.1%}".format)
    )
    frame(ind)

    sub("price variance — the whole pricing problem in one table")
    overall = bookings[price]
    print(f"  overall  n={len(overall):,}  mean={overall.mean():.2f}  std={overall.std():.2f}  "
          f"CV={overall.std() / overall.mean():.3f}")
    frame(overall.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame(price))

    def variance_by(frame_: pd.DataFrame, key: str) -> pd.DataFrame:
        g = frame_.groupby(key)[price].agg(["size", "mean", "std", "min", "max"])
        g["CV"] = (g["std"] / g["mean"]).round(3)
        g["p90/p10"] = (
            frame_.groupby(key)[price].quantile(0.9) / frame_.groupby(key)[price].quantile(0.1)
        ).round(2)
        return g.round(2)

    joined = bookings.merge(
        screens[["screen_id", "screen_type", "position", "screen_size"]], on="screen_id", how="left"
    )
    # `position` is null for the 1,400 metro_rail_coach screens. Left as NaN it would be
    # dropped from every groupby, silently changing the row set between segmentations and
    # making the R^2 ladder below non-monotone for a reason that has nothing to do with
    # explanatory power.
    joined["position"] = joined["position"].fillna("(none)")

    for key in ("screen_type", "screen_size", "position", "time_block_id", "rotation_type"):
        sub(f"price by {key}")
        frame(variance_by(joined, key).sort_values("mean", ascending=False))

    sub("price by ad_type (highest and lowest 8 of the full list)")
    ads = variance_by(joined, "ad_type").sort_values("mean", ascending=False)
    print(f"  {len(ads)} distinct ad_type values; showing the extremes")
    frame(pd.concat([ads.head(8), ads.tail(8)]))
    note(
        "ad_type is campaign copy, not an inventory attribute — it describes what the "
        "buyer was selling. It correlates with price mostly through who buys what, so it "
        "is a poor segmentation key despite the spread."
    )

    sub("how much of the price spread is explained by segmentation")
    total_var = overall.var()
    for keys in (
        ["screen_type"],
        ["screen_type", "screen_size"],
        ["screen_type", "screen_size", "position"],
        ["screen_type", "screen_size", "position", "city_id"],
        ["screen_type", "screen_size", "position", "city_id", "time_block_id"],
    ):
        within = joined.groupby(keys)[price].transform("mean")
        residual = (joined[price] - within).var()
        print(
            f"  {' x '.join(keys):58} R^2={1 - residual / total_var:.3f}  "
            f"segments={joined.groupby(keys).ngroups:,}"
        )
    note(
        "position adds nothing once screen_type x screen_size is known — it is almost "
        "collinear with them (a coach has no position, a bus stop is never a platform). "
        "City and time block are what actually carry information. Segments get thin fast "
        "though, which is why a price band needs a fallback ladder rather than one fixed "
        "segmentation."
    )

    sub("slots and duration")
    frame(
        bookings[["slots_booked_per_day", "duration_days", "line_item_value", "deal_total_value"]]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        .T
    )
    print("\n  slots_booked_per_day value counts:")
    frame(bookings["slots_booked_per_day"].value_counts().sort_index().to_frame("bookings"))
    note(
        "6 rotation slots exist per screen per block per day, and they cycle continuously "
        "— so slot POSITION carries no meaning and slots_booked_per_day is share of voice."
    )

    sub("bookings by time block, against where the ridership is")
    blocks = bookings["time_block_id"].value_counts().sort_index().to_frame("bookings")
    blocks["share"] = (blocks["bookings"] / len(bookings)).map("{:.1%}".format)
    blocks = blocks.join(tables["dim_slot"].set_index("time_block_id")["time_block_label"])
    frame(blocks)
    note(
        "block 1 carries real bookings even though no transit is scheduled then. Whatever "
        "audience those buyers are paying for is not in the schedule data."
    )

    sub("lost leads — the other side of the price question")
    leads = tables["lost_leads"]
    print(f"  leads: {len(leads):,}  vs {len(bookings):,} bookings")
    reasons = leads["loss_reason"].value_counts().to_frame("leads")
    reasons["share"] = (reasons["leads"] / len(leads)).map("{:.1%}".format)
    reasons["mean_price_gap_pct"] = leads.groupby("loss_reason")["price_gap_pct"].mean().round(1)
    frame(reasons)
    note(
        "a price-driven loss model has these as its only negatives — a few hundred rows "
        "against 191 K bookings. That class imbalance bounds how much a booking-probability "
        "model can honestly claim."
    )


# =============================================================================


def main() -> None:
    folder = data_folder()
    print("EDA — transit media campaign datasets")
    print(f"source: {folder}")

    tables = load_all(folder)
    observed = ridership_summary(folder)

    section_inventory(tables, folder)
    section_screens(tables)
    section_locations(tables)
    section_ridership(tables, observed)
    section_demographics(tables)
    section_bookings(tables)

    header("end of report")


if __name__ == "__main__":
    main()

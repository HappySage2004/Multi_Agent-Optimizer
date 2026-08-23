"""DuckDB access over the source CSVs.

Scope note: this module is the one place the source CSVs are read. Add views here rather
than reading CSVs elsewhere. `_create_audience_views` is the feature layer behind
`app/tools/relevance_tools.py`; `app/ml/loaders.py` issues column-projected SQL against
the base views for pricing.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pandas as pd

from app.config import get_settings

# Tables that must exist for the app to start.
REQUIRED_TABLES = (
    "cities",
    "zone_demographics",
    "locations",
    "screens",
    "vehicles",
    "route_stops",
    "route_schedules",
    "dim_slot",
    "bookings",
    "client_facts",
    "points_of_interest",
    "events",
    "lost_leads",
)

# Present only when provisioned separately (124 MB, gitignored).
OPTIONAL_TABLES = ("ridership_actuals",)

_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


def _csv_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    # A view, not a table: DuckDB streams the CSV on demand, so the 124 MB
    # ridership file is never materialized in memory.
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true, sample_size=-1)"
    )


def get_connection() -> duckdb.DuckDBPyConnection:
    """Process-wide DuckDB connection with every available CSV registered as a view."""
    global _conn
    with _lock:
        if _conn is not None:
            return _conn

        settings = get_settings()
        con = duckdb.connect(database=":memory:")

        missing = []
        for name in REQUIRED_TABLES:
            path = settings.datasets_dir / f"{name}.csv"
            if not path.exists():
                missing.append(name)
                continue
            _csv_view(con, name, path)
        if missing:
            raise FileNotFoundError(
                f"Missing required dataset CSVs in {settings.datasets_dir}: {', '.join(missing)}"
            )

        for name in OPTIONAL_TABLES:
            path = settings.datasets_dir / f"{name}.csv"
            if path.exists():
                _csv_view(con, name, path)

        _create_views(con)
        _conn = con
        return _conn


def _create_views(con: duckdb.DuckDBPyConnection) -> None:
    """Stable views. `v_screen_geography` resolves the fixed/mobile split for every screen."""
    con.execute(
        """
        CREATE OR REPLACE VIEW v_screen_geography AS
        -- Fixed screens: one row, one location.
        SELECT
            s.screen_id,
            s.city_id,
            s.screen_type,
            s.screen_size,
            s.position,
            'fixed'          AS inventory_class,
            l.location_id,
            l.zone_id,
            l.location_type,
            NULL             AS corridor_id
        FROM screens s
        JOIN locations l ON s.location_id = l.location_id
        WHERE s.location_id IS NOT NULL

        UNION ALL

        -- Mobile screens: no single location by construction. Zone is left NULL and the
        -- corridor is the geographic unit. Do NOT fan out to route_stops here — that
        -- would multiply one screen into every stop on its corridor.
        SELECT
            s.screen_id,
            s.city_id,
            s.screen_type,
            s.screen_size,
            s.position,
            'mobile'         AS inventory_class,
            NULL             AS location_id,
            NULL             AS zone_id,
            NULL             AS location_type,
            v.corridor_id
        FROM screens s
        JOIN vehicles v ON s.vehicle_id = v.vehicle_id
        WHERE s.vehicle_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW v_corridor_zones AS
        SELECT DISTINCT rs.corridor_id, l.zone_id, l.city_id
        FROM route_stops rs
        JOIN locations l ON rs.location_id = l.location_id
        """
    )
    _create_audience_views(con)


def _create_audience_views(con: duckdb.DuckDBPyConnection) -> None:
    """Feature views behind the audience relevance engine (`app/tools/relevance_tools.py`).

    `v_screen_demand_history` is the one that matters: average daily ridership per screen,
    per dim_slot time block, per day type. It is the audience-volume signal the whole
    relevance and reach story rests on, so the joins are spelled out here rather than
    being reassembled in pandas by every caller.

    Views are created in dependency order — DuckDB binds a view when it is defined, so a
    view cannot reference one that does not exist yet.
    """
    _create_poi_view(con)

    # Vehicles working each corridor. Defined once and used in TWO places that must agree:
    # `v_corridor_block_demand` divides corridor ridership by it to get one vehicle's share,
    # and `v_screen_profile` publishes it as `pool_partition_count` so the optimizer can
    # multiply back up to the corridor's whole crowd. If these two ever used different
    # counts, pool populations would silently disagree with the per-screen figures.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_corridor_vehicle_count AS
        SELECT corridor_id, greatest(count(DISTINCT vehicle_id), 1) AS n_vehicles
        FROM vehicles
        GROUP BY 1
        """
    )

    # One row per scheduled departure, tagged with the time block its start hour falls in.
    # dim_slot end_hour is exclusive, and block 6 ends at 24, so every hour 0-23 matches
    # exactly one block.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_schedule_block AS
        SELECT
            s.schedule_id,
            s.route_id,
            s.corridor_id,
            s.day_type,
            s.estimated_ridership,
            d.time_block_id
        FROM route_schedules s
        JOIN dim_slot d
          ON hour(CAST(s.start_time AS TIME)) >= d.start_hour
         AND hour(CAST(s.start_time AS TIME)) <  d.end_hour
        """
    )

    # Corridor-level block totals, divided by the vehicles working that corridor. A screen
    # inside ONE vehicle is exposed only to the trips THAT vehicle makes, not to every trip
    # on the corridor, and route_schedules carries no vehicle_id — so one vehicle's share
    # is the available approximation. This is a modelling judgement, not an exact figure.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_corridor_block_demand AS
        WITH corridor AS (
            SELECT corridor_id, time_block_id, day_type,
                   sum(estimated_ridership) AS riders
            FROM v_schedule_block
            GROUP BY 1, 2, 3
        )
        SELECT
            c.corridor_id,
            c.time_block_id,
            c.day_type,
            c.riders / coalesce(v.n_vehicles, 1) AS avg_daily_ridership
        FROM corridor c
        LEFT JOIN v_corridor_vehicle_count v USING (corridor_id)
        """
    )

    # Route-level block totals for stop-mounted screens.
    #
    # actual_ridership is PER TRIP: one row is one departure on one date. A screen at a
    # stop is passed by every trip in the block, so trips are SUMMED within a day and only
    # then averaged ACROSS days. Averaging first would describe a single vehicle rather
    # than the volume passing the stop, understating it by roughly the trip count.
    if _has_table(con, "ridership_actuals"):
        con.execute(
            """
            CREATE OR REPLACE VIEW v_route_block_demand AS
            WITH daily AS (
                SELECT
                    sb.route_id,
                    sb.time_block_id,
                    CASE WHEN lower(r.day_of_week) IN ('saturday', 'sunday')
                         THEN 'weekend' ELSE 'weekday' END       AS day_type,
                    r.date,
                    sum(r.actual_ridership)                      AS riders
                FROM ridership_actuals r
                JOIN v_schedule_block sb ON r.schedule_id = sb.schedule_id
                GROUP BY 1, 2, 3, 4
            )
            SELECT route_id, time_block_id, day_type,
                   avg(riders) AS avg_daily_ridership
            FROM daily
            GROUP BY 1, 2, 3
            """
        )
    else:
        # ridership_actuals is gitignored and optional. Degrade to the scheduled estimate
        # — the same quantity the corridor path already uses — rather than failing. Lower
        # fidelity, same units; `demand_source` on the engine reports which one is live.
        con.execute(
            """
            CREATE OR REPLACE VIEW v_route_block_demand AS
            SELECT route_id, time_block_id, day_type,
                   sum(estimated_ridership) AS avg_daily_ridership
            FROM v_schedule_block
            GROUP BY 1, 2, 3
            """
        )

    _create_route_stop_weight_view(con)

    con.execute(
        """
        CREATE OR REPLACE VIEW v_screen_demand_history AS
        -- Stop-mounted: SUM across every route serving the location, but each route
        -- contributes only THIS STOP'S SHARE of its riders — see
        -- `_create_route_stop_weight_view` for why the share is essential.
        SELECT
            g.screen_id,
            rb.time_block_id,
            rb.day_type,
            sum(rb.avg_daily_ridership * w.stop_share) AS daily_impressions
        FROM v_screen_geography g
        JOIN v_route_stop_weight w   ON w.location_id = g.location_id
        JOIN v_route_block_demand rb ON rb.route_id = w.route_id
        WHERE g.inventory_class = 'fixed'
        GROUP BY 1, 2, 3

        UNION ALL

        -- Vehicle-mounted: one vehicle's share of its corridor.
        SELECT
            g.screen_id,
            cb.time_block_id,
            cb.day_type,
            cb.avg_daily_ridership AS daily_impressions
        FROM v_screen_geography g
        JOIN v_corridor_block_demand cb ON cb.corridor_id = g.corridor_id
        WHERE g.inventory_class = 'mobile'
        """
    )

    _create_location_site_view(con)

    # Screen-level demographics and context. `pool_key` is the physical-audience unit:
    # screens at one SITE see the same passersby, screens on one corridor see the same
    # riders. Anything summing audience across screens must group by it first.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_screen_profile AS
        SELECT
            g.screen_id,
            g.city_id,
            g.screen_type,
            g.screen_size,
            g.position,
            g.inventory_class,
            g.location_id,
            g.zone_id,
            g.corridor_id,
            g.location_type,
            l.name                                      AS location_name,
            coalesce(site.site_key, g.corridor_id)      AS pool_key,
            -- How many partitions the pool's audience was divided into to produce this
            -- screen's figure. 1 for stop-mounted screens: every screen at a location is
            -- passed by the same crowd, so the per-screen figure IS the pool's. For
            -- vehicle-mounted screens `v_screen_demand_history` divides the corridor by the
            -- vehicles working it, so the pool's whole crowd is per-screen x this count.
            -- The optimizer needs the pool total, not one vehicle's share, or it
            -- under-buys in-vehicle inventory by exactly this factor.
            CASE WHEN g.inventory_class = 'mobile'
                 THEN coalesce(cv.n_vehicles, 1)
                 ELSE 1 END                            AS pool_partition_count,
            l.city_zone,
            z.zone_name,
            z.resident_population,
            z.population_density_per_sqkm,
            z.median_age,
            z.pct_age_18_34,
            z.pct_age_35_54,
            z.median_household_income,
            z.income_index,
            z.pct_bachelor_or_higher,
            z.dominant_occupation,
            z.daytime_population_multiplier,
            coalesce(p.num_nearby_pois, 0)              AS num_nearby_pois,
            coalesce(p.weighted_nearby_footfall, 0.0)   AS weighted_nearby_footfall,
            p.closest_poi_distance_km,
            coalesce(p.nearby_poi_types, [])            AS nearby_poi_types
        FROM v_screen_geography g
        LEFT JOIN locations l        ON l.location_id = g.location_id
        LEFT JOIN zone_demographics z ON z.zone_id = g.zone_id
        LEFT JOIN v_screen_poi p      ON p.screen_id = g.screen_id
        LEFT JOIN v_corridor_vehicle_count cv ON cv.corridor_id = g.corridor_id
        LEFT JOIN v_location_site site ON site.location_id = g.location_id
        """
    )


def _create_location_site_view(con: duckdb.DuckDBPyConnection) -> None:
    """`location_id` -> `site_key`, the real physical-audience unit for fixed screens.

    THE BUG THIS FIXES. `pool_key` used to be the raw `location_id`, and one physical
    station is modelled as SEVERAL location rows — 910 stop-mounted `location_id`s resolve
    to 878 distinct sites. Screens on opposite platforms of one station see the same
    people, so treating them as separate pools let reach count that crowd twice. Merging
    makes reported reach slightly SMALLER and more honest.

    Grouped on `(city_id, name, the set of corridors serving it)`. All three keys are
    load-bearing, measured on this inventory:

        910  raw location_id                 -- under-merges: splits one station
        626  (city_id, name)                 -- OVER-merges by ~31%. Station names are a
                                                low-cardinality template and unrelated
                                                real stations coincidentally share one.
        878  (city_id, name, corridor set)   -- correct on both counts

    The corridor set is what separates two genuinely different stops that share a name:
    same name AND the same serving corridors is a platform pair, not a coincidence.

    `site_key` is a compact synthetic id rather than the composite string, because it is
    carried on every candidate row, grouped on in the solver and written to parquet.
    `dense_rank` over a deterministic ordering makes it stable across rebuilds. The human
    name travels separately as `location_name`, which is what a reason string should cite.
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW v_location_site AS
        WITH corridor_set AS (
            -- Sorted + deduplicated so two locations serving the same corridors produce
            -- byte-identical keys regardless of route_stops row order.
            SELECT location_id,
                   array_to_string(list_sort(list_distinct(list(corridor_id))), ',')
                       AS corridors
            FROM route_stops
            GROUP BY 1
        ),
        grouped AS (
            SELECT
                l.location_id,
                l.city_id,
                l.name,
                coalesce(c.corridors, '') AS corridors
            FROM locations l
            LEFT JOIN corridor_set c ON c.location_id = l.location_id
        )
        SELECT
            location_id,
            'SITE-' || city_id || '-' || lpad(CAST(
                dense_rank() OVER (
                    PARTITION BY city_id ORDER BY name, corridors
                ) AS VARCHAR), 4, '0') AS site_key
        FROM grouped
        """
    )


TERMINUS_WEIGHT = 1.5
"""Weight a route's first/last stop carries relative to a mid-route stop.

An ASSUMED constant, not a measured one: termini concentrate boardings and alightings, so
more of a route's riders pass through them. Nothing the validator checks may depend on an
assumed constant (CLAUDE.md 31), and nothing does — this only redistributes a route's
riders between its own stops. Set it to 1.0 and every stop shares equally; the corridor
total is identical either way, because `stop_share` renormalizes.
"""


def _create_route_stop_weight_view(con: duckdb.DuckDBPyConnection) -> None:
    """Each stop's share of its route's riders. Shares sum to exactly 1.0 per route.

    THE BUG THIS FIXES. The fixed-screen branch of `v_screen_demand_history` used to sum
    each serving route's WHOLE ridership at every stop on it. A rider travelling one line
    was therefore counted once at every stop they passed, so summing the modelled audience
    of the stops along one corridor came to **20.4x (median) that corridor's measured
    ridership** — a rider cannot be 20 different people. Bus routes average 12.58 stops and
    metro routes 13.86, which is the scale of the over-count; do not hard-code a divisor,
    because it varies per route and `stop_share` derives it.

    A stop's share is its weight over its route's total weight, so a route's riders are
    distributed across its stops and never multiplied by them. Multiple routes serving one
    stop still SUM — that is genuinely more people passing the screen.

    Termini are weighted `TERMINUS_WEIGHT`. If `is_first_stop`/`is_last_stop` ever stop
    being clean booleans, every stop falls back to equal weight rather than failing: the
    weighting is a refinement, the renormalization is the correction. Both columns are
    clean `BOOLEAN` on all 2,436 rows today, so the fallback is defensive only.
    """
    terminus_expr = "CASE WHEN is_first_stop OR is_last_stop THEN {w} ELSE 1.0 END"
    if _boolean_columns(con, "route_stops", ("is_first_stop", "is_last_stop")):
        weight = terminus_expr.format(w=TERMINUS_WEIGHT)
    else:
        from app.logging_utils import info as _info

        _info(
            "route_stops.is_first_stop/is_last_stop are missing or not BOOLEAN — falling "
            "back to equal stop weights. Corridor totals are unaffected; only the "
            "distribution between a route's own stops is."
        )
        weight = "1.0"

    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_route_stop_weight AS
        WITH weighted AS (
            SELECT route_id, location_id, {weight} AS weight
            FROM route_stops
        ),
        -- A route that visits one stop twice (a loop) legitimately passes it twice, so
        -- the weights are summed per (route, location) rather than deduplicated.
        per_stop AS (
            SELECT route_id, location_id, sum(weight) AS weight
            FROM weighted
            GROUP BY 1, 2
        )
        SELECT
            route_id,
            location_id,
            weight / sum(weight) OVER (PARTITION BY route_id) AS stop_share
        FROM per_stop
        """
    )


def _create_poi_view(con: duckdb.DuckDBPyConnection) -> None:
    """POI context per screen. Footfall is inverse-distance weighted, 1/(km + 0.1)."""
    con.execute(
        """
        CREATE OR REPLACE VIEW v_screen_poi AS
        SELECT
            g.screen_id,
            count(p.poi_id)                                       AS num_nearby_pois,
            sum(p.est_daily_footfall / (p.distance_to_location_km + 0.1))
                                                                  AS weighted_nearby_footfall,
            min(p.distance_to_location_km)                        AS closest_poi_distance_km,
            list_distinct(list(p.poi_type))                       AS nearby_poi_types
        FROM v_screen_geography g
        JOIN points_of_interest p ON p.anchor_location_id = g.location_id
        GROUP BY 1
        """
    )


def query_df(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run read-only SQL and return a DataFrame. Never hand the result to an LLM directly."""
    con = get_connection()
    rel = con.execute(sql, params) if params else con.execute(sql)
    return rel.fetch_df()


def table_schema(table: str) -> pd.DataFrame:
    return query_df(f"DESCRIBE {table}")


def available_tables() -> list[str]:
    df = query_df("SELECT table_name FROM information_schema.tables ORDER BY table_name")
    return df["table_name"].tolist()


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Existence check against an explicit connection.

    Used during view creation, which runs inside `get_connection` while `_lock` is held —
    going through `query_df` there would re-enter `get_connection` and deadlock.
    """
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1", [name]
    ).fetchall()
    return bool(rows)


def _boolean_columns(con: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]) -> bool:
    """True when every named column exists on `table` AND is typed BOOLEAN.

    Same explicit-connection reason as `_has_table`: this runs during view creation, while
    `_lock` is held.
    """
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND column_name IN ? AND data_type = 'BOOLEAN'",
        [table, list(columns)],
    ).fetchall()
    return len(rows) == len(columns)


def has_ridership_actuals() -> bool:
    """False when the 124 MB file was not provisioned; callers must fall back."""
    return "ridership_actuals" in available_tables()

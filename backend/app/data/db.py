"""DuckDB access over the source CSVs.

Scope note: this module exists so the Master Agent can verify specialist output against
real data, and so the specialist stubs can return real screen IDs. The full analytical
view layer (v_screen_demand_history, v_historical_pricing, ...) belongs to the Data
Intelligence Agent and its owner — add views here rather than reading CSVs elsewhere.
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


def has_ridership_actuals() -> bool:
    """False when the 124 MB file was not provisioned; callers must fall back."""
    return "ridership_actuals" in available_tables()

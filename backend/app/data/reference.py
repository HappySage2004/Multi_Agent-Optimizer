"""Cached reference lookups used for hard-constraint checks and geography resolution.

These are the facts the Master Agent trusts. It checks specialist output against them
rather than taking a subagent's word for what exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from app.data.db import query_df


@dataclass(frozen=True)
class ScreenFacts:
    screen_id: str
    city_id: str
    zone_id: str | None
    corridor_id: str | None
    screen_type: str
    screen_size: str | None
    inventory_class: str
    # Human-readable labels. Carried here so anything shown to a client can name a place
    # the way they would — "Grant Rd & Kingsley Rd", not "LH-ZONE-005". Every location, all
    # 30 zones and all 94 corridors have a name, so a null means the screen genuinely has no
    # location or no zone (every vehicle-mounted screen) rather than a missing label.
    location_name: str | None = None
    zone_name: str | None = None
    corridor_name: str | None = None

    @property
    def place_label(self) -> str:
        """Where this screen is, in the words a CLIENT would use.

        The stop or station name comes FIRST — "Grant Rd & Kingsley Rd", "East Commons
        Station" — because that is the thing an advertiser can picture standing in front of.
        A zone is a planning unit and a zone id means nothing outside this codebase.
        Vehicle-mounted screens have no fixed location, so their route names them. The ids
        are last and exist only so this never renders an empty cell.
        """
        return (
            self.location_name
            or self.corridor_name
            or self.zone_name
            or self.zone_id
            or self.corridor_id
            or self.city_id
        )

    @property
    def screen_type_label(self) -> str:
        """`metro_rail_coach` -> `Metro Rail Coach`. Snake case is an engineering artifact."""
        return self.screen_type.replace("_", " ").title()


@dataclass
class GeographyIndex:
    city_ids: set[str] = field(default_factory=set)
    zone_ids: set[str] = field(default_factory=set)
    corridor_ids: set[str] = field(default_factory=set)
    city_name_to_id: dict[str, str] = field(default_factory=dict)
    zone_name_to_id: dict[str, str] = field(default_factory=dict)
    zone_to_city: dict[str, str] = field(default_factory=dict)


@lru_cache(maxsize=1)
def geography_index() -> GeographyIndex:
    idx = GeographyIndex()

    cities = query_df("SELECT city_id, city_name FROM cities")
    idx.city_ids = set(cities["city_id"])
    idx.city_name_to_id = {n.strip().lower(): c for c, n in cities.itertuples(index=False)}

    zones = query_df("SELECT zone_id, city_id, zone_name FROM zone_demographics")
    idx.zone_ids = set(zones["zone_id"])
    for zone_id, city_id, zone_name in zones.itertuples(index=False):
        idx.zone_name_to_id[zone_name.strip().lower()] = zone_id
        idx.zone_to_city[zone_id] = city_id

    idx.corridor_ids = set(query_df("SELECT DISTINCT corridor_id FROM route_stops")["corridor_id"])
    return idx


@lru_cache(maxsize=1)
def screen_facts() -> dict[str, ScreenFacts]:
    """All 11,163 screens keyed by id. ~1 MB resident; loaded once per process.

    The name joins are LEFT joins on purpose. A vehicle-mounted screen has no zone at all,
    so `zone_name` being null there is the correct answer rather than a gap to fill; the
    route name is what names its geography.
    """
    df = query_df(
        """
        SELECT g.screen_id, g.city_id, g.zone_id, g.corridor_id, g.screen_type,
               g.screen_size, g.inventory_class,
               l.name AS location_name,
               z.zone_name,
               r.route_name AS corridor_name
        FROM v_screen_geography g
        LEFT JOIN locations l         ON l.location_id = g.location_id
        LEFT JOIN zone_demographics z ON z.zone_id = g.zone_id
        LEFT JOIN (
            -- Exactly one route_name per corridor across all 94 of them, verified, so any
            -- aggregate is the name rather than an arbitrary pick from several.
            SELECT corridor_id, min(route_name) AS route_name
            FROM route_stops
            GROUP BY corridor_id
        ) r ON r.corridor_id = g.corridor_id
        """
    )
    return {
        row.screen_id: ScreenFacts(
            screen_id=row.screen_id,
            city_id=row.city_id,
            zone_id=None if _isnull(row.zone_id) else row.zone_id,
            corridor_id=None if _isnull(row.corridor_id) else row.corridor_id,
            screen_type=row.screen_type,
            screen_size=None if _isnull(row.screen_size) else row.screen_size,
            inventory_class=row.inventory_class,
            location_name=None if _isnull(row.location_name) else row.location_name,
            zone_name=None if _isnull(row.zone_name) else row.zone_name,
            corridor_name=None if _isnull(row.corridor_name) else row.corridor_name,
        )
        for row in df.itertuples(index=False)
    }


@lru_cache(maxsize=1)
def time_block_labels() -> dict[str, str]:
    """time_block_id -> "16:00-20:00 (Evening)".

    A rep reads a clock time and a daypart; "Block 5" on its own means nothing outside this
    codebase. Both halves come from `dim_slot`, so neither is invented here.
    """
    df = query_df("SELECT time_block_id, time_block_label, nearest_daypart FROM dim_slot")
    return {
        str(row.time_block_id): f"{row.time_block_label} ({str(row.nearest_daypart).title()})"
        for row in df.itertuples(index=False)
    }


@lru_cache(maxsize=1)
def time_block_ids() -> set[str]:
    df = query_df("SELECT time_block_id FROM dim_slot")
    return {str(v) for v in df["time_block_id"]}


@lru_cache(maxsize=1)
def corridor_zones() -> dict[str, set[str]]:
    """corridor_id -> zones it passes through. Lets mobile inventory be zone-checked."""
    df = query_df("SELECT corridor_id, zone_id FROM v_corridor_zones")
    out: dict[str, set[str]] = {}
    for corridor_id, zone_id in df.itertuples(index=False):
        out.setdefault(corridor_id, set()).add(zone_id)
    return out


def _isnull(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def resolve_geography(terms: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Map free-text or ID-like geography terms onto real IDs.

    Returns `({'city_ids': [...], 'zone_ids': [...], 'corridor_ids': [...]}, unresolved)`.
    Deliberately conservative: an unmatched term is reported, never guessed at.
    """
    idx = geography_index()
    resolved: dict[str, list[str]] = {"city_ids": [], "zone_ids": [], "corridor_ids": []}
    unresolved: list[str] = []

    for raw in terms:
        term = raw.strip()
        key = term.lower()
        if term in idx.city_ids:
            resolved["city_ids"].append(term)
        elif term in idx.zone_ids:
            resolved["zone_ids"].append(term)
        elif term in idx.corridor_ids:
            resolved["corridor_ids"].append(term)
        elif key in idx.city_name_to_id:
            resolved["city_ids"].append(idx.city_name_to_id[key])
        elif key in idx.zone_name_to_id:
            resolved["zone_ids"].append(idx.zone_name_to_id[key])
        else:
            unresolved.append(term)

    for bucket in resolved.values():
        bucket[:] = sorted(dict.fromkeys(bucket))
    return resolved, unresolved


def eligible_screen_ids(
    city_ids: list[str], zone_ids: list[str], corridor_ids: list[str]
) -> set[str]:
    """Screens inside the requested geography. Mobile screens qualify via their corridor."""
    facts = screen_facts()
    czones = corridor_zones()
    wanted_cities, wanted_zones, wanted_corridors = set(city_ids), set(zone_ids), set(corridor_ids)

    out: set[str] = set()
    for sid, f in facts.items():
        if wanted_corridors and f.corridor_id in wanted_corridors:
            out.add(sid)
            continue
        if wanted_zones:
            if f.zone_id in wanted_zones:
                out.add(sid)
                continue
            # Mobile screen whose corridor touches a requested zone.
            if f.corridor_id and czones.get(f.corridor_id, set()) & wanted_zones:
                out.add(sid)
                continue
        # A city filter alone qualifies the whole city, but only when no finer
        # geography was requested — otherwise the narrower filter must win.
        if (
            wanted_cities
            and not wanted_zones
            and not wanted_corridors
            and f.city_id in wanted_cities
        ):
            out.add(sid)
    return out

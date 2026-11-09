#!/usr/bin/env python3
"""Print nonstop flights between two airports on a given date as a pandas DataFrame.

With a return day, prints nonstop round-trip pairs (outbound + return, one total price).

Usage:
    python nonstop_search.py JFK LAX 0          # one-way, today
    python nonstop_search.py JFK LAX 30         # one-way, 30 days from today
    python nonstop_search.py JFK LAX 30 37      # round trip: leave in 30 days, return in 37
    python nonstop_search.py --file searches.xlsx            # run every row in a spreadsheet
    python nonstop_search.py --file searches.xlsx --out-dir results   # also save Parquet files

Spreadsheet columns: origin, destination, days_ahead, return_days (blank = one-way).

Requires: pip install flights pandas openpyxl (pyarrow too, for --out-dir)
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from fli.models import (
    Airport,
    FlightSearchFilters,
    FlightSegment,
    MaxStops,
    PassengerInfo,
    SeatType,
    SortBy,
    TripType,
)
from fli.search import SearchFlights

LOCAL_TZ = ZoneInfo("America/New_York")


def ensure_airport(code: str) -> None:
    """Add an IATA code missing from fli's Airport enum (runtime patch)."""
    if code in Airport.__members__:
        return
    member = object.__new__(Airport)
    member._name_ = code
    member._value_ = code
    Airport._member_map_[code] = member
    Airport._value2member_map_[code] = member
    Airport._member_names_.append(code)


ensure_airport("DJT")


AIRPORT_GROUPS = {
    "NYC": ["EWR", "JFK", "LGA"],
    "TYO": ["NRT", "HND"],
    "PBI": ["DJT"],
    "LON": ["LHR", "LGW", "STN", "LUT", "LCY"],
    "TOR": ["YYZ", "YTZ"],
}


def parse_airports(arg: str) -> list[Airport]:
    """Parse 'EWR,JFK', 'NYC', 'HND', or any mix into a list of Airport values."""
    airports = []
    for code in arg.split(","):
        code = code.strip().upper()
        for c in AIRPORT_GROUPS.get(code, [code]):
            try:
                airports.append(Airport[c])
            except KeyError:
                sys.exit(f"Unknown airport code: {c}")
    return list(dict.fromkeys(airports))  # remove duplicates, keep order


def days_from_today(days: int) -> date:
    """Date `days` days from today in New York time: 0 = today, 1 = tomorrow."""
    if days < 0:
        sys.exit("Days ahead must be 0 or more.")
    return datetime.now(LOCAL_TZ).date() + timedelta(days=days)


def build_one_way_filters(origins: list[Airport], destinations: list[Airport],
                          travel_date: date) -> FlightSearchFilters:
    """Search filters for one adult, economy, nonstop, one-way trip."""
    return FlightSearchFilters(
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[[a, 0] for a in origins],
                arrival_airport=[[a, 0] for a in destinations],
                travel_date=travel_date.strftime("%Y-%m-%d"),
            )
        ],
        seat_type=SeatType.ECONOMY,
        stops=MaxStops.NON_STOP,
        sort_by=SortBy.DEPARTURE_TIME,
    )


def build_round_trip_filters(origins: list[Airport], destinations: list[Airport],
                             depart_date: date, return_date: date) -> FlightSearchFilters:
    """Search filters for one adult, economy, nonstop, round-trip search."""
    return FlightSearchFilters(
        trip_type=TripType.ROUND_TRIP,
        passenger_info=PassengerInfo(adults=1),
        flight_segments=[
            FlightSegment(
                departure_airport=[[a, 0] for a in origins],
                arrival_airport=[[a, 0] for a in destinations],
                travel_date=depart_date.strftime("%Y-%m-%d"),
            ),
            FlightSegment(
                departure_airport=[[a, 0] for a in destinations],
                arrival_airport=[[a, 0] for a in origins],
                travel_date=return_date.strftime("%Y-%m-%d"),
            ),
        ],
        seat_type=SeatType.ECONOMY,
        stops=MaxStops.NON_STOP,
        # top_n expands the first N outbounds in this order, so CHEAPEST
        # means the 10 cheapest outbounds get their return legs searched.
        sort_by=SortBy.CHEAPEST,
    )


def filter_nonstops(results, travel_date: date):
    """Keep only single-leg flights that depart on the requested date, sorted by departure time."""
    nonstops = [
        f for f in results
        if len(f.legs) == 1 and f.legs[0].departure_datetime.date() == travel_date
    ]
    return sorted(nonstops, key=lambda f: f.legs[0].departure_datetime)


def filter_round_trips(pairs, depart_date: date, return_date: date):
    """Keep (outbound, return) pairs where both are nonstop and on their requested dates."""
    kept = [
        (out, ret) for out, ret in pairs
        if len(out.legs) == 1 and len(ret.legs) == 1
        and out.legs[0].departure_datetime.date() == depart_date
        and ret.legs[0].departure_datetime.date() == return_date
    ]
    # Cheapest round trip first; pairs with no price go last.
    return sorted(kept, key=lambda p: (p[0].price is None, p[0].price or 0))


def airline_code(leg) -> str:
    """Best-effort IATA code for the flight number prefix."""
    code = getattr(leg.airline, "name", "")
    return code if len(code) <= 2 else ""


def leg_fields(leg, lookup_time: datetime) -> dict:
    """Columns describing one flight leg."""
    return {
        "airline": getattr(leg.airline, "value", str(leg.airline)),
        "flight": f"{airline_code(leg)}{leg.flight_number}",
        "from": getattr(leg.departure_airport, "name", str(leg.departure_airport)),
        "to": getattr(leg.arrival_airport, "name", str(leg.arrival_airport)),
        "departure_time": leg.departure_datetime.strftime("%I:%M %p"),
        "departure_date": leg.departure_datetime.date(),
        "days_from_now": (leg.departure_datetime.date() - lookup_time.date()).days,
        "aircraft": leg.aircraft,  # may be None when Google doesn't publish it
    }


def one_way_to_dataframe(flights, lookup_time: datetime, direction: str) -> pd.DataFrame:
    rows = []
    for f in flights:
        rows.append({
            "direction": direction,
            "lookup_time": lookup_time,
            **leg_fields(f.legs[0], lookup_time),
            "duration_minutes": f.duration,
            "price": f.price,
        })
    return pd.DataFrame(rows, columns=[
        "direction", "lookup_time", "airline", "flight", "from", "to", "departure_time",
        "departure_date", "days_from_now", "aircraft", "duration_minutes", "price",
    ])


def round_trips_to_dataframe(pairs, lookup_time: datetime) -> pd.DataFrame:
    rows = []
    for out, ret in pairs:
        out_leg = out.legs[0]
        ret_leg = ret.legs[0]
        rows.append({
            "lookup_time": lookup_time,
            **{f"out_{k}": v for k, v in leg_fields(out_leg, lookup_time).items()},
            "out_duration_minutes": out.duration,
            **{f"ret_{k}": v for k, v in leg_fields(ret_leg, lookup_time).items()},
            "ret_duration_minutes": ret.duration,
            "total_price": out.price,  # the library reports the round-trip total on the outbound
        })
    return pd.DataFrame(rows, columns=[
        "lookup_time",
        "out_airline", "out_flight", "out_from", "out_to", "out_departure_time",
        "out_departure_date", "out_days_from_now", "out_aircraft", "out_duration_minutes",
        "ret_airline", "ret_flight", "ret_from", "ret_to", "ret_departure_time",
        "ret_departure_date", "ret_days_from_now", "ret_aircraft", "ret_duration_minutes",
        "total_price",
    ])


def run_search(origin: str, destination: str, days_ahead: int,
               return_days: int | None = None) -> pd.DataFrame:
    """Run one search and return the results as a DataFrame."""
    origins = parse_airports(origin)
    destinations = parse_airports(destination)
    depart_date = days_from_today(days_ahead)
    lookup_time = datetime.now(LOCAL_TZ)

    if return_days is None:
        filters = build_one_way_filters(origins, destinations, depart_date)
        flights = filter_nonstops(SearchFlights().search(filters) or [], depart_date)
        return one_way_to_dataframe(flights, lookup_time, "outbound")

    return_date = days_from_today(return_days)
    filters = build_round_trip_filters(origins, destinations, depart_date, return_date)
    # top_n=10 (the maximum) expands the 10 outbounds ranked first by sort_by.
    raw_pairs = SearchFlights().search(filters, top_n=10) or []
    pairs = filter_round_trips(raw_pairs, depart_date, return_date)
    return round_trips_to_dataframe(pairs, lookup_time)


def is_blank(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def read_searches(path: str) -> list[dict]:
    """Read searches from the first sheet of an .xlsx file. Blank return_days means one-way."""
    df = pd.read_excel(path, engine="openpyxl").dropna(how="all")
    searches = []
    for _, row in df.iterrows():
        searches.append({
            "origin": str(row["origin"]).strip(),
            "destination": str(row["destination"]).strip(),
            "days_ahead": int(row["days_ahead"]),
            "return_days": None if is_blank(row.get("return_days")) else int(row["return_days"]),
        })
    return searches


def main() -> None:
    parser = argparse.ArgumentParser(description="List nonstop flights on a date.")
    parser.add_argument("origin", nargs="?", help="Departure airport IATA code or city, e.g. JFK, NYC")
    parser.add_argument("destination", nargs="?", help="Arrival airport IATA code or city, e.g. LAX, LON")
    parser.add_argument("days_ahead", type=int, nargs="?",
                        help="Days from today: 0 = today, 1 = tomorrow, etc.")
    parser.add_argument("return_days", type=int, nargs="?", default=None,
                        help="Optional: days from today for the return flight (round trip)")
    parser.add_argument("--file", help="Run every search in an .xlsx file "
                                       "(columns: origin, destination, days_ahead, return_days)")
    parser.add_argument("--out-dir", help="Also save each result as a Parquet file in this folder")
    args = parser.parse_args()

    if args.file:
        searches = read_searches(args.file)
    else:
        if args.origin is None or args.destination is None or args.days_ahead is None:
            parser.error("give origin, destination and days_ahead, or use --file")
        searches = [{
            "origin": args.origin,
            "destination": args.destination,
            "days_ahead": args.days_ahead,
            "return_days": args.return_days,
        }]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for s in searches:
        label = f"{s['origin']} -> {s['destination']}, day {s['days_ahead']}"
        if s["return_days"] is not None:
            label += f", return day {s['return_days']}"

        try:
            df = run_search(s["origin"], s["destination"], s["days_ahead"], s["return_days"])
        except (SystemExit, Exception) as e:  # bad input exits via sys.exit; keep going
            print(f"{label}: ERROR: {e or repr(e)}\n")
            continue

        print(f"== {label} ({len(df)} rows) ==")
        print(df)
        print()

        if args.out_dir:
            stamp = datetime.now(LOCAL_TZ).strftime("%Y%m%dT%H%M%S")
            route = f"{s['origin']}_{s['destination']}".replace(",", "-").upper()
            suffix = f"d{s['days_ahead']}" + (
                f"_r{s['return_days']}" if s["return_days"] is not None else "")
            path = os.path.join(args.out_dir, f"{route}_{stamp}_{suffix}.parquet")
            df.to_parquet(path, index=False)  # requires pyarrow
            print(f"Saved {path}\n")


if __name__ == "__main__":
    main()
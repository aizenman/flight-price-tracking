#!/usr/bin/env python3
"""
Read all .parquet files in a directory, find the lowest price for each unique
(from, to, days_from_now) combination, and plot each (from, to) route as a line
showing that price as a percentage of the route's maximum lowest price.
Each route's maximum is annotated with its calendar date.

Usage:
    python price_analysis.py /path/to/parquet/dir

Requirements:
    pip install pandas pyarrow matplotlib
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

# Column names to look for, in order of preference.
# Round-trip files use out_* for the outbound leg and total_price for price.
COLUMN_CANDIDATES = {
    "from": ["from", "out_from"],
    "to": ["to", "out_to"],
    "days_from_now": ["days_from_now", "out_days_from_now"],
    "price": ["price", "total_price"],
}

# Airport codes -> city codes. Any code not listed is left as-is.
CITY_MAP = {
    "LGA": "NYC",
    "EWR": "NYC",
    "JFK": "NYC",
    "YYZ": "TOR",
    "YTZ": "TOR",
}


def resolve_columns(schema_names: list[str]) -> dict[str, str]:
    """Return {internal_name: actual_column_name} for the first match of each."""
    resolved = {}
    for internal, candidates in COLUMN_CANDIDATES.items():
        match = next((c for c in candidates if c in schema_names), None)
        if match is None:
            raise SystemExit(
                f"Could not find a column for '{internal}'. "
                f"Looked for {candidates}; file has {schema_names}"
            )
        resolved[internal] = match
    return resolved


def load_all_parquet(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found in {directory}")

    frames = []
    for f in files:
        schema_names = pq.read_schema(f).names
        cols = resolve_columns(schema_names)
        df = pd.read_parquet(f, columns=list(cols.values()))
        # Rename to the internal names so the rest of the script is uniform.
        df = df.rename(columns={actual: internal for internal, actual in cols.items()})
        frames.append(df)
        print(f"Loaded {len(df):>8,} rows from {f.name}")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <directory>")

    directory = Path(sys.argv[1]).expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    data = load_all_parquet(directory)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price", "days_from_now", "from", "to"])

    # Collapse airports into their city codes
    data["from"] = data["from"].replace(CITY_MAP)
    data["to"] = data["to"].replace(CITY_MAP)

    # Lowest price for each unique [from, to, days_from_now] combination
    lowest = (
        data.groupby(["from", "to", "days_from_now"], as_index=False)["price"]
        .min()
    )

    # Express each price as a % of the maximum (lowest) price for its route
    route_max = lowest.groupby(["from", "to"])["price"].transform("max")
    lowest["pct_of_max"] = lowest["price"] / route_max * 100

    today = date.today()

    fig, ax = plt.subplots(figsize=(12, 7))
    for (origin, dest), group in lowest.groupby(["from", "to"]):
        group = group.sort_values("days_from_now")
        line, = ax.plot(
            group["days_from_now"],
            group["pct_of_max"],
            marker=".",
            linewidth=1.2,
            label=f"{origin} → {dest}",
        )

        # Annotate the route's maximum with its calendar date
        peak = group.loc[group["pct_of_max"].idxmax()]
        peak_day = int(peak["days_from_now"])
        peak_date = today + timedelta(days=peak_day)
        ax.annotate(
            f"{peak_date:%b %d, %Y}",
            xy=(peak_day, peak["pct_of_max"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=line.get_color(),
            arrowprops=dict(arrowstyle="-", color=line.get_color(), lw=0.6),
        )

    ax.set_xlabel("Days from now")
    ax.set_ylabel("% of maximum lowest price")
    ax.set_title("Lowest price by days from now, as % of each route's maximum")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, alpha=0.3)

    if lowest[["from", "to"]].drop_duplicates().shape[0] <= 20:
        ax.legend(loc="best", fontsize="small")
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="x-small", ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
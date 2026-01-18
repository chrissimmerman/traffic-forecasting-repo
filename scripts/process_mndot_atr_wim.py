#!/usr/bin/env python3
"""
Process MnDOT ATR/WIM hourly volume CSVs into station-day totals and directional split metrics.

Inputs
------
Either:
  1) a .zip containing one or more yearly CSVs (like 2017_wim_atr_volume-*.csv), OR
  2) a directory containing those CSVs.

Each CSV is expected to have columns:
  station_id, dir_of_travel, lane_of_travel, date, 1, 2, ..., 24

Outputs
-------
1) atr_station_day_<minyear>_<maxyear>.csv.gz
   Columns:
     station_id, date, station_volume,
     dir_a, dir_b, vol_a, vol_b, share_a, share_b,
     n_dirs, valid_day

2) station_candidates_<startyear>_<endyear>_coverage.csv
   Columns:
     station_id, years_present, min_cov, avg_cov, avg_volume

Example
-------
python process_mndot_atr_wim.py --input data.zip --outdir mndot_outputs
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


HOUR_COLS = [str(i) for i in range(1, 25)]


def _iter_csv_streams(input_path: Path) -> List[Tuple[str, io.BufferedReader]]:
    """Return list of (name, file-like-bytes) for CSVs inside a zip or directory."""
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(input_path)
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        members.sort()
        return [(m, zf.open(m)) for m in members]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.csv"))
        return [(f.name, open(f, "rb")) for f in files]
    raise ValueError(f"--input must be a .zip or a directory of CSVs: got {input_path}")


def _close_streams(streams: List[Tuple[str, io.BufferedReader]]) -> None:
    for _, s in streams:
        try:
            s.close()
        except Exception:
            pass


def _read_in_chunks(byte_stream, chunksize: int = 500_000) -> Iterable[pd.DataFrame]:
    """Stream-read a CSV byte stream in chunks."""
    yield from pd.read_csv(byte_stream, chunksize=chunksize, low_memory=False)


def _aggregate_dir_day(streams: List[Tuple[str, io.BufferedReader]], chunksize: int) -> pd.DataFrame:
    """
    Read all sources and aggregate to direction-day volumes:
      station_id, date, dir_of_travel -> vol_dir
    """
    dir_parts = []
    min_year = None
    max_year = None

    for name, stream in streams:
        for chunk in _read_in_chunks(stream, chunksize=chunksize):
            missing = {"station_id", "dir_of_travel", "lane_of_travel", "date"}.difference(chunk.columns)
            if missing:
                raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

            if chunk["date"].notna().any():
                y0 = int(chunk["date"].min().year)
                y1 = int(chunk["date"].max().year)
                min_year = y0 if min_year is None else min(min_year, y0)
                max_year = y1 if max_year is None else max(max_year, y1)

            hours = chunk[HOUR_COLS].apply(pd.to_numeric, errors="coerce")
            lane_total = hours.sum(axis=1, min_count=1)

            tmp = pd.DataFrame(
                {
                    "station_id": chunk["station_id"].astype("int64", errors="ignore"),
                    "dir_of_travel": chunk["dir_of_travel"].astype("int64", errors="ignore"),
                    "date": chunk["date"],
                    "lane_total": lane_total,
                }
            )
            tmp = tmp[tmp["date"].notna()]

            g = (
                tmp.groupby(["station_id", "date", "dir_of_travel"], as_index=False)["lane_total"]
                .sum()
                .rename(columns={"lane_total": "vol_dir"})
            )
            dir_parts.append(g)

    if not dir_parts:
        raise ValueError("No data read from input.")

    dir_day = pd.concat(dir_parts, ignore_index=True)
    dir_day = dir_day.groupby(["station_id", "date", "dir_of_travel"], as_index=False)["vol_dir"].sum()
    dir_day.attrs["min_year"] = min_year
    dir_day.attrs["max_year"] = max_year
    return dir_day


def _pick_top_two_dirs(dir_day: pd.DataFrame) -> pd.DataFrame:
    """
    For each station, pick the two directions that appear most frequently (by count of dates).
    Returns: station_id, dir_a, dir_b, n_dirs
    """
    counts = (
        dir_day.assign(has_obs=1)
        .groupby(["station_id", "dir_of_travel"], as_index=False)["has_obs"]
        .sum()
        .rename(columns={"has_obs": "n_days_present"})
    )

    n_dirs = (
        counts.groupby("station_id", as_index=False)["dir_of_travel"]
        .nunique()
        .rename(columns={"dir_of_travel": "n_dirs"})
    )

    counts = counts.sort_values(["station_id", "n_days_present", "dir_of_travel"], ascending=[True, False, True])
    top2 = counts.groupby("station_id").head(2).copy()
    top2["rank"] = top2.groupby("station_id").cumcount() + 1

    pivot = top2.pivot(index="station_id", columns="rank", values="dir_of_travel").reset_index()
    pivot = pivot.rename(columns={1: "dir_a", 2: "dir_b"})

    return pivot.merge(n_dirs, on="station_id", how="left")


def _build_station_day(dir_day: pd.DataFrame) -> pd.DataFrame:
    """Build station-day totals + directional split fields."""
    station_day = (
        dir_day.groupby(["station_id", "date"], as_index=False)["vol_dir"].sum()
        .rename(columns={"vol_dir": "station_volume"})
    )

    station_dirs = _pick_top_two_dirs(dir_day)
    base = station_day.merge(station_dirs, on="station_id", how="left")

    dir_a_df = dir_day.rename(columns={"dir_of_travel": "dir_a", "vol_dir": "vol_a"})
    dir_b_df = dir_day.rename(columns={"dir_of_travel": "dir_b", "vol_dir": "vol_b"})

    base = base.merge(dir_a_df[["station_id", "date", "dir_a", "vol_a"]], on=["station_id", "date", "dir_a"], how="left")
    base = base.merge(dir_b_df[["station_id", "date", "dir_b", "vol_b"]], on=["station_id", "date", "dir_b"], how="left")

    denom = base["vol_a"].fillna(0) + base["vol_b"].fillna(0)
    base["share_a"] = np.where(denom > 0, base["vol_a"] / denom, np.nan)
    base["share_b"] = np.where(denom > 0, base["vol_b"] / denom, np.nan)

    base["valid_day"] = (
        (base["station_volume"] > 0)
        & (base["n_dirs"] == 2)
        & base["vol_a"].notna()
        & base["vol_b"].notna()
        & (denom > 0)
    )

    cols = [
        "station_id", "date", "station_volume",
        "dir_a", "dir_b", "vol_a", "vol_b",
        "share_a", "share_b", "n_dirs", "valid_day",
    ]
    return base[cols].sort_values(["station_id", "date"]).reset_index(drop=True)


def _expected_days_in_year(year: int) -> int:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    return (end - start).days + 1


def _build_coverage_table(station_day: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    """Build a compact station candidate table for the chosen coverage window."""
    df = station_day.copy()
    df["year"] = df["date"].dt.year
    df = df[(df["year"] >= year_start) & (df["year"] <= year_end)]

    by_year = (
        df.groupby(["station_id", "year"], as_index=False)
        .agg(
            valid_days=("valid_day", "sum"),
            avg_volume=("station_volume", "mean"),
        )
    )
    by_year["expected_days"] = by_year["year"].apply(_expected_days_in_year)
    by_year["cov"] = by_year["valid_days"] / by_year["expected_days"]

    station_cov = (
        by_year.groupby("station_id", as_index=False)
        .agg(
            years_present=("year", "nunique"),
            min_cov=("cov", "min"),
            avg_cov=("cov", "mean"),
            avg_volume=("avg_volume", "mean"),
        )
        .sort_values(["years_present", "min_cov", "avg_cov", "avg_volume"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    return station_cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to a .zip of CSVs OR a directory containing CSVs.")
    ap.add_argument("--outdir", type=str, required=True, help="Directory to write outputs.")
    ap.add_argument("--chunksize", type=int, default=500_000, help="CSV rows per chunk (default: 500k).")
    ap.add_argument("--coverage-year-start", type=int, default=2020, help="Coverage window start year (default: 2020).")
    ap.add_argument("--coverage-year-end", type=int, default=2024, help="Coverage window end year (default: 2024).")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    streams = _iter_csv_streams(input_path)
    try:
        dir_day = _aggregate_dir_day(streams, chunksize=args.chunksize)
    finally:
        _close_streams(streams)

    station_day = _build_station_day(dir_day)

    min_year = int(dir_day.attrs.get("min_year", station_day["date"].dt.year.min()))
    max_year = int(dir_day.attrs.get("max_year", station_day["date"].dt.year.max()))
    station_out = outdir / f"atr_station_day_{min_year}_{max_year}.csv"
    station_day.to_csv(station_out, index=False)



    cov = _build_coverage_table(
        station_day,
        year_start=args.coverage_year_start,
        year_end=args.coverage_year_end,
    )
    cov_out = outdir / f"station_candidates_{args.coverage_year_start}_{args.coverage_year_end}_coverage.csv"
    cov.to_csv(cov_out, index=False)

    print(f"Wrote:\n  {station_out}\n  {cov_out}")


if __name__ == "__main__":
    main()

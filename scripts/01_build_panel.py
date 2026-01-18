#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_forecasting.config import ProjectConfig
from traffic_forecasting.panel import load_station_day, select_eligible_stations, build_panel_model_table


def main() -> None:
    ap = argparse.ArgumentParser(description='Build station-day panel with lag/rolling features and time splits.')
    ap.add_argument('--station-day', type=str, default=str(ProjectConfig().station_day_path), help='Input station-day CSV from processing step.')
    ap.add_argument('--out', type=str, default=str(ProjectConfig().panel_out_path), help='Output .csv.gz path for model panel.')
    ap.add_argument('--window-start', type=str, default=str(ProjectConfig().window_start.date()))
    ap.add_argument('--window-end', type=str, default=str(ProjectConfig().window_end.date()))
    ap.add_argument('--cov-th', type=float, default=ProjectConfig().coverage_threshold)
    ap.add_argument('--min-valid-days', type=int, default=ProjectConfig().min_valid_days)
    ap.add_argument('--train-end', type=str, default=str(ProjectConfig().train_end.date()))
    ap.add_argument('--val-end', type=str, default=str(ProjectConfig().val_end.date()))
    args = ap.parse_args()

    cfg = ProjectConfig(
        station_day_path=Path(args.station_day),
        panel_out_path=Path(args.out),
        window_start=pd.Timestamp(args.window_start),
        window_end=pd.Timestamp(args.window_end),
        coverage_threshold=args.cov_th,
        min_valid_days=args.min_valid_days,
        train_end=pd.Timestamp(args.train_end),
        val_end=pd.Timestamp(args.val_end),
    )

    df = load_station_day(cfg.station_day_path)

    eligible_ids, stats = select_eligible_stations(
        df,
        window_start=cfg.window_start,
        window_end=cfg.window_end,
        coverage_threshold=cfg.coverage_threshold,
        min_valid_days=cfg.min_valid_days,
        require_valid_day=True,
    )

    print(f'Eligible stations: {len(eligible_ids)}')

    panel = build_panel_model_table(
        df_station_day=df,
        eligible_ids=eligible_ids,
        window_start=cfg.window_start,
        window_end=cfg.window_end,
        lag_days=cfg.lag_days,
        roll_windows=cfg.roll_windows,
        train_end=cfg.train_end,
        val_end=cfg.val_end,
    )

    cfg.panel_out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cfg.panel_out_path, index=False, compression='gzip')

    stats_out = cfg.panel_out_path.with_suffix('').with_suffix('')
    stats_csv = cfg.panel_out_path.parent / 'eligible_station_stats.csv'
    stats.to_csv(stats_csv, index=False)

    print(f'Wrote panel: {cfg.panel_out_path} ({panel.shape[0]:,} rows)')
    print(f'Wrote station stats: {stats_csv}')


if __name__ == '__main__':
    main()

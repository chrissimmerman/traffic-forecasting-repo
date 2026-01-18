#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_forecasting.backtest import rolling_backtest_extratrees, RollingBacktestConfig
from traffic_forecasting.modeling import default_features


def main() -> None:
    ap = argparse.ArgumentParser(description='Run rolling backtest (expanding window) using ExtraTrees.')
    ap.add_argument('--panel', type=str, required=True, help='Panel CSV (output of 01_build_panel.py).')
    ap.add_argument('--out', type=str, default='reports/backtest/rolling_backtest.csv')
    ap.add_argument('--horizon-days', type=int, default=28)
    ap.add_argument('--step-days', type=int, default=56)
    ap.add_argument('--min-train-days', type=int, default=365)
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    cfg = RollingBacktestConfig(
        horizon_days=args.horizon_days,
        step_days=args.step_days,
        min_train_days=args.min_train_days,
    )

    bt = rolling_backtest_extratrees(df_panel=df, features=default_features(), cfg=cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out, index=False)
    print(f'Wrote: {out} ({bt.shape[0]} folds)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_forecasting.config import ProjectConfig
from traffic_forecasting.modeling import default_features, evaluate_models


def main() -> None:
    ap = argparse.ArgumentParser(description='Run multi-model bakeoff on the panel and write a comparison CSV.')
    ap.add_argument('--panel', type=str, default=str(ProjectConfig().panel_out_path))
    ap.add_argument('--out', type=str, default='reports/models/model_comparison.csv')
    ap.add_argument('--sort-by', type=str, default='test_VOL_RMSE', help='Column to sort results by (ascending).')
    args = ap.parse_args()

    panel = pd.read_csv(args.panel)
    panel['date'] = pd.to_datetime(panel['date'], errors='coerce')

    features = default_features()
    required = features + ['station_id', 'y', 'split', 'station_volume', 'y_lag_1', 'y_lag_7', 'y_roll_mean_7']
    panel = panel.dropna(subset=[c for c in required if c in panel.columns]).copy()

    train = panel[panel['split'] == 'train'].copy()
    val = panel[panel['split'] == 'val'].copy()
    test = panel[panel['split'] == 'test'].copy()

    res = evaluate_models(train=train, val=val, test=test, features=features)

    if args.sort_by in res.columns:
        res = res.sort_values([args.sort_by], ascending=True, na_position='last')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()

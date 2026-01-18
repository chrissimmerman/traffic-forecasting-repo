#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_forecasting.config import ProjectConfig
from traffic_forecasting.modeling import default_features, make_preprocessor, build_model_zoo
from traffic_forecasting.prediction import fit_and_predict_val_test


def main() -> None:
    ap = argparse.ArgumentParser(description='Fit ExtraTrees (default best) and write tidy predictions for val+test.')
    ap.add_argument('--panel', type=str, default=str(ProjectConfig().panel_out_path))
    ap.add_argument('--out', type=str, default='reports/predictions/pred_val_test.csv.gz')
    ap.add_argument('--model', type=str, default='ExtraTrees_600_leaf1', help='Model name from model zoo.')
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    features = default_features()
    features2 = features + ['station_id']

    dfm = df.dropna(subset=features2 + ['y', 'y_lag_7', 'station_volume', 'split']).copy()

    train = dfm[dfm['split'] == 'train'].copy()
    val = dfm[dfm['split'] == 'val'].copy()
    test = dfm[dfm['split'] == 'test'].copy()

    # Build model zoo
    cat_cols = ['station_id']
    num_cols = [c for c in features if c not in cat_cols]
    pre = make_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    zoo = build_model_zoo(pre=pre)

    if args.model not in zoo:
        raise SystemExit(f"Unknown model '{args.model}'. Available: {sorted(zoo.keys())}")

    model = zoo[args.model]
    pred = fit_and_predict_val_test(model=model, train=train, val=val, test=test, features=features, target='y', baseline_col='y_lag_7')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False, compression='gzip')
    print(f'Wrote: {out} ({pred.shape[0]:,} rows)')


if __name__ == '__main__':
    main()

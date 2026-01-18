from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .metrics import rmse, smape, inv_log1p


@dataclass(frozen=True)
class RollingBacktestConfig:
    horizon_days: int = 28
    step_days: int = 56
    min_train_days: int = 365
    baseline_col: str = 'y_lag_7'
    target: str = 'y'
    seed: int = 7
    n_jobs: int = -1
    n_estimators: int = 600
    min_samples_leaf: int = 1


def rolling_backtest_extratrees(
    df_panel: pd.DataFrame,
    features: List[str],
    cfg: RollingBacktestConfig = RollingBacktestConfig(),
) -> pd.DataFrame:
    """Expanding-window rolling backtest (station-day panel)."""
    required = ['station_id', 'date', cfg.target, cfg.baseline_col, 'station_volume']
    for c in required:
        if c not in df_panel.columns:
            raise ValueError(f'Missing required column: {c}')

    dfm = df_panel.dropna(subset=features + ['station_id', cfg.target, cfg.baseline_col, 'station_volume']).copy()
    dfm['date'] = pd.to_datetime(dfm['date'], errors='coerce')
    dfm = dfm.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    num_features = features
    cat_features = ['station_id']

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('sid', ohe, cat_features),
        ],
        remainder='drop',
    )

    d0 = dfm['date'].min().normalize()
    d1 = dfm['date'].max().normalize()

    first_test_start = (d0 + pd.Timedelta(days=cfg.min_train_days)).normalize()
    last_test_start = (d1 - pd.Timedelta(days=cfg.horizon_days - 1)).normalize()

    test_starts = []
    t = first_test_start
    while t <= last_test_start:
        test_starts.append(t)
        t = (t + pd.Timedelta(days=cfg.step_days)).normalize()

    rows = []

    for fold_i, test_start in enumerate(test_starts, start=1):
        test_end = test_start + pd.Timedelta(days=cfg.horizon_days - 1)

        train = dfm[dfm['date'] < test_start].copy()
        test = dfm[(dfm['date'] >= test_start) & (dfm['date'] <= test_end)].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train[num_features + cat_features]
        y_train = train[cfg.target].values
        X_test = test[num_features + cat_features]
        y_test = test[cfg.target].values

        model = ExtraTreesRegressor(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
        )

        pipe = Pipeline([('prep', preprocess), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_base = test[cfg.baseline_col].values

        # y-space
        y_smape_m = smape(y_test, y_pred)
        y_smape_b = smape(y_test, y_base)

        # volume-space
        vol_true = inv_log1p(y_test)
        vol_pred = inv_log1p(y_pred)
        vol_base = inv_log1p(y_base)

        vol_rmse_m = rmse(vol_true, vol_pred)
        vol_rmse_b = rmse(vol_true, vol_base)

        vol_smape_m = smape(vol_true, vol_pred)
        vol_smape_b = smape(vol_true, vol_base)

        rows.append({
            'fold': fold_i,
            'train_end': (test_start - pd.Timedelta(days=1)).date(),
            'test_start': test_start.date(),
            'test_end': test_end.date(),
            'rows_test': int(len(test)),
            'stations_test': int(test['station_id'].nunique()),

            'y_smape_base': y_smape_b,
            'y_smape_et': y_smape_m,
            'y_smape_improve_pp': y_smape_b - y_smape_m,

            'vol_smape_base': vol_smape_b,
            'vol_smape_et': vol_smape_m,
            'vol_smape_improve_pp': vol_smape_b - vol_smape_m,

            'vol_rmse_base': vol_rmse_b,
            'vol_rmse_et': vol_rmse_m,
            'vol_rmse_improve_pct': 100.0 * (vol_rmse_b - vol_rmse_m) / max(vol_rmse_b, 1e-9),
        })

    return pd.DataFrame(rows)

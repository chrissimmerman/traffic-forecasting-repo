from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .metrics import inv_log1p


def fit_and_predict_val_test(model, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str = 'y', baseline_col: str = 'y_lag_7') -> pd.DataFrame:
    """Fit on train and return tidy predictions for val+test (includes volume-scale columns)."""
    features2 = features + ['station_id']

    X_train = train[features2].copy()
    y_train = train[target].to_numpy()

    X_val = val[features2].copy()
    X_test = test[features2].copy()

    model.fit(X_train, y_train)

    val_pred_y = model.predict(X_val)
    test_pred_y = model.predict(X_test)

    val_base_y = val[baseline_col].to_numpy()
    test_base_y = test[baseline_col].to_numpy()

    pred_val = pd.DataFrame({
        'split': 'val',
        'date': pd.to_datetime(val['date']).to_numpy(),
        'station_id': val['station_id'].to_numpy(),
        'y_true': val[target].to_numpy(),
        'y_pred': val_pred_y,
        'y_base': val_base_y,
        'vol_true': val['station_volume'].to_numpy(dtype=float),
    })

    pred_test = pd.DataFrame({
        'split': 'test',
        'date': pd.to_datetime(test['date']).to_numpy(),
        'station_id': test['station_id'].to_numpy(),
        'y_true': test[target].to_numpy(),
        'y_pred': test_pred_y,
        'y_base': test_base_y,
        'vol_true': test['station_volume'].to_numpy(dtype=float),
    })

    pred = pd.concat([pred_val, pred_test], ignore_index=True)
    pred = pred.dropna(subset=['date', 'station_id', 'y_true', 'y_pred', 'y_base', 'vol_true']).copy()

    pred['vol_pred'] = inv_log1p(pred['y_pred'])
    pred['vol_base'] = inv_log1p(pred['y_base'])

    return pred

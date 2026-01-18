from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat, eps: float = 1e-9) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yhat) / denom)) * 100.0)


def smape(y, yhat, eps: float = 1e-9) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.maximum(np.abs(y) + np.abs(yhat), eps)
    return float(np.mean(2.0 * np.abs(yhat - y) / denom) * 100.0)


def inv_log1p(z):
    z = np.asarray(z, dtype=float)
    return np.clip(np.expm1(z), 0, None)


def score_block(tag: str, y, yhat) -> dict:
    """Metrics computed in the current scale of y/yhat."""
    return {
        f"{tag}_RMSE": rmse(y, yhat),
        f"{tag}_MAE": mae(y, yhat),
        f"{tag}_MAPE_%": mape(y, yhat),
        f"{tag}_sMAPE_%": smape(y, yhat),
    }


def score_block_vol(tag: str, y_log, yhat_log) -> dict:
    """Assumes y/yhat are log1p; reports metrics on original volume scale."""
    y = inv_log1p(y_log)
    yhat = inv_log1p(yhat_log)
    return {
        f"{tag}_VOL_RMSE": rmse(y, yhat),
        f"{tag}_VOL_MAE": mae(y, yhat),
        f"{tag}_VOL_MAPE_%": mape(y, yhat),
        f"{tag}_VOL_sMAPE_%": smape(y, yhat),
    }


def score_block_station_weighted(tag: str, df_split: pd.DataFrame, yhat, station_col: str = 'station_id', y_col: str = 'y') -> dict:
    """Equal-weight each station: compute metrics per-station, then average."""
    tmp = df_split[[station_col]].copy()
    tmp['y'] = np.asarray(df_split[y_col].values, dtype=float)
    tmp['yhat'] = np.asarray(yhat, dtype=float)

    per_station = []
    for sid, g in tmp.groupby(station_col):
        per_station.append({
            'station_id': sid,
            'RMSE': rmse(g['y'], g['yhat']),
            'MAE': mae(g['y'], g['yhat']),
            'MAPE_%': mape(g['y'], g['yhat']),
            'sMAPE_%': smape(g['y'], g['yhat']),
        })

    ps = pd.DataFrame(per_station)
    return {
        f"{tag}_STW_RMSE": float(ps['RMSE'].mean()),
        f"{tag}_STW_MAE": float(ps['MAE'].mean()),
        f"{tag}_STW_MAPE_%": float(ps['MAPE_%'].mean()),
        f"{tag}_STW_sMAPE_%": float(ps['sMAPE_%'].mean()),
        f"{tag}_STW_n_stations": int(ps.shape[0]),
    }


def score_block_station_weighted_vol(tag: str, df_split: pd.DataFrame, yhat_log, station_col: str = 'station_id', y_col: str = 'y') -> dict:
    """Station-weighted metrics on original volume scale (assumes log1p)."""
    tmp = df_split[[station_col]].copy()
    tmp['y_log'] = np.asarray(df_split[y_col].values, dtype=float)
    tmp['yhat_log'] = np.asarray(yhat_log, dtype=float)

    per_station = []
    for sid, g in tmp.groupby(station_col):
        y = inv_log1p(g['y_log'])
        yhat = inv_log1p(g['yhat_log'])
        per_station.append({
            'station_id': sid,
            'VOL_RMSE': rmse(y, yhat),
            'VOL_MAE': mae(y, yhat),
            'VOL_MAPE_%': mape(y, yhat),
            'VOL_sMAPE_%': smape(y, yhat),
        })

    ps = pd.DataFrame(per_station)
    return {
        f"{tag}_STW_VOL_RMSE": float(ps['VOL_RMSE'].mean()),
        f"{tag}_STW_VOL_MAE": float(ps['VOL_MAE'].mean()),
        f"{tag}_STW_VOL_MAPE_%": float(ps['VOL_MAPE_%'].mean()),
        f"{tag}_STW_VOL_sMAPE_%": float(ps['VOL_sMAPE_%'].mean()),
        f"{tag}_STW_n_stations": int(ps.shape[0]),
    }

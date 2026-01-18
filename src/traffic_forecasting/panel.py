from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_station_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['station_id', 'date']).copy()
    df['station_id'] = df['station_id'].astype(int)
    if 'valid_day' in df.columns:
        df['valid_day'] = df['valid_day'].astype(bool)
    return df.sort_values(['station_id', 'date']).reset_index(drop=True)


def select_eligible_stations(
    df_station_day: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    coverage_threshold: float = 0.90,
    min_valid_days: int = 730,
    require_valid_day: bool = True,
) -> Tuple[List[int], pd.DataFrame]:
    dfw = df_station_day[(df_station_day['date'] >= window_start) & (df_station_day['date'] <= window_end)].copy()

    if require_valid_day and 'valid_day' in dfw.columns:
        dfw = dfw[dfw['valid_day']].copy()

    expected_days = int((window_end - window_start).days) + 1

    stats = (
        dfw.groupby('station_id', as_index=False)
        .agg(
            valid_days=('date', 'nunique'),
            mean_volume=('station_volume', 'mean'),
            min_date=('date', 'min'),
            max_date=('date', 'max'),
        )
    )

    stats['coverage'] = stats['valid_days'] / expected_days
    stats['eligible_cov'] = stats['coverage'] >= coverage_threshold
    stats['eligible_days'] = stats['valid_days'] >= min_valid_days
    stats['eligible'] = stats['eligible_cov'] & stats['eligible_days']

    eligible_ids = stats.loc[stats['eligible'], 'station_id'].astype(int).tolist()
    return eligible_ids, stats.sort_values(['eligible', 'coverage', 'mean_volume'], ascending=[False, False, False]).reset_index(drop=True)


def build_panel_model_table(
    df_station_day: pd.DataFrame,
    eligible_ids: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    lag_days: List[int],
    roll_windows: List[int],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    """Reproduce the notebook's panel build: calendar + lags + rolling stats on log1p(volume)."""

    dfw = df_station_day[(df_station_day['date'] >= window_start) & (df_station_day['date'] <= window_end)].copy()

    # supervised rows only on valid days (matches notebook)
    if 'valid_day' in dfw.columns:
        dfw = dfw[dfw['valid_day']].copy()

    dfw = dfw[dfw['station_id'].isin(eligible_ids)].copy()

    idx = pd.date_range(window_start, window_end, freq='D')
    base = pd.MultiIndex.from_product([eligible_ids, idx], names=['station_id', 'date']).to_frame(index=False)

    base = base.merge(
        dfw[['station_id', 'date', 'station_volume']],
        on=['station_id', 'date'],
        how='left'
    )

    base['y'] = np.log1p(base['station_volume'])

    base['dow'] = base['date'].dt.dayofweek
    base['month'] = base['date'].dt.month
    base['year'] = base['date'].dt.year
    base['dayofyear'] = base['date'].dt.dayofyear
    base['sin_doy'] = np.sin(2 * np.pi * base['dayofyear'] / 365.25)
    base['cos_doy'] = np.cos(2 * np.pi * base['dayofyear'] / 365.25)

    base = base.sort_values(['station_id', 'date']).reset_index(drop=True)
    g = base.groupby('station_id', group_keys=False)

    for lag in lag_days:
        base[f'y_lag_{lag}'] = g['y'].shift(lag)

    for w in roll_windows:
        minp = max(2, w // 3)
        base[f'y_roll_mean_{w}'] = g['y'].apply(lambda s: s.shift(1).rolling(window=w, min_periods=minp).mean())
        base[f'y_roll_std_{w}'] = g['y'].apply(lambda s: s.shift(1).rolling(window=w, min_periods=minp).std())

    for lag in lag_days:
        base[f'has_y_lag_{lag}'] = base[f'y_lag_{lag}'].notna().astype(int)

    for w in roll_windows:
        base[f'has_y_roll_{w}'] = base[f'y_roll_mean_{w}'].notna().astype(int)

    model_rows = base[base['y'].notna() & base['y_lag_7'].notna()].copy()

    def label_split(d: pd.Timestamp) -> str:
        if d <= train_end:
            return 'train'
        if d <= val_end:
            return 'val'
        return 'test'

    model_rows['split'] = model_rows['date'].apply(label_split)

    return model_rows.reset_index(drop=True)

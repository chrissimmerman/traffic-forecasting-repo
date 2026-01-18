from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class ProjectConfig:
    # Paths (repo-relative by default)
    station_day_path: Path = Path('mndot_outputs/atr_station_day_2017_2024.csv')
    panel_out_path: Path = Path('data/processed/panel_model_table_2021-06_2024-12_cov0p90_strict.csv.gz')

    # Window and eligibility (post-COVID focus)
    window_start: pd.Timestamp = pd.Timestamp('2021-06-01')
    window_end: pd.Timestamp = pd.Timestamp('2024-12-31')
    coverage_threshold: float = 0.90
    min_valid_days: int = 730

    # Feature engineering
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 28])
    roll_windows: List[int] = field(default_factory=lambda: [7, 28])

    # Splits
    train_end: pd.Timestamp = pd.Timestamp('2023-06-30')
    val_end: pd.Timestamp = pd.Timestamp('2023-12-31')

    # Runtime
    seed: int = 7
    n_jobs: int = -1

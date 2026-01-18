from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import (
    score_block,
    score_block_station_weighted,
    score_block_station_weighted_vol,
    score_block_vol,
)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.status import Status


def default_features() -> List[str]:
    return [
        'dow', 'month', 'sin_doy', 'cos_doy',
        'y_lag_1', 'y_lag_7', 'y_lag_14', 'y_lag_28',
        'y_roll_mean_7', 'y_roll_std_7',
        'y_roll_mean_28', 'y_roll_std_28',
    ]


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ('cat', ohe, cat_cols),
            ('num', num_pipe, num_cols),
        ],
        sparse_threshold=0.0,
    )


def build_model_zoo(pre: ColumnTransformer, seed: int = 7, n_jobs: int = -1) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models['DummyMean'] = Pipeline([('pre', pre), ('model', DummyRegressor(strategy='mean'))])
    models['LinearRegression'] = Pipeline([('pre', pre), ('model', LinearRegression())])
    models['Ridge_a1'] = Pipeline([('pre', pre), ('model', Ridge(alpha=1.0, random_state=seed))])

    for a in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        models[f'Lasso_a{a:g}'] = Pipeline([('pre', pre), ('model', Lasso(alpha=a, max_iter=20000, random_state=seed))])
        models[f'ElasticNet_a{a:g}_l1r0.5'] = Pipeline([('pre', pre), ('model', ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000, random_state=seed))])

    models['HuberRobust'] = Pipeline([('pre', pre), ('model', HuberRegressor(max_iter=1000))])

    for k in [10, 25, 50, 100]:
        models[f'KNN_k{k}_dist'] = Pipeline([('pre', pre), ('model', KNeighborsRegressor(n_neighbors=k, weights='distance'))])
        models[f'KNN_k{k}_uniform'] = Pipeline([('pre', pre), ('model', KNeighborsRegressor(n_neighbors=k, weights='uniform'))])

    models['MLP_128x64'] = Pipeline([
        ('pre', pre),
        ('model', MLPRegressor(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        )),
    ])

    models['RandomForest_400_leaf2'] = Pipeline([('pre', pre), ('model', RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=n_jobs,
    ))])

    models['ExtraTrees_600_leaf1'] = Pipeline([('pre', pre), ('model', ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=n_jobs,
    ))])

    models['GradientBoosting_500_lr0.05_depth3_sub0.8'] = Pipeline([('pre', pre), ('model', GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    ))])

    models['HGB_d6_lr0.05_it400_leaf50_l2e-6'] = Pipeline([('pre', pre), ('model', HistGradientBoostingRegressor(
        loss='squared_error',
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=50,
        l2_regularization=1e-6,
        random_state=seed,
    ))])

    models['HGB_leaf20_lr0.05_it600_d6_l2e-3'] = Pipeline([('pre', pre), ('model', HistGradientBoostingRegressor(
        loss='squared_error',
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    ))])

    models['HGB_leaf20_lr0.05_it800_nodes63_l2e-3'] = Pipeline([('pre', pre), ('model', HistGradientBoostingRegressor(
        loss='squared_error',
        max_depth=None,
        max_leaf_nodes=63,
        learning_rate=0.05,
        max_iter=800,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    ))])

    # Optional XGBoost
    try:
        from xgboost import XGBRegressor  # type: ignore

        models['XGBoost_curr_1200_lr0.03_d8_sub0.9_col0.9'] = Pipeline([('pre', pre), ('model', XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=seed,
            n_jobs=n_jobs,
        ))])

        models['XGBoost_safe_1800_lr0.02_d6_child10_sub0.8_col0.8'] = Pipeline([('pre', pre), ('model', XGBRegressor(
            n_estimators=1800,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=seed,
            n_jobs=n_jobs,
        ))])

    except Exception:
        pass

    return models


def evaluate_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target: str = "y",
    baselines: Dict[str, str] | None = None,
    seed: int = 7,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Evaluate simple panel baselines + a model zoo on train/val/test splits.

    Uses Rich progress + per-model spinner (fit progress isn't available for most sklearn estimators,
    so we show "activity" while fitting and advance the bar when each model completes).
    """
    baselines = baselines or {
        "seasonal_naive_lag7": "y_lag_7",
        "naive_lag1": "y_lag_1",
        "rolling_mean_7": "y_roll_mean_7",
    }

    # --- feature setup ---
    cat_cols = ["station_id"]
    num_cols = [c for c in features if c not in cat_cols]
    features2 = features + cat_cols

    pre = make_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    models = build_model_zoo(pre=pre, seed=seed, n_jobs=n_jobs)

    # --- matrices ---
    X_train = train[features2].copy()
    X_val = val[features2].copy()
    X_test = test[features2].copy()

    for X in (X_train, X_val, X_test):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

    y_train = train[target].to_numpy()
    y_val = val[target].to_numpy()
    y_test = test[target].to_numpy()

    rows: List[dict] = []

    # --- rich UI ---
    console = Console()
    console.print(
        f"[dim]Train/Val/Test rows:[/dim] {len(train):,} / {len(val):,} / {len(test):,}"
    )
    console.print(f"[dim]Models:[/dim] {len(models)}  |  [dim]Baselines:[/dim] {len(baselines)}")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        # -----------------
        # Panel baselines
        # -----------------
        t_base = progress.add_task("Baselines", total=len(baselines))
        for bname, col in baselines.items():
            progress.update(t_base, advance=1)

            if col not in val.columns or col not in test.columns:
                rows.append({"model": f"BASELINE::{bname}", "notes": f"missing column: {col}"})
                continue

            val_pred = val[col].values
            test_pred = test[col].values

            rows.append(
                {
                    "model": f"BASELINE::{bname}",
                    "fit_seconds": 0.0,
                    **score_block("val", y_val, val_pred),
                    **score_block("test", y_test, test_pred),
                    **score_block_vol("val", y_val, val_pred),
                    **score_block_vol("test", y_test, test_pred),
                    **score_block_station_weighted("val", val, val_pred),
                    **score_block_station_weighted("test", test, test_pred),
                    **score_block_station_weighted_vol("val", val, val_pred),
                    **score_block_station_weighted_vol("test", test, test_pred),
                    "notes": f"uses {col}",
                }
            )

        # -----------------
        # Learned models
        # -----------------
        t_models = progress.add_task("Models", total=len(models))
        for name, mdl in models.items():
            try:
                # show a spinner while the model is fitting
                with console.status(f"[bold]Fitting[/bold] {name} …", spinner="dots"):
                    t0 = time.perf_counter()
                    mdl.fit(X_train, y_train)  # FIT ONCE
                    fit_s = time.perf_counter() - t0

                    val_pred = mdl.predict(X_val)
                    test_pred = mdl.predict(X_test)

                rows.append(
                    {
                        "model": name,
                        "fit_seconds": fit_s,
                        **score_block("val", y_val, val_pred),
                        **score_block("test", y_test, test_pred),
                        **score_block_vol("val", y_val, val_pred),
                        **score_block_vol("test", y_test, test_pred),
                        **score_block_station_weighted("val", val, val_pred),
                        **score_block_station_weighted("test", test, test_pred),
                        **score_block_station_weighted_vol("val", val, val_pred),
                        **score_block_station_weighted_vol("test", test, test_pred),
                        "notes": "",
                    }
                )

                # Optional: one-line timing per model (doesn't break the progress display)
                console.print(f"[green]✓[/green] {name}  [dim]fit[/dim] {fit_s:.2f}s")

            except Exception as e:
                rows.append(
                    {
                        "model": name,
                        "fit_seconds": np.nan,
                        "notes": f"FAILED: {type(e).__name__}: {e}",
                    }
                )
                console.print(f"[red]✗[/red] {name}  {type(e).__name__}: {e}")

            progress.update(t_models, advance=1)

    return pd.DataFrame(rows)


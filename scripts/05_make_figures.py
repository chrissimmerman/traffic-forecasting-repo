#!/usr/bin/env python
"""Generate all figures from the notebook.

This fixes the common issue where "make_figures" only produces one output.

Expected inputs:
- station-day CSV (raw): mndot_outputs/atr_station_day_2017_2024.csv
- predictions CSV: reports/preds/pred_val_test.csv.gz
- rolling backtest CSV (optional): reports/backtest/rolling_backtest_results.csv

Outputs:
- assets/figures/fig01..fig08 as PNG + HTML
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Shared metrics
# -------------------------

def smape(y, yhat, eps=1e-9) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.maximum(np.abs(y) + np.abs(yhat), eps)
    return float(np.mean(2.0 * np.abs(yhat - y) / denom) * 100.0)


def _safe_write_image(fig, path: Path, **kwargs) -> bool:
    try:
        fig.write_image(path, **kwargs)
        return True
    except Exception as e:
        print(f"[warn] Could not write PNG {path.name}: {e}")
        return False


# -------------------------
# Figure 1
# -------------------------

def make_fig01(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    WINDOW_START = pd.Timestamp("2021-06-01")
    WINDOW_END   = pd.Timestamp("2024-12-31")
    expected_days_window = (WINDOW_END - WINDOW_START).days + 1

    # ----------------------------
    # Load + basic cleaning (match notebook)
    # ----------------------------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["station_id", "date"]).copy()

    df["station_id"] = df["station_id"].astype(int)
    df["valid_day"] = df["valid_day"].astype(bool)

    # ----------------------------
    # Station eligibility (post-COVID window)
    # ----------------------------
    df_win = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].copy()
    df_win_valid = df_win[df_win["valid_day"]].copy()

    station_stats = (
        df_win_valid.groupby("station_id", as_index=False)
        .agg(
            valid_days=("date", "nunique"),
            mean_volume=("station_volume", "mean"),
        )
    )
    station_stats["coverage"] = station_stats["valid_days"] / expected_days_window
    station_stats["eligible"] = (station_stats["coverage"] >= 0.90) & (station_stats["valid_days"] >= 730)

    eligible = station_stats.loc[station_stats["eligible"], "station_id"].tolist()

    # ----------------------------
    # 2019 baseline per station (median of valid days)
    # ----------------------------
    df_2019 = df[
        (df["date"] >= "2019-01-01")
        & (df["date"] <= "2019-12-31")
        & (df["valid_day"])
        & (df["station_id"].isin(eligible))
    ].copy()

    baseline_2019 = (
        df_2019.groupby("station_id")["station_volume"]
        .median()
        .rename("baseline_2019_med")
        .reset_index()
    )

    baseline_stations = set(baseline_2019["station_id"].tolist())

    # ----------------------------
    # Build ratio series (valid days only), focus on 2019 onward
    # ----------------------------
    tmp = df[
        (df["valid_day"])
        & (df["station_id"].isin(baseline_stations))
        & (df["date"] >= "2019-01-01")
    ].copy()

    tmp = tmp.merge(baseline_2019, on="station_id", how="inner")
    tmp["ratio_to_2019"] = tmp["station_volume"] / tmp["baseline_2019_med"]

    weekly = (
        tmp.set_index("date")
        .groupby(pd.Grouper(freq="W"))["ratio_to_2019"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .dropna(subset=["median", "q25", "q75"])
    )

    # Stability proxy: rolling std of median (13-week)
    weekly["median_roll_std_13w"] = weekly["median"].rolling(13, min_periods=8).std()
    pre = weekly[weekly["date"] < WINDOW_START]["median_roll_std_13w"].dropna()
    post = weekly[weekly["date"] >= WINDOW_START]["median_roll_std_13w"].dropna()

    stability_msg = None
    if len(pre) > 0 and len(post) > 0:
        change = 100.0 * (1 - post.median() / pre.median())
        stability_msg = f"Median volatility (13-week std) ↓ {change:.0f}% after Jun 2021"

    # ----------------------------
    # Plotly (match notebook)
    # ----------------------------
    fig = go.Figure()

    x_ribbon = pd.concat([weekly["date"], weekly["date"][::-1]])
    y_ribbon = pd.concat([weekly["q75"], weekly["q25"][::-1]])

    fig.add_trace(
        go.Scatter(
            x=x_ribbon,
            y=y_ribbon,
            fill="toself",
            name="IQR (25th–75th)",
            hoverinfo="skip",
            line=dict(width=0),
            opacity=0.25,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["date"],
            y=weekly["median"],
            mode="lines",
            name="Median station",
            line=dict(width=3),
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Median=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=1.0, line_dash="dash")
    fig.add_vline(x=WINDOW_START, line_dash="dash")

    annotations = []
    if stability_msg:
        # Match notebook placement, but avoid IndexError if weekly is short
        idx = 5 if len(weekly) > 5 else 0
        annotations.append(
            dict(
                x=weekly["date"].iloc[idx],
                y=float(weekly["q25"].min()) + 0.05,
                xref="x",
                yref="y",
                text=stability_msg,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=4,
            )
        )

    fig.update_layout(
        title=dict(
            text="Figure 1 — Traffic volumes normalized to 2019 baseline (system stabilizes post-COVID)",
            x=0.02,
            xanchor="left",
        ),
        xaxis_title="Date",
        yaxis_title="Daily volume ratio vs 2019 median",
        template="plotly_white",
        hovermode="x unified",
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(t=90),
    )

    fig.update_xaxes(dtick="M12", tickformat="%Y")

    png_out  = fig_dir / "fig01_relative_to_2019_baseline.png"
    html_out = fig_dir / "fig01_relative_to_2019_baseline.html"

    # Match notebook export behavior
    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None

    fig.write_image(png_out, width=1400, height=600, scale=2)
    fig.write_html(
        html_out,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
    )

    return [html_out, png_out]



# -------------------------
# Figure 2
# -------------------------
def make_fig02(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    # Match notebook context: Figure 2 is meant to reflect the *modeling window*
    # and typically the *eligible station set* (the df in-notebook was already filtered).
    WINDOW_START = pd.Timestamp("2021-06-01")
    WINDOW_END = pd.Timestamp("2024-12-31")
    COVID_START = pd.Timestamp("2020-03-01")

    expected_days_window = (WINDOW_END - WINDOW_START).days + 1

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["station_id", "date"]).copy()
    df["station_id"] = df["station_id"].astype(int)

    if "valid_day" in df.columns:
        df["valid_day"] = df["valid_day"].astype(bool)
    else:
        # If the column is missing, treat all rows as valid (best-effort)
        df["valid_day"] = True

    # ----------------------------
    # Station eligibility (same logic as notebook/fig01/fig03)
    # - within modeling window
    # - valid_day only
    # - >= 90% coverage in window
    # - >= 2 years of valid days (>= 730)
    # ----------------------------
    df_win = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].copy()
    df_win_valid = df_win[df_win["valid_day"]].copy()

    station_stats = (
        df_win_valid.groupby("station_id", as_index=False)
        .agg(
            valid_days=("date", "nunique"),
            mean_volume=("station_volume", "mean"),
        )
    )
    station_stats["coverage"] = station_stats["valid_days"] / expected_days_window
    station_stats["eligible"] = (station_stats["coverage"] >= 0.90) & (station_stats["valid_days"] >= 730)

    eligible = station_stats.loc[station_stats["eligible"], "station_id"].tolist()

    # Plot only the modeling window + valid days + eligible stations
    df_plot = df[
        (df["valid_day"])
        & (df["station_id"].isin(eligible))
        & (df["date"] >= WINDOW_START)
        & (df["date"] <= WINDOW_END)
    ].copy()

    # ----------------------------
    # Aggregate + weekly rollup (same as notebook)
    # ----------------------------
    daily = (
        df_plot.groupby("date", as_index=False)
        .agg(
            total_volume=("station_volume", "sum"),
            active_stations=("station_id", "nunique"),
        )
        .sort_values("date")
    )

    weekly = (
        daily.set_index("date")
        .resample("W-SUN")
        .agg(
            total_volume=("total_volume", "sum"),
            active_stations=("active_stations", "median"),
            n_days=("total_volume", "count"),
        )
        .reset_index()
        .rename(columns={"date": "week_ending"})
    )

    weekly = weekly[weekly["n_days"] >= 6].copy()

    weekly["vol_per_station"] = weekly["total_volume"] / weekly["active_stations"]
    weekly["vps_ma8"] = weekly["vol_per_station"].rolling(8, min_periods=1).mean()
    weekly["stations_ma8"] = weekly["active_stations"].rolling(8, min_periods=1).mean()

    x_min = weekly["week_ending"].min()
    x_max = weekly["week_ending"].max()

    # Strict inequalities: prevents the dashed marker showing up when WINDOW_START
    # is exactly at the left edge (which is what you see in the notebook output).
    show_vline = (x_min < WINDOW_START) and (WINDOW_START < x_max)

    shade_start = max(COVID_START, x_min)
    shade_end = min(WINDOW_START, x_max)
    do_shade = shade_start < shade_end

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["week_ending"],
            y=weekly["vol_per_station"],
            mode="lines",
            name="Weekly volume per active station",
            line=dict(width=1.5),
            opacity=0.25,
            hovertemplate="Week ending=%{x|%Y-%m-%d}<br>Per-station=%{y:,.0f}<extra></extra>",
            legendgroup="traffic",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["week_ending"],
            y=weekly["vps_ma8"],
            mode="lines",
            name="Per-station (8-week rolling mean)",
            line=dict(width=4),
            hovertemplate="Week ending=%{x|%Y-%m-%d}<br>8wk mean=%{y:,.0f}<extra></extra>",
            legendgroup="traffic",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["week_ending"],
            y=weekly["active_stations"],
            mode="lines",
            name="Active stations (weekly median)",
            line=dict(width=1.5, dash="dot"),
            opacity=0.35,
            hovertemplate="Week ending=%{x|%Y-%m-%d}<br>Active=%{y:.0f}<extra></extra>",
            legendgroup="coverage",
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["week_ending"],
            y=weekly["stations_ma8"],
            mode="lines",
            name="Active stations (8-week mean)",
            line=dict(width=3),
            opacity=0.90,
            hovertemplate="Week ending=%{x|%Y-%m-%d}<br>8wk mean=%{y:.1f}<extra></extra>",
            legendgroup="coverage",
        ),
        row=2, col=1
    )

    if show_vline:
        fig.add_vline(x=WINDOW_START, line_dash="dash", line_width=2)

    if do_shade:
        fig.add_vrect(
            x0=shade_start,
            x1=shade_end,
            fillcolor="rgba(0,0,0,0.05)",
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=dict(text="Figure 2 — System traffic over time (coverage-adjusted)", x=0.02, xanchor="left"),
        width=1400,
        height=850,
        margin=dict(l=80, r=40, t=120, b=130),
        legend=dict(orientation="h", x=0.02, xanchor="left", y=-0.08, yanchor="top"),
    )

    # Match notebook axis formatting (including the “override” you had)
    fig.update_yaxes(title_text="Weekly volume per active station", row=1, col=1, tickformat=",")
    fig.update_yaxes(title_text="Active stations", row=2, col=1)
    fig.update_xaxes(title_text="Date (week ending)", dtick="M12", tickformat="%Y", row=2, col=1)

    fig.update_yaxes(
        title_text="Total weekly station volume (millions)",
        row=1, col=1,
        tickformat="~s",
    )

    html_out = fig_dir / "fig02_system_traffic_coverage_adjusted.html"
    png_out = fig_dir / "fig02_system_traffic_coverage_adjusted.png"

    # Responsive HTML
    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    # PNG (single write, fixed size)
    _safe_write_image(fig, png_out, width=1400, height=850, scale=2)

    return [html_out, png_out]






# -------------------------
# Figure 3
# -------------------------

def make_fig03(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    WINDOW_START = pd.Timestamp("2021-06-01")
    WINDOW_END = pd.Timestamp("2024-12-31")
    expected_days_window = (WINDOW_END - WINDOW_START).days + 1

    # --- Match notebook cleaning assumptions ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["station_id", "date"]).copy()
    df["station_id"] = df["station_id"].astype(int)
    df["valid_day"] = df["valid_day"].astype(bool)

    # Keep only modeling window
    df_win = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].copy()

    # Use only valid station-days as supervised targets
    df_win_valid = df_win[df_win["valid_day"]].copy()

    station_stats = (
        df_win_valid.groupby("station_id", as_index=False)
        .agg(
            valid_days=("date", "nunique"),
            mean_volume=("station_volume", "mean"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
    )
    station_stats["coverage"] = station_stats["valid_days"] / expected_days_window
    station_stats["eligible_cov_0p90"] = station_stats["coverage"] >= 0.90
    station_stats["eligible_2yrs"] = station_stats["valid_days"] >= 730
    station_stats["eligible"] = station_stats["eligible_cov_0p90"] & station_stats["eligible_2yrs"]

    eligible_stations = station_stats.loc[station_stats["eligible"], "station_id"].tolist()
    df_model = df_win_valid[df_win_valid["station_id"].isin(eligible_stations)].copy()

    d = df_model[["station_id", "date", "station_volume"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["station_id", "date", "station_volume"])

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d["dow"] = d["date"].dt.dayofweek
    d["dow_name"] = d["dow"].map(dict(enumerate(dow_names)))
    d["is_weekend"] = d["dow"] >= 5

    # Normalize by station median
    station_med = d.groupby("station_id")["station_volume"].median().rename("station_median")
    d = d.merge(station_med, on="station_id", how="left")
    d["vol_ratio"] = d["station_volume"] / d["station_median"]

    # Winsorize
    lo, hi = d["vol_ratio"].quantile([0.01, 0.99])
    d["vol_ratio_w"] = d["vol_ratio"].clip(lower=lo, upper=hi)

    # Station-level averages by DOW
    station_dow = (
        d.groupby(["station_id", "dow", "dow_name"], as_index=False)
        .agg(mean_ratio=("vol_ratio_w", "mean"))
    )

    weekly_summary = (
        station_dow.groupby(["dow", "dow_name"], as_index=False)
        .agg(
            p25=("mean_ratio", lambda x: np.quantile(x, 0.25)),
            med=("mean_ratio", "median"),
            p75=("mean_ratio", lambda x: np.quantile(x, 0.75)),
        )
        .sort_values("dow")
    )

    x_dow = weekly_summary["dow_name"].tolist()

    # Weekday vs weekend: per-station means (pivot) -> arrays for violins
    station_weekpart = (
        d.groupby(["station_id", "is_weekend"], as_index=False)
        .agg(mean_ratio=("vol_ratio_w", "mean"))
    )

    pivot = (
        station_weekpart
        .pivot(index="station_id", columns="is_weekend", values="mean_ratio")
        .rename(columns={False: "weekday_mean", True: "weekend_mean"})
        .reset_index()
        .dropna(subset=["weekday_mean", "weekend_mean"])
    )
    pivot = pivot[pivot["weekday_mean"] > 0].copy()

    wkday = pivot["weekday_mean"].to_numpy()
    wkend = pivot["weekend_mean"].to_numpy()

    # -----------------------------
    # Figure (match notebook)
    # -----------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        subplot_titles=[
            "Day-of-week seasonality (median across stations, normalized)",
            "Weekday vs weekend level (distribution across stations)",
        ],
        horizontal_spacing=0.10,
    )

    # LEFT: IQR via tonexty
    fig.add_trace(
        go.Scatter(
            x=x_dow,
            y=weekly_summary["p75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_dow,
            y=weekly_summary["p25"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="IQR (25th–75th)",
            hovertemplate="DOW=%{x}<br>P25=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_dow,
            y=weekly_summary["med"],
            mode="lines+markers",
            name="Median station (ratio to station median)",
            hovertemplate="DOW=%{x}<br>Median=%{y:.3f}<extra></extra>",
            line=dict(width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_hline(y=1.0, line_dash="dash", row=1, col=1)

    # RIGHT: violin plots (no legend)
    fig.add_trace(
        go.Violin(
            y=wkday,
            name="Weekday (Mon–Fri)",
            box_visible=True,
            meanline_visible=True,
            points=False,
            showlegend=False,
            hovertemplate="Weekday<br>Ratio=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Violin(
            y=wkend,
            name="Weekend (Sat–Sun)",
            box_visible=True,
            meanline_visible=True,
            points=False,
            showlegend=False,
            hovertemplate="Weekend<br>Ratio=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Figure 3 — Weekly seasonality intuition", x=0.02, xanchor="left"),
        width=1750,
        height=650,
        margin=dict(l=90, r=80, t=120, b=95),
        hovermode="x unified",
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.10,
            xanchor="left",
            y=0.06,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
        )
    )

    fig.update_yaxes(title_text="Volume ratio (station-normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Volume ratio (station-normalized)", row=1, col=2)
    fig.update_xaxes(title_text="Day of week", row=1, col=1)

    html_out = fig_dir / "fig03_weekly_seasonality.html"
    png_out = fig_dir / "fig03_weekly_seasonality.png"

    # Notebook exports HTML (PNG was commented out there)
    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    # Keep PNG export in figures script (since script promises PNG + HTML)
    _safe_write_image(fig, png_out, width=1750, height=650, scale=2)

    return [html_out, png_out]



# -------------------------
# Figures 4–6 and 8 (need predictions)
# -------------------------

def make_fig04(pred: pd.DataFrame, fig_dir: Path) -> list[Path]:
    df_t = pred[pred["split"] == "test"].copy()

    by_station = (
        df_t.groupby("station_id", as_index=False)
        .apply(lambda g: pd.Series({
            "smape_base": smape(g["vol_true"], g["vol_base"]),
            "smape_et": smape(g["vol_true"], g["vol_pred"]),
            "mean_volume": float(g["vol_true"].mean()),
        }))
        .reset_index(drop=True)
    )
    by_station["delta_smape_pp"] = by_station["smape_base"] - by_station["smape_et"]

    improved_pct = 100.0 * (by_station["delta_smape_pp"] > 0).mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=by_station["delta_smape_pp"],
        nbinsx=30,
        hovertemplate="Δ sMAPE=%{x:.2f} pp<br>Count=%{y}<extra></extra>",
        name="Stations",
        showlegend=False,
    ))

    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"Figure 4 — Station-level improvement on test (Δ sMAPE = baseline − ET) | Improved: {improved_pct:.1f}%",
            x=0.02,
            xanchor="left",
        ),
        width=1500,
        height=520,
        margin=dict(l=80, r=40, t=95, b=70),
    )

    fig.update_xaxes(title_text="Δ sMAPE (percentage points); >0 means ET improves over baseline")
    fig.update_yaxes(title_text="Number of stations")

    html_out = fig_dir / "fig04_station_improvement_distribution.html"
    png_out = fig_dir / "fig04_station_improvement_distribution.png"

    _safe_write_image(fig, png_out, width=1500, height=520, scale=2)

    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    return [html_out, png_out]


def make_fig05(pred: pd.DataFrame, fig_dir: Path) -> list[Path]:
    df_t = pred[pred["split"] == "test"].copy()

    def _station_metrics(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "smape_base": smape(g["vol_true"], g["vol_base"]),
            "smape_et": smape(g["vol_true"], g["vol_pred"]),
            "mean_volume": float(g["vol_true"].mean()),
        })

    by_station = (
        df_t.groupby("station_id", as_index=False)
        .apply(_station_metrics)
        .reset_index(drop=True)
    )
    by_station["delta_smape_pp"] = by_station["smape_base"] - by_station["smape_et"]

    by_station = by_station.dropna(subset=["mean_volume", "delta_smape_pp"]).copy()
    by_station["tier"] = pd.qcut(by_station["mean_volume"], q=3, labels=["low", "mid", "high"]).astype(str)

    df_sc = by_station.sort_values("mean_volume").copy()
    df_sc["mean_volume"] = df_sc["mean_volume"].clip(lower=1)
    df_sc["log_mean_vol"] = np.log10(df_sc["mean_volume"])

    n_bins = 8
    df_sc["qbin"] = pd.qcut(df_sc["log_mean_vol"], q=n_bins, duplicates="drop")

    trend = (
        df_sc.groupby("qbin", as_index=False)
        .agg(
            logx=("log_mean_vol", "median"),
            med=("delta_smape_pp", "median"),
            q25=("delta_smape_pp", lambda s: np.quantile(s, 0.25)),
            q75=("delta_smape_pp", lambda s: np.quantile(s, 0.75)),
            n=("delta_smape_pp", "size"),
        )
    )
    trend["x"] = 10 ** trend["logx"]

    x_ribbon = np.concatenate([trend["x"].values, trend["x"].values[::-1]])
    y_ribbon = np.concatenate([trend["q75"].values, trend["q25"].values[::-1]])

    y_min = min(float(df_sc["delta_smape_pp"].min()), 0.0) - 0.5
    y_max = max(float(df_sc["delta_smape_pp"].max()), 0.0) + 0.5

    tickvals = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]
    ticktext = ["1k", "2k", "5k", "10k", "20k", "50k", "100k"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.10,
        subplot_titles=[
            "<b>A.</b> Improvement vs station scale (test) — each point is a station",
            "<b>B.</b> Improvement by volume tier (test)",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df_sc["mean_volume"],
            y=df_sc["delta_smape_pp"],
            mode="markers",
            marker=dict(size=9, opacity=0.45),
            customdata=np.stack([df_sc["station_id"].values, df_sc["smape_base"].values, df_sc["smape_et"].values], axis=1),
            hovertemplate=(
                "Station=%{customdata[0]}<br>"
                "Mean vol=%{x:,.0f}<br>"
                "Δ sMAPE=%{y:.2f} pp<br>"
                "sMAPE(base)=%{customdata[1]:.2f}<br>"
                "sMAPE(ET)=%{customdata[2]:.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_ribbon,
            y=y_ribbon,
            fill="toself",
            line=dict(width=0),
            opacity=0.18,
            name="Binned IQR (25–75%)",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=trend["x"],
            y=trend["med"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=7),
            name="Binned median trend",
            customdata=trend["n"].values,
            hovertemplate="Mean vol≈%{x:,.0f}<br>Median Δ=%{y:.2f} pp<br>n=%{customdata}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dash", opacity=0.9, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", opacity=0.9, row=1, col=2)

    fig.update_xaxes(
        title_text="Mean station volume (vehicles/day, log scale)",
        type="log",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Δ sMAPE (pp) = baseline − ET (positive is better)",
        range=[y_min, y_max],
        row=1,
        col=1,
    )

    x_anno = float(trend["x"].max()) if len(trend) else float(df_sc["mean_volume"].max())
    fig.add_annotation(
        x=x_anno,
        y=0,
        xref="x1",
        yref="y1",
        text="0 = no change",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.10)",
        borderwidth=1,
        borderpad=3,
    )

    tier_order = ["low", "mid", "high"]
    for t in tier_order:
        vals = by_station.loc[by_station["tier"] == t, "delta_smape_pp"].values
        fig.add_trace(
            go.Box(
                x=[t] * len(vals),
                y=vals,
                name=t,
                boxpoints="all",
                jitter=0.35,
                pointpos=0.0,
                marker=dict(size=6, opacity=0.35),
                line=dict(width=2),
                showlegend=False,
                hovertemplate=f"Tier={t}<br>Δ=%{{y:.2f}} pp<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(
        row=1,
        col=2,
        title_text="Volume tier",
        type="category",
        categoryorder="array",
        categoryarray=tier_order,
    )

    fig.update_yaxes(title_text="Δ sMAPE (pp)", range=[y_min, y_max], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Figure 5 — Where ET helps most (error slicing on test)", x=0.02, xanchor="left"),
        width=1700,
        height=640,
        margin=dict(l=80, r=40, t=110, b=80),
        hovermode="closest",
        legend=dict(orientation="h", x=0.02, xanchor="left", y=1.03, yanchor="bottom", font=dict(size=11)),
    )

    html_out = fig_dir / "fig05_error_slicing_improvement_vs_volume.html"
    png_out = fig_dir / "fig05_error_slicing_improvement_vs_volume.png"

    _safe_write_image(fig, png_out, width=1700, height=640, scale=2)

    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    return [html_out, png_out]


def make_fig06(pred: pd.DataFrame, fig_dir: Path) -> list[Path]:
    df_t = pred[pred["split"] == "test"].copy()

    st = df_t.groupby("station_id", as_index=False).agg(mean_vol=("vol_true", "mean")).sort_values("mean_vol")

    low_ids = st.head(2)["station_id"].tolist()
    mid_ids = st.iloc[len(st) // 2 - 1 : len(st) // 2 + 1]["station_id"].tolist()
    high_ids = st.tail(2)["station_id"].tolist()
    station_ids = low_ids + mid_ids + high_ids

    end_date = df_t["date"].max()
    start_date = end_date - pd.Timedelta(days=90)
    df_plot = df_t[(df_t["station_id"].isin(station_ids)) & (df_t["date"] >= start_date)].copy()

    COLORS = {
        "Actual": "#1f77b4",
        "Baseline (lag-7)": "#ff7f0e",
        "ET": "#2ca02c",
    }

    def daily_smape(y, yhat, eps=1e-9):
        y = np.asarray(y)
        yhat = np.asarray(yhat)
        denom = np.maximum(np.abs(y) + np.abs(yhat), eps)
        return 2.0 * np.abs(yhat - y) / denom

    def win_rate_smape(vol_true, vol_pred, vol_base):
        d_et = daily_smape(vol_true, vol_pred)
        d_base = daily_smape(vol_true, vol_base)
        return float((d_et < d_base).mean())

    title_map = {}
    for sid in station_ids:
        g = df_plot[df_plot["station_id"] == sid].sort_values("date")
        s_base = smape(g["vol_true"], g["vol_base"])
        s_et = smape(g["vol_true"], g["vol_pred"])
        delta = s_base - s_et
        wr = win_rate_smape(g["vol_true"].values, g["vol_pred"].values, g["vol_base"].values)
        title_map[sid] = (
            f"Station {sid}<br>"
            f"ET {s_et:.2f}% vs Base {s_base:.2f}% (Δ {delta:+.2f} pp)<br>"
            f"Wins {wr*100:.1f}% of days"
        )

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[title_map[sid] for sid in station_ids],
        vertical_spacing=0.11,
        horizontal_spacing=0.12,
    )

    def add_series(sub_row, sub_col, g):
        show_leg = sub_row == 1 and sub_col == 1

        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["vol_true"],
                mode="lines",
                name="Actual",
                legendgroup="actual",
                showlegend=show_leg,
                line=dict(color=COLORS["Actual"], width=2),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Actual=%{y:,.0f}<extra></extra>",
            ),
            row=sub_row,
            col=sub_col,
        )

        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["vol_base"],
                mode="lines",
                name="Baseline (lag-7)",
                legendgroup="base",
                showlegend=show_leg,
                line=dict(color=COLORS["Baseline (lag-7)"], width=2, dash="dot"),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Baseline=%{y:,.0f}<extra></extra>",
            ),
            row=sub_row,
            col=sub_col,
        )

        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["vol_pred"],
                mode="lines",
                name="ET",
                legendgroup="et",
                showlegend=show_leg,
                line=dict(color=COLORS["ET"], width=2.5),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>ET=%{y:,.0f}<extra></extra>",
            ),
            row=sub_row,
            col=sub_col,
        )

    for i, sid in enumerate(station_ids):
        r = i // 2 + 1
        c = i % 2 + 1
        g = df_plot[df_plot["station_id"] == sid].sort_values("date")
        add_series(r, c, g)

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Figure 6 — Example forecasts (last 90 days of test): Actual vs Baseline vs ET",
            x=0.02,
            xanchor="left",
            y=0.99,
            yanchor="top",
        ),
        width=1750,
        height=1050,
        margin=dict(l=70, r=40, t=165, b=70),
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, yanchor="bottom", font=dict(size=12)),
    )

    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12), yshift=-14)

    for rr in range(1, 4):
        for cc in range(1, 3):
            fig.update_yaxes(title_text="Vehicles/day" if cc == 1 else "", row=rr, col=cc)
            fig.update_xaxes(title_text="Date" if rr == 3 else "", row=rr, col=cc)

    html_out = fig_dir / "fig06_example_forecasts_smallmultiples.html"
    png_out = fig_dir / "fig06_example_forecasts_smallmultiples.png"

    _safe_write_image(fig, png_out, width=1750, height=1050, scale=2)

    fig.update_layout(autosize=True, width=None, height=920, title_font=dict(size=18))
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    return [html_out, png_out]


# -------------------------
# Figure 7 (needs backtest)
# -------------------------

def make_fig07(bt: pd.DataFrame, fig_dir: Path) -> list[Path]:
    # Match notebook behavior as closely as possible
    bt = bt.copy()
    bt["test_start"] = pd.to_datetime(bt["test_start"], errors="coerce")
    bt = bt.dropna(subset=["test_start"]).sort_values("test_start")

    x = bt["test_start"]
    y = bt["vol_smape_improve_pp"].astype(float)  # Δ sMAPE = baseline − ET (pp)

    mean_imp = float(np.mean(y))
    median_imp = float(np.median(y))
    min_imp = float(np.min(y))
    max_imp = float(np.max(y))

    # Highlight "most volatile" baseline window(s) (highest baseline sMAPE)
    topk = 1
    volatile_idx = bt["vol_smape_base"].nlargest(topk).index if "vol_smape_base" in bt.columns else []
    volatile_dates = bt.loc[volatile_idx, "test_start"].tolist() if len(volatile_idx) else []

    # Axis range: emphasize "all positive" by starting at 0
    y_upper = max_imp + 0.75  # headroom for callouts
    y_lower = 0

    # Try to match notebook title: use explicit HORIZON_DAYS / STEP_DAYS if present,
    # otherwise fall back to inferring from columns.
    horizon_days = None
    step_days = None

    # Prefer explicit columns if they exist (some pipelines store these)
    for c in ["horizon_days", "HORIZON_DAYS", "horizon", "HORIZON"]:
        if c in bt.columns:
            try:
                horizon_days = int(bt[c].iloc[0])
                break
            except Exception:
                pass

    for c in ["step_days", "STEP_DAYS", "step", "STEP"]:
        if c in bt.columns:
            try:
                step_days = int(bt[c].iloc[0])
                break
            except Exception:
                pass

    # Otherwise infer best-effort from test_end and successive test_start
    if horizon_days is None and "test_end" in bt.columns and len(bt) > 0:
        te = pd.to_datetime(bt["test_end"], errors="coerce")
        if te.notna().any():
            try:
                horizon_days = int((te.iloc[0] - bt["test_start"].iloc[0]).days + 1)
            except Exception:
                pass

    if step_days is None and len(bt) >= 2:
        try:
            step_days = int((bt["test_start"].iloc[1] - bt["test_start"].iloc[0]).days)
        except Exception:
            pass

    # -----------------------------
    # Figure
    # -----------------------------
    fig = go.Figure()

    # Bars: improvement per fold
    # Notebook hover shows Baseline and ET in tooltip via customdata
    customdata = None
    if ("vol_smape_base" in bt.columns) and ("vol_smape_et" in bt.columns):
        customdata = np.stack([bt["vol_smape_base"].values, bt["vol_smape_et"].values], axis=1)

    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name="Δ sMAPE (baseline − ET)",
            opacity=0.85,
            hovertemplate=(
                "Fold start=%{x|%Y-%m-%d}<br>"
                "Δ sMAPE=%{y:.2f} pp<br>"
                "Baseline=%{customdata[0]:.2f}%<br>"
                "ET=%{customdata[1]:.2f}%"
                "<extra></extra>"
            ) if customdata is not None else (
                "Fold start=%{x|%Y-%m-%d}<br>"
                "Δ sMAPE=%{y:.2f} pp"
                "<extra></extra>"
            ),
            customdata=customdata,
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", opacity=0.9)

    # Mean + median reference lines
    fig.add_hline(y=median_imp, line_dash="solid", line_width=2, opacity=0.85)
    fig.add_hline(y=mean_imp, line_dash="dot", line_width=2, opacity=0.85)

    # Shade the most volatile window(s) (28-day block)
    for d in volatile_dates:
        fig.add_vrect(
            x0=d,
            x1=d + pd.Timedelta(days=28),
            fillcolor="rgba(0,0,0,0.05)",
            line_width=0,
            layer="below",
        )

    # -----------------------------
    # Annotations (non-overlapping)
    # -----------------------------
    label_box = dict(
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(0,0,0,0.10)",
        borderwidth=1,
        borderpad=3,
    )

    # Right-edge labels for reference lines
    fig.add_annotation(
        x=0.99, xref="paper",
        y=median_imp, yref="y",
        text=f"Median: {median_imp:.2f} pp",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        **label_box,
    )
    fig.add_annotation(
        x=0.99, xref="paper",
        y=mean_imp, yref="y",
        text=f"Mean: {mean_imp:.2f} pp",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        **label_box,
    )

    # Label the shaded "volatile" window
    if volatile_dates:
        d0 = volatile_dates[0]
        fig.add_annotation(
            x=d0 + pd.Timedelta(days=2),
            y=y_upper - 0.10,
            xref="x",
            yref="y",
            text="Most volatile baseline window",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            **label_box,
        )

    # Stats callout (anchor to paper coords so it stays top-right)
    # Notebook hardcodes "(16 folds)" and "All folds improved (Δ>0)"
    n_folds = int(len(bt))
    all_improved = bool(np.all(y > 0))

    stats_text = (
        f"<b>Backtest summary ({n_folds} folds)</b><br>"
        f"{'All folds improved (Δ>0)<br>' if all_improved else ''}"
        f"Range: {min_imp:.2f} → {max_imp:.2f} pp<br>"
        f"Median: {median_imp:.2f} pp<br>"
        f"Mean: {mean_imp:.2f} pp"
    )
    fig.add_annotation(
        x=0.985, xref="paper",
        y=y_upper - 0.10, yref="y",
        text=stats_text,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        borderpad=6,
    )

    # -----------------------------
    # Layout
    # -----------------------------
    FIG_W, FIG_H = 1700, 600

    # Title exactly like notebook: include horizon + step
    # If we can't determine them, omit rather than invent.
    if (horizon_days is not None) and (step_days is not None):
        title = (
            "Figure 7 — Rolling backtest improvement per fold "
            f"(Δ sMAPE = baseline − ET, {horizon_days}-day horizon, step={step_days}d)"
        )
    else:
        title = "Figure 7 — Rolling backtest improvement per fold (Δ sMAPE = baseline − ET)"

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, xanchor="left"),
        width=FIG_W,
        height=FIG_H,
        margin=dict(l=80, r=40, t=90, b=70),
        hovermode="x unified",
        showlegend=False,
        bargap=0.25,
    )

    fig.update_xaxes(title_text="Fold test start date", dtick="M3", tickformat="%Y-%m")
    fig.update_yaxes(title_text="Δ volume sMAPE (percentage points)", range=[y_lower, y_upper])

    # -----------------------------
    # Export (match notebook)
    # -----------------------------
    html_out = fig_dir / "fig07_rolling_backtest_improvement_per_fold.html"
    png_out = fig_dir / "fig07_rolling_backtest_improvement_per_fold.png"

    _safe_write_image(fig, png_out, width=FIG_W, height=FIG_H, scale=2)

    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    return [html_out, png_out]



# -------------------------
# Figure 8
# -------------------------

def make_fig08(pred: pd.DataFrame, fig_dir: Path) -> list[Path]:
    dfp = pred.copy()

    date_col = "date" if "date" in dfp.columns else None
    if date_col is None:
        for c in ["ds", "timestamp", "day", "datetime"]:
            if c in dfp.columns:
                date_col = c
                break
    if date_col is None:
        raise ValueError("Could not find a date column in predictions")

    dfp[date_col] = pd.to_datetime(dfp[date_col], errors="coerce")
    dfp = dfp.dropna(subset=[date_col, "station_id", "vol_true", "vol_pred"]).copy()
    dfp["station_id"] = dfp["station_id"].astype(int)

    if "split" in dfp.columns:
        dfp = dfp[dfp["split"].eq("test")].copy()

    daily = (
        dfp.groupby(date_col, as_index=False)
        .agg(true_total=("vol_true", "sum"), hgb_total=("vol_pred", "sum"), active_stations=("station_id", "nunique"))
        .sort_values(date_col)
    )

    has_base = "vol_base" in dfp.columns
    if has_base:
        daily_base = dfp.groupby(date_col, as_index=False).agg(base_total=("vol_base", "sum"))
        daily = daily.merge(daily_base, on=date_col, how="left")

    weekly = (
        daily.set_index(date_col)
        .resample("W-SUN")
        .agg(
            true_total=("true_total", "sum"),
            hgb_total=("hgb_total", "sum"),
            active_stations=("active_stations", "median"),
            n_days=("true_total", "count"),
            **({"base_total": ("base_total", "sum")} if has_base else {}),
        )
        .reset_index()
        .rename(columns={date_col: "week_ending"})
    )

    weekly = weekly[weekly["n_days"] >= 6].copy()

    weekly["true_ma8"] = weekly["true_total"].rolling(8, min_periods=1).mean()
    weekly["hgb_ma8"] = weekly["hgb_total"].rolling(8, min_periods=1).mean()
    if has_base:
        weekly["base_ma8"] = weekly["base_total"].rolling(8, min_periods=1).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.78, 0.22],
        subplot_titles=[
            "System total volume (weekly) — Actual vs ET",
            "Coverage: active stations contributing each week",
        ],
    )

    fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["true_total"], mode="lines", name="Actual (weekly)", line=dict(width=1.5), opacity=0.25), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["hgb_total"], mode="lines", name="ET (weekly)", line=dict(width=1.5), opacity=0.25), row=1, col=1)

    fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["true_ma8"], mode="lines", name="Actual (8-week mean)", line=dict(width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["hgb_ma8"], mode="lines", name="ET (8-week mean)", line=dict(width=4)), row=1, col=1)

    if has_base:
        fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["base_ma8"], mode="lines", name="Baseline lag-7 (8-week mean)", line=dict(width=2, dash="dot"), opacity=0.6), row=1, col=1)

    fig.add_trace(go.Scatter(x=weekly["week_ending"], y=weekly["active_stations"], mode="lines", name="Active stations (median)", line=dict(width=2, dash="dot"), opacity=0.85, showlegend=False), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=dict(text="Figure 8 — ET system-total forecast vs actual (aggregated across stations)", x=0.02, xanchor="left"),
        width=1500,
        height=820,
        margin=dict(l=85, r=40, t=110, b=85),
        legend=dict(orientation="h", x=0.02, xanchor="left", y=-0.10, yanchor="top"),
    )

    fig.update_yaxes(title_text="Weekly total volume", row=1, col=1, tickformat="~s")
    fig.update_yaxes(title_text="Active stations", row=2, col=1)
    fig.update_xaxes(title_text="Date (week ending)", dtick="M12", tickformat="%Y", row=2, col=1)

    html_out = fig_dir / "fig08_system_total_actual_vs_et.html"
    png_out = fig_dir / "fig08_system_total_actual_vs_et.png"

    _safe_write_image(fig, png_out, width=1500, height=820, scale=2)

    fig.update_layout(autosize=True)
    fig.layout.width = None
    fig.layout.height = None
    fig.write_html(html_out, include_plotlyjs="cdn", config={"responsive": True, "displayModeBar": False})

    return [html_out, png_out]


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--station-day", type=Path, default=Path("mndot_outputs/atr_station_day_2017_2024.csv"))
    p.add_argument("--pred", type=Path, default=Path("reports/preds/pred_val_test.csv.gz"))
    p.add_argument("--backtest", type=Path, default=Path("reports/backtest/rolling_backtest_results.csv"))
    p.add_argument("--outdir", type=Path, default=Path("assets/figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    if args.station_day.exists():
        df = pd.read_csv(args.station_day)
        outputs += make_fig01(df, args.outdir)
        outputs += make_fig02(df, args.outdir)
        outputs += make_fig03(df, args.outdir)
    else:
        print(f"[warn] station-day CSV not found: {args.station_day}. Skipping Figures 1–3.")

    if args.pred.exists():
        pred = pd.read_csv(args.pred, compression="infer")
        pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
        outputs += make_fig04(pred, args.outdir)
        outputs += make_fig05(pred, args.outdir)
        outputs += make_fig06(pred, args.outdir)
        outputs += make_fig08(pred, args.outdir)
    else:
        print(f"[warn] predictions file not found: {args.pred}. Skipping Figures 4–6 and 8.")

    if args.backtest.exists():
        bt = pd.read_csv(args.backtest)
        outputs += make_fig07(bt, args.outdir)
    else:
        print(f"[warn] backtest file not found: {args.backtest}. Skipping Figure 7.")

    # Summarize
    wrote = [p for p in outputs if p.exists()]
    print(f"\nDone. Wrote {len(wrote)} files to {args.outdir}:\n")
    for p in wrote:
        print(" -", p.name)


if __name__ == "__main__":
    main()

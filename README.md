# MnDOT Daily Traffic Volume Forecasting (Station-Day Global Model)

This repository builds a **station-day panel** from MnDOT ATR/WIM station data, engineers lag/rolling features, and evaluates baseline + learned models to forecast **daily traffic volume**. It is designed to be **fully reproducible from the command line** (no notebooks required).

---

## What this repo does

Given MnDOT station-day volume observations (2017–2024), this project:

- Builds a modeling-ready **station-day panel** with consistent filtering and feature engineering
- Trains and compares multiple model families against simple baselines
- Produces **out-of-sample validation + test predictions** at the station-day level
- Runs an **expanding-window rolling backtest** to test stability over time
- Exports the final **figures as HTML + PNG** for use in reports / portfolio writeups

---

## Repo layout

- `src/traffic_forecasting/` – reusable library code (feature design, modeling, metrics, backtesting)
- `scripts/` – command-line entrypoints (reproducible runs)
- `data/` – (ignored by default) raw/interim/processed artifacts
- `reports/` – model results, predictions, backtests (ignored by default)
- `assets/figures/` – exported figures (ignored by default)

> Note: This repo assumes the notebook is **removed**. The scripts are the source of truth.

---

## Requirements

- Python 3.10+
- Dependencies are defined in `pyproject.toml` (Plotly image export uses `kaleido`)

---

## Quickstart

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
````

---

## Data

### Data download (MnDOT ATR hourly volume)

You can download the raw ATR hourly volume data directly from MnDOT here:

https://www.dot.state.mn.us/traffic/data/reports-hrvol-atr.html

From that page, download the ATR hourly volume report(s) you need, then use **Step 0** (optional) to convert the hourly files into the station-day totals used by this repo.

### Inputs

This pipeline expects a **station-day CSV** with (at minimum):

* `station_id`
* `date`
* `station_volume`
* `valid_day` (boolean indicator for usable station-days)

By default the figures script assumes:

* `mndot_outputs/atr_station_day_2017_2024.csv`

If you already have station-day totals, you can skip the hourly-to-daily build step and point scripts directly at your station-day CSV.

### Output artifacts

By default, the pipeline writes:

* Panel: `data/processed/panel_model_table.csv.gz`
* Predictions: `reports/predictions/pred_val_test.csv.gz`
* Backtest results: (default path determined by the backtest script)
* Figures: `assets/figures/*.png` and `assets/figures/*.html`

---

## Pipeline

Run the steps below in order. **Do not skip steps** unless you already have the exact artifacts they produce.

### 0) (Optional) Build station-day totals from hourly ATR/WIM CSVs

```bash
python scripts/process_mndot_atr_wim.py --input path/to/data.zip --outdir mndot_outputs
```

This writes `mndot_outputs/atr_station_day_<minyear>_<maxyear>.csv`.

### 1) Build the modeling panel (post-COVID window + lag/rolling features)

```bash
python scripts/01_build_panel.py   --station-day mndot_outputs/atr_station_day_2017_2024.csv   --out data/processed/panel_model_table.csv.gz
```

### 2) Run the model bake-off

```bash
python scripts/02_model_bakeoff.py --panel data/processed/panel_model_table.csv.gz
```

### 3) Fit the selected model and export predictions (val + test)

```bash
python scripts/03_fit_best_model_and_preds.py   --panel data/processed/panel_model_table.csv.gz   --out reports/predictions/pred_val_test.csv.gz
```

### 4) Rolling backtest

```bash
python scripts/04_rolling_backtest.py --panel data/processed/panel_model_table.csv.gz
```

### 5) Make figures

```bash
python scripts/05_make_figures.py --pred reports/predictions/pred_val_test.csv.gz
```

---

## Modeling approach

### Problem framing

* **Unit of prediction:** station-day
* **Target:** daily traffic volume (vehicles/day)
* **Goal:** a single **global model** that generalizes across stations while remaining stable across time

### Key choices

* **Post-COVID modeling window:** The panel is built on a post-COVID period (default: `2021-06-01 → 2024-12-31`) to avoid mixing regimes where traffic dynamics and seasonality differ sharply.
* **Station eligibility:** Stations are filtered to those with sufficient coverage and history inside the modeling window to support stable training and evaluation.
* **Station-normalized evaluation:** Metrics are computed at the station level and aggregated so that each station contributes comparably (reducing domination by the highest-volume locations).

---

## Features (high level)

The panel construction creates time-series predictors designed for daily traffic:

* **Lag features** (e.g., lag-1, lag-7)
* **Rolling statistics** (e.g., short/medium rolling means to capture local trend)
* **Calendar effects** (day-of-week patterns and weekly seasonality)
* **Coverage/valid-day handling** to avoid contaminating the target with known-bad days

Exact feature definitions live in the library code and are used consistently across scripts.

---

## Evaluation

### Splits

The bake-off and prediction export script produces evaluation splits that include:

* Validation predictions
* Test predictions

These are stored in the exported predictions file and drive multiple downstream figures.

### Metrics

Typical reporting includes:

* **sMAPE** in original units (interpretable percentage error)
* Additional summary metrics as produced by the model scripts

Station-level metrics are also used to:

* Measure **how often** the learned model beats the baseline (win-rate style summaries)
* Slice performance by station scale (low/mid/high volume tiers)

---

## Figures

The figures script exports the same figures used in the report as:

* `.png` (static)
* `.html` (interactive, Plotly)

The key figure themes:

* **System traffic trend** over time (coverage-adjusted)
* **Weekly seasonality intuition** (station-normalized day-of-week profile)
* **Station-level improvements** vs baseline (distribution + slicing by station scale)
* **Example forecast panels** for representative stations
* **Rolling backtest improvement** by fold (stability over time)

All figure generation is centralized in:

* `scripts/05_make_figures.py`

---

## Reproducibility notes

* The pipeline is intentionally script-first so results can be reproduced without notebooks.
* Outputs are written to `data/`, `reports/`, and `assets/figures/` which are typically ignored in version control to keep the repo lightweight.
* If you are running in a clean environment, ensure `kaleido` is installed (it is included in the default dependencies) so Plotly can export PNGs.

---

## Where to start

* If you want to reproduce results end-to-end: start at **Step 1** and run through **Step 5**
* If you already have the panel file: start at **Step 2**
* If you want figures only: run **Step 5** once `reports/predictions/pred_val_test.csv.gz` exists

---

## Contact / usage

This repo is intended for portfolio-style reproducible modeling work. If you reuse pieces:

* Cite MnDOT as the source of the underlying traffic counts
* Keep the split / evaluation logic consistent to avoid overstating performance

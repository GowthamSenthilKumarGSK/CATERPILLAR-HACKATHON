"""
Equipment rental demand forecasting -- simplified.

Demand is defined as one thing only: total working (engine) hours per
(site_id, equipment_type, week). No rental counts, no switchable targets, no
Poisson/Tweedie objective selection -- just plain regression.

Two models, compared honestly on the same held-out weeks:
  - LightGBM: one global model across all (site, type) series, using lag/
    rolling features + calendar seasonality.
  - ARIMA (auto_arima): a simple univariate model fit per (site, type) series.

Validation is walk-forward (train on the past, predict the next weeks, never
shuffle) -- the only correct way to validate a time series model.
"""

from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. Load data
# --------------------------------------------------------------------------- #

def load_joined_rental_data(db_path: str = "equipment_rental.db") -> pd.DataFrame:
    """Load rentals joined with equipment type."""
    conn = sqlite3.connect(db_path)
    rentals = pd.read_sql_query(
        """
        SELECT r.id, r.site_id, r.check_in_date, r.rental_days,
               r.engine_hours_per_day, e.type AS equipment_type
        FROM rentals r
        LEFT JOIN equipment e ON r.equipment_id = e.equipment_id
        """,
        conn,
        parse_dates=["check_in_date"],
    )
    conn.close()

    rentals = rentals.dropna(subset=["check_in_date"]).copy()
    rentals["site_id"] = rentals["site_id"].fillna("Unknown")
    rentals["equipment_type"] = rentals["equipment_type"].fillna("Unknown")
    return rentals


# --------------------------------------------------------------------------- #
# 2. Build weekly working-hours panel
# --------------------------------------------------------------------------- #

def build_demand_panel(rentals: pd.DataFrame, freq: str = "W-MON") -> pd.DataFrame:
    """
    Build a complete (site_id, equipment_type, week) panel of total working
    hours, filling weeks with no activity as 0.

    A rental's engine hours are spread across every week it actually runs
    through (e.g. a 15-day rental contributes hours to all 3 weeks it spans),
    not just its start week.
    """
    r = rentals.copy()
    r["rental_days"] = r["rental_days"].fillna(1).clip(lower=1).astype(int)
    r["engine_hours_per_day"] = r["engine_hours_per_day"].fillna(0.0)
    as_of = r["check_in_date"].max()

    # Expand each rental into one row per active day it ran.
    rep_idx = np.repeat(np.arange(len(r)), r["rental_days"].to_numpy())
    offsets = np.concatenate([np.arange(n) for n in r["rental_days"]])
    daily = r.iloc[rep_idx].reset_index(drop=True)
    daily["active_date"] = daily["check_in_date"].to_numpy() + pd.to_timedelta(offsets, unit="D")
    daily = daily[daily["active_date"] <= as_of]  # don't project hours into the future

    daily["week"] = daily["active_date"].dt.to_period("W").dt.to_timestamp(how="start")
    hours = (
        daily.groupby(["site_id", "equipment_type", "week"])["engine_hours_per_day"]
        .sum()
        .rename("working_hours")
        .reset_index()
    )

    # Fill in every (site, type, week) combination, including weeks with 0 hours.
    sites = r["site_id"].unique()
    types = r["equipment_type"].unique()
    full_weeks = pd.date_range(hours["week"].min(), hours["week"].max(), freq=freq)
    idx = pd.MultiIndex.from_product([sites, types, full_weeks], names=["site_id", "equipment_type", "week"])

    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(hours, on=["site_id", "equipment_type", "week"], how="left")
    panel["working_hours"] = panel["working_hours"].fillna(0.0)
    panel = panel.sort_values(["site_id", "equipment_type", "week"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# 3. Features for LightGBM
# --------------------------------------------------------------------------- #

def add_features(panel: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 4, 8)) -> pd.DataFrame:
    """Lag + rolling + calendar features. All shifted so no row sees its own target."""
    df = panel.sort_values(["site_id", "equipment_type", "week"]).copy()
    grp = df.groupby(["site_id", "equipment_type"])["working_hours"]

    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    key = [df["site_id"], df["equipment_type"]]
    df["roll_mean_4"] = shifted.groupby(key).transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["roll_mean_8"] = shifted.groupby(key).transform(lambda s: s.rolling(8, min_periods=1).mean())

    df["month"] = df["week"].dt.month
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)

    df["site_id"] = df["site_id"].astype("category")
    df["equipment_type"] = df["equipment_type"].astype("category")
    return df


FEATURE_COLS = [
    "site_id", "equipment_type",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8",
    "roll_mean_4", "roll_mean_8",
    "month", "weekofyear",
]


# --------------------------------------------------------------------------- #
# 4. Walk-forward validation (LightGBM: one global model, all series at once)
# --------------------------------------------------------------------------- #

@dataclass
class ValidationResult:
    fold_metrics: pd.DataFrame
    mean_mae: float
    mean_rmse: float


def validate_lightgbm(features_df: pd.DataFrame, n_folds: int = 4, min_train_weeks: int = 20) -> ValidationResult:
    """
    Expanding-window walk-forward validation: train on all weeks up to a
    cutoff, predict the next week, score only on that held-out week, advance
    the cutoff, repeat. Never a random train/test split -- that would let the
    model see the future through neighboring weeks in the same series.
    """
    weeks = sorted(features_df["week"].unique())
    if len(weeks) < min_train_weeks + n_folds:
        raise ValueError(
            f"Only {len(weeks)} weeks available; need at least "
            f"{min_train_weeks + n_folds} for {n_folds} honest folds. "
            f"Reduce n_folds/min_train_weeks or provide more history."
        )

    rows = []
    test_start = len(weeks) - n_folds
    for fold in range(n_folds):
        cutoff = test_start + fold
        train_weeks, test_week = weeks[:cutoff], weeks[cutoff]

        train = features_df[features_df["week"].isin(train_weeks)].dropna(subset=FEATURE_COLS)
        test = features_df[features_df["week"] == test_week].dropna(subset=FEATURE_COLS)
        if train.empty or test.empty:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, random_state=RANDOM_STATE, verbose=-1,
        )
        model.fit(train[FEATURE_COLS], train["working_hours"],
                   categorical_feature=["site_id", "equipment_type"])
        preds = np.clip(model.predict(test[FEATURE_COLS]), 0, None)

        rows.append({
            "fold": fold,
            "test_week": test_week,
            "n_test_rows": len(test),
            "mae": mean_absolute_error(test["working_hours"], preds),
            "rmse": np.sqrt(mean_squared_error(test["working_hours"], preds)),
        })

    if not rows:
        raise ValueError("No valid folds produced -- check week count vs. n_folds/min_train_weeks.")

    fold_metrics = pd.DataFrame(rows)
    return ValidationResult(fold_metrics, fold_metrics["mae"].mean(), fold_metrics["rmse"].mean())


def fit_final_lightgbm(features_df: pd.DataFrame) -> lgb.LGBMRegressor:
    """Fit on all available data -- this is the deployed model."""
    train = features_df.dropna(subset=FEATURE_COLS)
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(train[FEATURE_COLS], train["working_hours"],
               categorical_feature=["site_id", "equipment_type"])
    return model


def forecast_lightgbm(model: lgb.LGBMRegressor, features_df: pd.DataFrame, horizon: int = 8) -> pd.DataFrame:
    """Recursive multi-step forecast: each predicted week feeds the next week's lag features."""
    history = features_df[["site_id", "equipment_type", "week", "working_hours"]].copy()
    last_week = features_df["week"].max()
    future_weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    lag_numbers = [1, 2, 3, 4, 8]

    all_rows = []
    for future_week in future_weeks:
        step_rows = []
        for (site, etype), g in history.groupby(["site_id", "equipment_type"]):
            vals = g.sort_values("week")["working_hours"].tolist()
            row = {"site_id": site, "equipment_type": etype, "week": future_week}
            for lag in lag_numbers:
                row[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else 0.0
            row["roll_mean_4"] = np.mean(vals[-4:]) if vals else 0.0
            row["roll_mean_8"] = np.mean(vals[-8:]) if vals else 0.0
            row["month"] = future_week.month
            row["weekofyear"] = int(future_week.isocalendar().week)
            step_rows.append(row)

        step_df = pd.DataFrame(step_rows)
        step_df["site_id"] = step_df["site_id"].astype(pd.CategoricalDtype(history["site_id"].unique()))
        step_df["equipment_type"] = step_df["equipment_type"].astype(pd.CategoricalDtype(history["equipment_type"].unique()))

        preds = np.clip(model.predict(step_df[FEATURE_COLS]), 0, None)
        step_df["forecast"] = preds
        all_rows.append(step_df[["site_id", "equipment_type", "week", "forecast"]])

        new_hist = step_df[["site_id", "equipment_type", "week"]].copy()
        new_hist["working_hours"] = preds
        history = pd.concat([history, new_hist], ignore_index=True)

    return pd.concat(all_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 5. ARIMA (per site/type series, auto-selected order)
# --------------------------------------------------------------------------- #

def validate_arima(panel: pd.DataFrame, n_folds: int = 4, min_train_weeks: int = 20) -> ValidationResult:
    """
    Same walk-forward scheme as LightGBM, but one ARIMA model per (site, type)
    series -- ARIMA cannot pool across series the way the global LightGBM
    model does, so it must be refit separately for every combination, every fold.
    """
    weeks = sorted(panel["week"].unique())
    if len(weeks) < min_train_weeks + n_folds:
        raise ValueError(f"Only {len(weeks)} weeks available; need at least {min_train_weeks + n_folds}.")

    rows = []
    test_start = len(weeks) - n_folds
    for fold in range(n_folds):
        cutoff = test_start + fold
        train_weeks, test_week = weeks[:cutoff], weeks[cutoff]

        preds, actuals = [], []
        for (site, etype), g in panel.groupby(["site_id", "equipment_type"]):
            train_g = g[g["week"].isin(train_weeks)].sort_values("week")
            test_g = g[g["week"] == test_week]
            if train_g.empty or test_g.empty or len(train_g) < min_train_weeks:
                continue
            try:
                model = pm.auto_arima(
                    train_g["working_hours"], seasonal=False, stepwise=True,
                    suppress_warnings=True, error_action="ignore", max_p=3, max_q=3,
                )
                pred = max(0.0, model.predict(n_periods=1).iloc[0])
            except Exception:
                pred = train_g["working_hours"].iloc[-4:].mean()  # fallback if a fit fails
            preds.append(pred)
            actuals.append(test_g["working_hours"].iloc[0])

        if not actuals:
            continue
        rows.append({
            "fold": fold,
            "test_week": test_week,
            "n_test_rows": len(actuals),
            "mae": mean_absolute_error(actuals, preds),
            "rmse": np.sqrt(mean_squared_error(actuals, preds)),
        })

    if not rows:
        raise ValueError("No valid ARIMA folds produced -- check week count vs. n_folds/min_train_weeks.")

    fold_metrics = pd.DataFrame(rows)
    return ValidationResult(fold_metrics, fold_metrics["mae"].mean(), fold_metrics["rmse"].mean())


def fit_final_arima_models(panel: pd.DataFrame) -> dict[tuple[str, str], object]:
    """
    Fit one ARIMA model per (site, type) on all available data. Returns a dict
    keyed by (site_id, equipment_type) -> fitted pmdarima model (or None if
    fitting failed, in which case forecast_arima falls back to a simple mean).
    """
    models = {}
    for (site, etype), g in panel.groupby(["site_id", "equipment_type"]):
        series = g.sort_values("week")["working_hours"]
        try:
            models[(site, etype)] = pm.auto_arima(
                series, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore", max_p=3, max_q=3,
            )
        except Exception:
            models[(site, etype)] = None
    return models


def forecast_arima(panel: pd.DataFrame, models: dict[tuple[str, str], object], horizon: int = 8) -> pd.DataFrame:
    """Forecast forward using already-fitted per-(site, type) ARIMA models."""
    rows = []
    for (site, etype), g in panel.groupby(["site_id", "equipment_type"]):
        series = g.sort_values("week")["working_hours"]
        last_week = g["week"].max()
        future_weeks = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
        model = models.get((site, etype))
        if model is not None:
            preds = np.clip(model.predict(n_periods=horizon).to_numpy(), 0, None)
        else:
            preds = np.full(horizon, series.iloc[-4:].mean())  # fallback if fitting failed
        for wk, p in zip(future_weeks, preds):
            rows.append({"site_id": site, "equipment_type": etype, "week": wk, "forecast": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #

def run_pipeline(db_path: str = "equipment_rental.db", n_folds: int = 4, horizon: int = 8) -> dict:
    rentals = load_joined_rental_data(db_path)
    panel = build_demand_panel(rentals)
    features_df = add_features(panel)

    n_weeks = features_df["week"].nunique()
    min_train_weeks = max(12, n_weeks // 3)
    if n_weeks < min_train_weeks + n_folds:
        raise ValueError(
            f"Only {n_weeks} weeks of history. Need at least {min_train_weeks + n_folds} "
            f"for {n_folds} honest folds -- reduce n_folds or provide more data."
        )

    # Simple sparsity check: if almost every (site, type, week) row is zero,
    # a low MAE just means "the model correctly guessed zero", not that it
    # learned anything -- results wouldn't be trustworthy either way.
    nonzero_share = (panel["working_hours"] > 0).mean()
    if nonzero_share < 0.10:
        raise ValueError(
            f"Only {nonzero_share:.1%} of (site, type, week) rows have any working "
            f"hours at all -- this data is too sparse for a meaningful forecast. "
            f"Provide more history or aggregate to a coarser grain."
        )

    lgbm_val = validate_lightgbm(features_df, n_folds=n_folds, min_train_weeks=min_train_weeks)
    arima_val = validate_arima(panel, n_folds=n_folds, min_train_weeks=min_train_weeks)

    lgbm_model = fit_final_lightgbm(features_df)
    lgbm_forecast = forecast_lightgbm(lgbm_model, features_df, horizon=horizon)

    arima_models = fit_final_arima_models(panel)
    arima_forecast = forecast_arima(panel, arima_models, horizon=horizon)

    return {
        "panel": panel,
        "features_df": features_df,
        "lgbm_validation": lgbm_val,
        "arima_validation": arima_val,
        "lgbm_model": lgbm_model,
        "arima_models": arima_models,
        "lgbm_forecast": lgbm_forecast,
        "arima_forecast": arima_forecast,
    }


def print_pipeline_summary(results: dict) -> None:
    lgbm_val = results["lgbm_validation"]
    arima_val = results["arima_validation"]

    print("=== LightGBM: walk-forward validation (out-of-sample, working hours) ===")
    print(lgbm_val.fold_metrics.to_string(index=False))
    print(f"Mean MAE: {lgbm_val.mean_mae:.3f}   Mean RMSE: {lgbm_val.mean_rmse:.3f}")

    print("\n=== ARIMA: walk-forward validation (out-of-sample, working hours) ===")
    print(arima_val.fold_metrics.to_string(index=False))
    print(f"Mean MAE: {arima_val.mean_mae:.3f}   Mean RMSE: {arima_val.mean_rmse:.3f}")

    print("\n=== Comparison ===")
    better = "LightGBM" if lgbm_val.mean_mae < arima_val.mean_mae else "ARIMA"
    print(f"{better} has lower out-of-sample MAE "
          f"(LightGBM={lgbm_val.mean_mae:.3f} vs ARIMA={arima_val.mean_mae:.3f}).")

    print("\n=== LightGBM forecast, next weeks (first 15 rows) ===")
    print(results["lgbm_forecast"].head(15).to_string(index=False))

    print("\n=== ARIMA forecast, next weeks (first 15 rows) ===")
    arima_forecast = (
        results["arima_forecast"]
        .sort_values(["week", "site_id", "equipment_type"])
    )
    print(arima_forecast.head(15).to_string(index=False))


# --------------------------------------------------------------------------- #
# 7. Save / load models for later inference
# --------------------------------------------------------------------------- #

import json
from pathlib import Path
import joblib


def save_models(
    lgbm_model: lgb.LGBMRegressor,
    arima_models: dict[tuple[str, str], object],
    out_dir: str = "saved_models",
) -> None:
    """
    Save the final LightGBM model and the full set of per-(site, type) ARIMA
    models to disk, so both can be reloaded later without refitting.

    Files written to out_dir:
      - lgbm_model.joblib   -- the fitted LGBMRegressor
      - arima_models.joblib -- dict {(site_id, equipment_type): fitted pmdarima model or None}
      - feature_cols.json   -- the exact feature column list/order LightGBM was trained on,
                                so inference code always matches training exactly
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(lgbm_model, out / "lgbm_model.joblib")
    joblib.dump(arima_models, out / "arima_models.joblib")
    with open(out / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"Saved LightGBM model, {len(arima_models)} ARIMA models, and feature list to '{out_dir}/'.")


def load_models(out_dir: str = "saved_models") -> tuple[lgb.LGBMRegressor, dict[tuple[str, str], object], list[str]]:
    """Load back everything saved by save_models(). Returns (lgbm_model, arima_models, feature_cols)."""
    out = Path(out_dir)
    lgbm_model = joblib.load(out / "lgbm_model.joblib")
    arima_models = joblib.load(out / "arima_models.joblib")
    with open(out / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return lgbm_model, arima_models, feature_cols


def load_or_train_forecast_models(
    db_path: str = "equipment_rental.db",
    horizon: int = 8,
    out_dir: str = "saved_models",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved demand models if present; otherwise train and save them."""
    rentals = load_joined_rental_data(db_path)
    panel = build_demand_panel(rentals)
    features_df = add_features(panel)

    try:
        lgbm_model, arima_models, _ = load_models(out_dir)
    except FileNotFoundError:
        results = run_pipeline(db_path=db_path, n_folds=1, horizon=horizon)
        save_models(results["lgbm_model"], results["arima_models"], out_dir=out_dir)
        lgbm_model, arima_models, _ = load_models(out_dir)

    lgbm_forecast = forecast_lightgbm(lgbm_model, features_df, horizon=horizon)
    arima_forecast = forecast_arima(panel, arima_models, horizon=horizon)
    return panel, features_df, lgbm_forecast, arima_forecast


if __name__ == "__main__":
    results = run_pipeline(db_path="equipment_rental_large.db", n_folds=1)
    print_pipeline_summary(results)

    save_models(results["lgbm_model"], results["arima_models"])

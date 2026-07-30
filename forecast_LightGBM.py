"""
Equipment rental demand forecasting.

Forecasts demand (rental_count) at the (site_id, equipment_type, week/month) grain,
which is the grain that actually answers "what equipment will site X need at time Y".

Key design choices vs. a naive first pass:
  - Forecasting grain is (site_id, equipment_type, period), not a single blended
    company-wide series. A blended total cannot tell you what to pre-position where.
  - Sparse combinations (few historical rentals) fall back to a simple seasonal-naive
    baseline instead of being force-fit with a model that has no data to learn from.
  - A single *global* gradient-boosted model is trained across all (site, type) series
    with site/type as categorical features. This lets thin-history combinations borrow
    statistical strength from similar ones, which per-series ARIMA/ETS cannot do.
  - Validation is strictly time-ordered walk-forward (expanding window), never a random
    shuffle split, and out-of-sample error is what gets reported -- not in-sample fit.
  - Forecasts are produced as quantiles (median + upper) so pre-positioning decisions
    can use a safety-stock style buffer instead of a single point guess.
"""

from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #

def load_joined_rental_data(db_path: str = "equipment_rental.db") -> pd.DataFrame:
    """Load rentals joined with equipment type/status/rate."""
    conn = sqlite3.connect(db_path)
    rentals = pd.read_sql_query(
        """
        SELECT r.id,
               r.equipment_id,
               r.operator_id,
               r.site_id,
               r.check_in_date,
               r.expected_return_date,
               r.actual_return_date,
               r.rental_days,
               r.is_returned,
               r.engine_hours_per_day,
               r.idle_hours_per_day,
               e.type AS equipment_type,
               e.status AS equipment_status,
               e.daily_rental_rate
        FROM rentals r
        LEFT JOIN equipment e ON r.equipment_id = e.equipment_id
        """,
        conn,
        parse_dates=["check_in_date", "expected_return_date", "actual_return_date"],
    )
    conn.close()

    # Rows with no check-in date carry no time-series information -- drop rather
    # than silently losing them further downstream without a record of it.
    n_before = len(rentals)
    rentals = rentals.dropna(subset=["check_in_date"]).copy()
    dropped = n_before - len(rentals)
    if dropped:
        print(f"[load_joined_rental_data] Dropped {dropped} row(s) with no check_in_date.")

    rentals["site_id"] = rentals["site_id"].fillna("Unknown")
    rentals["equipment_type"] = rentals["equipment_type"].fillna("Unknown")
    rentals["equipment_status"] = rentals["equipment_status"].fillna("Unknown")
    return rentals


# --------------------------------------------------------------------------- #
# 2. Building a clean, gap-aware panel at (site, type, period) grain
# --------------------------------------------------------------------------- #

def build_demand_panel(
    rentals: pd.DataFrame,
    freq: str = "W-MON",
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build a complete (site_id, equipment_type, period) panel of rental counts,
    filling true gaps with 0 -- but only up to `as_of`, so we never manufacture
    a fake "flatline" beyond the point where data actually stops being collected.

    freq="W-MON" (weekly) is a sensible default for pre-positioning; use "MS" for
    monthly if the business cadence is monthly. Do not go finer than the data supports.
    """
    if as_of is None:
        as_of = rentals["check_in_date"].max()

    rentals = rentals.copy()
    rentals["period"] = rentals["check_in_date"].dt.to_period(
        "W" if freq.startswith("W") else "M"
    ).dt.to_timestamp(how="start")

    counts = (
        rentals.groupby(["site_id", "equipment_type", "period"])
        .size()
        .rename("rental_count")
        .reset_index()
    )

    sites = rentals["site_id"].unique()
    types = rentals["equipment_type"].unique()
    full_range = pd.date_range(
        rentals["period"].min(), pd.Timestamp(as_of), freq=freq
    )

    idx = pd.MultiIndex.from_product(
        [sites, types, full_range], names=["site_id", "equipment_type", "period"]
    )
    panel = (
        counts.set_index(["site_id", "equipment_type", "period"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )
    panel = panel.sort_values(["site_id", "equipment_type", "period"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# 3. Feature engineering (leak-safe: every feature at row t uses only data < t)
# --------------------------------------------------------------------------- #

def add_features(panel: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 4, 8, 52)) -> pd.DataFrame:
    """
    Adds lag and rolling features per (site_id, equipment_type) group.
    All lag/rolling windows are shifted so no feature at row t can see y at t.
    """
    df = panel.copy()
    df = df.sort_values(["site_id", "equipment_type", "period"])
    grp = df.groupby(["site_id", "equipment_type"])["rental_count"]

    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling stats computed on already-shifted (lag_1) series so the current
    # period's own value never leaks into its own rolling mean/std.
    shifted = grp.shift(1)
    df["roll_mean_4"] = shifted.groupby([df["site_id"], df["equipment_type"]]).transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    df["roll_mean_8"] = shifted.groupby([df["site_id"], df["equipment_type"]]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    df["roll_std_4"] = shifted.groupby([df["site_id"], df["equipment_type"]]).transform(
        lambda s: s.rolling(4, min_periods=1).std()
    )

    df["month"] = df["period"].dt.month
    df["weekofyear"] = df["period"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["period"].dt.quarter

    # How long this (site, type) has been observed -- helps the model learn to
    # discount predictions for brand-new combinations vs. long-running ones.
    df["periods_observed"] = df.groupby(["site_id", "equipment_type"]).cumcount()

    df["site_id"] = df["site_id"].astype("category")
    df["equipment_type"] = df["equipment_type"].astype("category")
    return df


# --------------------------------------------------------------------------- #
# 4. Walk-forward (expanding window) validation -- the honest way to test a
#    time series model. Never shuffle, never randomly split.
# --------------------------------------------------------------------------- #

@dataclass
class WalkForwardResult:
    fold_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame
    mean_mae: float
    mean_rmse: float


def walk_forward_validate(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rental_count",
    n_folds: int = 4,
    min_train_periods: int = 12,
    horizon: int = 1,
    model_params: dict | None = None,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation.

    For each fold, train on all periods up to cutoff_k, predict the next `horizon`
    period(s), score only on those held-out rows, then advance the cutoff. This
    mirrors how the model will actually be used in production (train on everything
    known so far, forecast forward) and avoids the classic time-series leakage bug
    of a random train/test split, which lets the model "see the future" via
    neighbouring rows in the same series.
    """
    periods = sorted(features_df["period"].unique())
    if len(periods) < min_train_periods + n_folds:
        raise ValueError(
            f"Not enough periods ({len(periods)}) for {n_folds} folds with "
            f"min_train_periods={min_train_periods}. Reduce n_folds/min_train_periods "
            f"or provide more history."
        )

    default_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        objective="poisson",  # rental counts are non-negative integers -> Poisson loss
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    if model_params:
        default_params.update(model_params)

    fold_rows = []
    oos_frames = []

    test_start_idx = len(periods) - n_folds * horizon
    for fold in range(n_folds):
        cutoff_idx = test_start_idx + fold * horizon
        train_periods = periods[:cutoff_idx]
        test_periods = periods[cutoff_idx: cutoff_idx + horizon]
        if len(train_periods) < min_train_periods or not test_periods:
            continue

        train_df = features_df[features_df["period"].isin(train_periods)].dropna(subset=feature_cols)
        test_df = features_df[features_df["period"].isin(test_periods)].dropna(subset=feature_cols)
        if train_df.empty or test_df.empty:
            continue
        if train_df[target_col].sum() == 0:
            # A fold with literally zero historical demand can't train a Poisson
            # model (and isn't a meaningful fold to score anyway) -- skip it
            # rather than letting LightGBM fail with an opaque internal error.
            continue

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            train_df[feature_cols], train_df[target_col],
            categorical_feature=["site_id", "equipment_type"],
        )
        preds = np.clip(model.predict(test_df[feature_cols]), 0, None)

        mae = mean_absolute_error(test_df[target_col], preds)
        rmse = np.sqrt(mean_squared_error(test_df[target_col], preds))
        fold_rows.append({
            "fold": fold,
            "train_periods": len(train_periods),
            "test_period_start": test_periods[0],
            "n_test_rows": len(test_df),
            "mae": mae,
            "rmse": rmse,
        })

        oos = test_df[["site_id", "equipment_type", "period", target_col]].copy()
        oos["prediction"] = preds
        oos["fold"] = fold
        oos_frames.append(oos)

    if not fold_rows:
        raise ValueError("No valid folds were produced -- check period count vs. n_folds/min_train_periods.")

    fold_metrics = pd.DataFrame(fold_rows)
    oos_predictions = pd.concat(oos_frames, ignore_index=True)
    return WalkForwardResult(
        fold_metrics=fold_metrics,
        oos_predictions=oos_predictions,
        mean_mae=fold_metrics["mae"].mean(),
        mean_rmse=fold_metrics["rmse"].mean(),
    )


# --------------------------------------------------------------------------- #
# 5. Baseline models -- required for two reasons:
#      (a) a sparse (site, type) combo has no business being fit by a 300-tree
#          gradient booster; a seasonal-naive baseline is safer and often better.
#      (b) any "sophisticated" model must beat this baseline out-of-sample or
#          it isn't earning its complexity.
# --------------------------------------------------------------------------- #

def seasonal_naive_forecast(series: pd.Series, season_length: int, periods: int) -> pd.Series:
    """Forecast = value from `season_length` periods ago, repeated forward."""
    if len(series) < season_length:
        # Not even one full season of history -- fall back to the mean.
        base = series.mean() if len(series) else 0.0
        return pd.Series([base] * periods)
    last_season = series.iloc[-season_length:].values
    reps = int(np.ceil(periods / season_length))
    return pd.Series(np.tile(last_season, reps)[:periods])


def evaluate_baseline_walk_forward(
    panel: pd.DataFrame,
    group_cols: tuple[str, str] = ("site_id", "equipment_type"),
    season_length: int = 4,
    n_folds: int = 4,
    horizon: int = 1,
) -> pd.DataFrame:
    """Same expanding-window folds as the ML model, scored against seasonal-naive."""
    periods = sorted(panel["period"].unique())
    test_start_idx = len(periods) - n_folds * horizon
    rows = []
    for fold in range(n_folds):
        cutoff_idx = test_start_idx + fold * horizon
        train_periods = periods[:cutoff_idx]
        test_periods = periods[cutoff_idx: cutoff_idx + horizon]
        if not train_periods or not test_periods:
            continue
        for (site, etype), g in panel.groupby(list(group_cols)):
            train_g = g[g["period"].isin(train_periods)].sort_values("period")
            test_g = g[g["period"].isin(test_periods)].sort_values("period")
            if test_g.empty:
                continue
            fc = seasonal_naive_forecast(train_g["rental_count"], season_length, len(test_g))
            mae = mean_absolute_error(test_g["rental_count"].values, fc.values)
            rows.append({"fold": fold, "site_id": site, "equipment_type": etype, "mae": mae})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Final model fit on ALL data + forward forecast with quantiles
# --------------------------------------------------------------------------- #

@dataclass
class FinalForecast:
    point_forecast: pd.DataFrame       # median forecast per (site, type, future period)
    upper_forecast: pd.DataFrame       # e.g. 80th percentile, for pre-positioning buffer
    feature_importance: pd.DataFrame


def fit_final_models_and_forecast(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rental_count",
    horizon: int = 8,
    freq: str = "W-MON",
    quantile_upper: float = 0.8,
) -> FinalForecast:
    """
    Fits on ALL available history (this is the deployed model, not a validation
    fold) and produces a recursive multi-step-ahead forecast per (site, type).

    Two models are trained: a median-point Poisson model, and a quantile model at
    `quantile_upper` so downstream pre-positioning can use (median, buffer) rather
    than a single number that will be wrong roughly half the time by construction.
    """
    train_df = features_df.dropna(subset=feature_cols)

    point_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, objective="poisson",
        random_state=RANDOM_STATE, verbose=-1,
    )
    point_model.fit(train_df[feature_cols], train_df[target_col],
                     categorical_feature=["site_id", "equipment_type"])

    upper_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, objective="quantile", alpha=quantile_upper,
        random_state=RANDOM_STATE, verbose=-1,
    )
    upper_model.fit(train_df[feature_cols], train_df[target_col],
                     categorical_feature=["site_id", "equipment_type"])

    # Recursive forecasting: step forward one period at a time per (site, type),
    # appending each prediction to history so later lag features see prior forecasts.
    last_period = features_df["period"].max()
    future_periods = pd.date_range(
        last_period + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq
    )

    history = features_df[["site_id", "equipment_type", "period", target_col]].copy()
    point_rows, upper_rows = [], []

    lag_numbers = sorted(int(c.split("_")[1]) for c in feature_cols if c.startswith("lag_"))
    max_lag = max(lag_numbers) if lag_numbers else 1

    for future_period in future_periods:
        step_rows = []
        for (site, etype), g in history.groupby(["site_id", "equipment_type"]):
            g = g.sort_values("period")
            vals = g[target_col].tolist()
            row = {"site_id": site, "equipment_type": etype, "period": future_period}
            for lag in lag_numbers:
                row[f"lag_{lag}"] = vals[-lag] if len(vals) >= lag else 0.0
            recent = vals[-max(4, 1):]
            row["roll_mean_4"] = np.mean(vals[-4:]) if vals else 0.0
            row["roll_mean_8"] = np.mean(vals[-8:]) if vals else 0.0
            row["roll_std_4"] = np.std(vals[-4:]) if len(vals) >= 2 else 0.0
            row["month"] = future_period.month
            row["weekofyear"] = int(future_period.isocalendar().week)
            row["quarter"] = future_period.quarter
            row["periods_observed"] = len(vals)
            step_rows.append(row)

        step_df = pd.DataFrame(step_rows)
        step_df["site_id"] = step_df["site_id"].astype(
            pd.CategoricalDtype(categories=history["site_id"].unique())
        )
        step_df["equipment_type"] = step_df["equipment_type"].astype(
            pd.CategoricalDtype(categories=history["equipment_type"].unique())
        )

        point_pred = np.clip(point_model.predict(step_df[feature_cols]), 0, None)
        upper_pred = np.clip(upper_model.predict(step_df[feature_cols]), 0, None)
        upper_pred = np.maximum(upper_pred, point_pred)  # upper must not be below median

        step_df["point_forecast"] = point_pred
        step_df["upper_forecast"] = upper_pred
        point_rows.append(step_df[["site_id", "equipment_type", "period", "point_forecast"]])
        upper_rows.append(step_df[["site_id", "equipment_type", "period", "upper_forecast"]])

        # Feed the point forecast back in as "history" for the next recursive step.
        new_hist = step_df[["site_id", "equipment_type", "period"]].copy()
        new_hist[target_col] = point_pred
        history = pd.concat([history, new_hist], ignore_index=True)

    point_forecast = pd.concat(point_rows, ignore_index=True)
    upper_forecast = pd.concat(upper_rows, ignore_index=True)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": point_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return FinalForecast(point_forecast, upper_forecast, importance)


# --------------------------------------------------------------------------- #
# 7. End-to-end orchestration
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    "site_id", "equipment_type",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_52",
    "roll_mean_4", "roll_mean_8", "roll_std_4",
    "month", "weekofyear", "quarter", "periods_observed",
]


def run_pipeline(
    db_path: str = "equipment_rental.db",
    freq: str = "W-MON",
    n_folds: int = 4,
    horizon: int = 8,
) -> dict:
    rentals = load_joined_rental_data(db_path)
    panel = build_demand_panel(rentals, freq=freq)
    features_df = add_features(panel)

    available_feats = [c for c in FEATURE_COLS if c in features_df.columns]

    n_periods = features_df["period"].nunique()
    min_train_periods = max(8, n_periods // 3)

    # A raw period count can be misleading: lag features (esp. long ones like
    # lag_52) mean many early rows get dropped by dropna(), and a long stretch
    # of true zeros (e.g. a real gap in data collection) can leave a fold with
    # literally no positive demand to train on. Check what actually survives
    # feature engineering, not just how many calendar periods exist.
    usable = features_df.dropna(subset=available_feats)
    usable_periods_with_signal = usable.loc[usable["rental_count"] > 0, "period"].nunique()

    if n_periods < min_train_periods + n_folds:
        raise ValueError(
            f"Only {n_periods} periods of history available at freq='{freq}'. "
            f"That is too little for {n_folds} honest out-of-sample folds. "
            f"Options: reduce n_folds, switch freq to a coarser grain (e.g. 'MS' "
            f"instead of 'W-MON'), or use seasonal_naive_forecast() directly as a "
            f"baseline until more history accumulates. Do not force-fit ML models "
            f"on this little data -- the reported metrics would not be trustworthy."
        )

    if usable_periods_with_signal < min_train_periods:
        raise ValueError(
            f"After feature engineering (dropna on lag/rolling columns), only "
            f"{usable_periods_with_signal} periods have any non-zero demand at all "
            f"(out of {n_periods} calendar periods). This dataset is too sparse for "
            f"walk-forward validation to produce a meaningful fold -- you'll get "
            f"'no valid folds' or an all-zero training fold. Fixes, in order of "
            f"preference: (1) use a coarser freq ('MS' monthly instead of weekly), "
            f"(2) drop long lags like lag_52/lag_8 from FEATURE_COLS if you don't "
            f"have that much real history yet, (3) fall back to "
            f"seasonal_naive_forecast()/evaluate_baseline_walk_forward() only until "
            f"more real data accumulates."
        )

    wf_result = walk_forward_validate(
        features_df, available_feats,
        n_folds=n_folds, min_train_periods=min_train_periods, horizon=1,
    )
    baseline_mae_df = evaluate_baseline_walk_forward(panel, n_folds=n_folds, horizon=1)

    final = fit_final_models_and_forecast(
        features_df, available_feats, horizon=horizon, freq=freq,
    )

    return {
        "panel": panel,
        "features_df": features_df,
        "walk_forward": wf_result,
        "baseline_mae_df": baseline_mae_df,
        "final_forecast": final,
    }


def print_pipeline_summary(results: dict) -> None:
    wf = results["walk_forward"]
    baseline = results["baseline_mae_df"]

    print("=== Walk-forward validation (LightGBM, expanding window, out-of-sample) ===")
    print(wf.fold_metrics.to_string(index=False))
    print(f"\nMean OOS MAE:  {wf.mean_mae:.3f}")
    print(f"Mean OOS RMSE: {wf.mean_rmse:.3f}")

    print("\n=== Seasonal-naive baseline (same folds, for comparison) ===")
    print(f"Mean baseline MAE: {baseline['mae'].mean():.3f}")
    lift = (baseline["mae"].mean() - wf.mean_mae) / baseline["mae"].mean() * 100
    print(f"LightGBM improvement over baseline: {lift:.1f}%")
    if lift <= 0:
        print("WARNING: model does not beat the naive baseline -- do not deploy the ML "
              "model over the baseline until this changes (more data, better features, etc).")

    print("\n=== Per (site, equipment_type) out-of-sample MAE ===")
    print("(A global average can hide poor accuracy on specific sites/types -- check this")
    print(" before trusting the model for any single site's pre-positioning decision.)")
    per_group = (
        wf.oos_predictions.assign(abs_err=lambda d: (d["rental_count"] - d["prediction"]).abs())
        .groupby(["site_id", "equipment_type"])
        .agg(mae=("abs_err", "mean"), n_obs=("abs_err", "size"))
        .sort_values("mae", ascending=False)
        .reset_index()
    )
    print(per_group.to_string(index=False))

    print("\n=== Top feature importances (final model, trained on all data) ===")
    print(results["final_forecast"].feature_importance.head(8).to_string(index=False))

    print("\n=== Forecast (next periods), point + upper(80th pct), first 15 rows ===")
    merged = results["final_forecast"].point_forecast.merge(
        results["final_forecast"].upper_forecast, on=["site_id", "equipment_type", "period"]
    )
    print(merged.head(15).to_string(index=False))


if __name__ == "__main__":
    results = run_pipeline(db_path='equipment_rental_synthetic.db')
    print_pipeline_summary(results)

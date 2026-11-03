"""Reusable keyed forecast evaluation for independent experiment notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .forecasting import ForecastModel, ForecastTask, PREDICTION_COLUMN, validate_frame

Metric = Callable[[np.ndarray, np.ndarray], float]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error: observed minus predicted."""
    return float(np.mean(actual - predicted))


DEFAULT_METRICS: Mapping[str, Metric] = {"mae": mae, "rmse": rmse, "bias": bias}


@dataclass
class EvaluationResult:
    """Metrics and row-level predictions; null actuals are unscored."""

    metrics: dict[str, float | int]
    predictions: pd.DataFrame


def evaluate_forecasts(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    task: ForecastTask,
    *,
    metrics: Mapping[str, Metric] | None = None,
) -> EvaluationResult:
    """Join by series/time keys, never by row position.

    Every actual row must have exactly one forecast and vice versa. Missing
    actuals are retained as unscored; observed zero counts are scored normally.
    No automatic imputation of missing UK BBS observations is performed.
    """
    validate_frame(actuals, task, labelled=True)
    validate_frame(forecasts, task, labelled=False)
    if PREDICTION_COLUMN not in forecasts.columns:
        raise ValueError(f"Forecasts must include '{PREDICTION_COLUMN}'")
    if len(actuals) != len(forecasts):
        raise ValueError("Actual and forecast key sets differ")
    aligned = actuals[[*task.keys, task.target]].merge(
        forecasts[[*task.keys, PREDICTION_COLUMN]],
        on=task.keys, how="left", sort=False, validate="one_to_one", indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise ValueError("Actual and forecast key sets differ")
    aligned = aligned.drop(columns="_merge")

    actual = pd.to_numeric(aligned[task.target], errors="raise").to_numpy(dtype=float)
    predicted = pd.to_numeric(aligned[PREDICTION_COLUMN], errors="raise").to_numpy(dtype=float)
    mask = ~np.isnan(actual)
    if not np.isfinite(predicted).all() or not np.isfinite(actual[mask]).all():
        raise ValueError("Predictions and observed targets must be finite")
    if not mask.any():
        raise ValueError("No observed test targets to score")

    aligned[task.target] = actual
    aligned[PREDICTION_COLUMN] = predicted
    aligned["residual"] = actual - predicted  # null actual -> NaN residual
    aligned["absolute_error"] = np.abs(aligned["residual"])
    selected_metrics = DEFAULT_METRICS if metrics is None else metrics
    scores: dict[str, float | int] = {
        "n_scored": int(mask.sum()), "n_unscored": int((~mask).sum())
    }
    for name, metric in selected_metrics.items():
        if name in scores:
            raise ValueError(f"Reserved metric name: {name}")
        value = float(metric(actual[mask], predicted[mask]))
        if not np.isfinite(value):
            raise ValueError(f"Metric {name!r} produced a nonfinite value")
        scores[name] = value
    return EvaluationResult(metrics=scores, predictions=aligned)


def run_forecast(
    model: ForecastModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    task: ForecastTask,
    *,
    metrics: Mapping[str, Metric] | None = None,
) -> EvaluationResult:
    """Fit any compatible model, forecast without test labels, and evaluate.

    Other test columns (such as future-known metadata) are passed to the model.
    It is the notebook's responsibility to exclude future-unknown covariates.
    """
    validate_frame(train, task, labelled=True)
    validate_frame(test, task, labelled=True)
    try:
        if not train[task.time_col].max() < test[task.time_col].min():
            raise ValueError("Training timestamps must strictly precede test timestamps")
    except TypeError as exc:
        raise ValueError("Training and test timestamps must be comparable") from exc
    model.fit(train.copy(), task)
    future = test.drop(columns=task.target).copy()
    predictions = model.predict(future)
    return evaluate_forecasts(test, predictions, task, metrics=metrics)

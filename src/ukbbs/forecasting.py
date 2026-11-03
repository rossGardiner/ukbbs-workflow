"""Small, library-independent interface for notebook forecasting experiments.

A model implements ``fit(train, task)`` and ``predict(future)``. Predictions
must be keyed to the requested future rows and contain a ``prediction`` column.
Models can use any other columns (e.g. known-in-advance metadata) they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import pandas as pd

from .preprocessing import COUNT_COLUMN

PREDICTION_COLUMN = "prediction"


@dataclass(frozen=True)
class ForecastTask:
    """Column conventions shared by a notebook's models and evaluator.

    ``group_cols=()`` describes a single aggregate time series. Keep genuine
    non-detections (observed zeros) distinct from unobserved square-years.
    """

    target: str = COUNT_COLUMN
    time_col: str = "year"
    group_cols: tuple[str, ...] = ("OS_GRID_REF", "species")

    def __post_init__(self) -> None:
        columns = (*self.group_cols, self.time_col, self.target)
        if any(not isinstance(name, str) or not name for name in columns):
            raise ValueError("Task columns must be non-empty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("Task columns must be distinct")
        if any(name in {PREDICTION_COLUMN, "residual", "absolute_error"} for name in columns):
            raise ValueError("Task columns cannot use reserved evaluation column names")

    @property
    def keys(self) -> list[str]:
        return [*self.group_cols, self.time_col]


class ForecastModel(Protocol):
    """Implement this contract to use a model with ``run_forecast``.

    ``future`` includes keys and optional features, but *never* the target.
    Return one row per future key with those keys and ``prediction``.
    """

    def fit(self, train: pd.DataFrame, task: ForecastTask) -> Self: ...

    def predict(self, future: pd.DataFrame) -> pd.DataFrame: ...


def validate_frame(frame: pd.DataFrame, task: ForecastTask, *, labelled: bool) -> None:
    """Check keys and time representation; do not manufacture missing rows."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    required = task.keys + ([task.target] if labelled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Forecast data must not be empty")
    if frame[task.keys].isna().any().any():
        raise ValueError("Forecast keys cannot be missing")
    if frame.duplicated(task.keys).any():
        raise ValueError(f"Duplicate forecast keys: {task.keys}")
    times = frame[task.time_col]
    if not (pd.api.types.is_numeric_dtype(times) or pd.api.types.is_datetime64_any_dtype(times)):
        raise ValueError(f"{task.time_col} must be numeric or datetime, not strings")
    if pd.api.types.is_numeric_dtype(times) and not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError(f"{task.time_col} must contain finite values")


def chronological_split(
    observations: pd.DataFrame, *, train_through: int | float | pd.Timestamp,
    task: ForecastTask,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train through an inclusive cutoff; test only strictly later periods.

    The split does NOT fill years, construct zeros, or remove unknown groups.
    Models decide whether they can forecast previously unseen groups.
    """
    validate_frame(observations, task, labelled=True)
    try:
        mask = observations[task.time_col].le(train_through)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_through must be comparable to the time column") from exc
    train, test = observations.loc[mask].copy(), observations.loc[~mask].copy()
    if train.empty or test.empty:
        raise ValueError("The temporal split must leave both training and test rows")
    return train.reset_index(drop=True), test.reset_index(drop=True)


class LastValueForecaster:
    """Per-group naive baseline: repeat the last *observed* training target.

    A training null is not converted to zero. Unknown groups are rejected, not
    silently assigned a population-wide prediction. A future timestamp must
    be after the last observed timestamp for its group.
    """

    def __init__(self) -> None:
        self._task: ForecastTask | None = None
        self._last: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame, task: ForecastTask) -> Self:
        validate_frame(train, task, labelled=True)
        observed = train.loc[train[task.target].notna()].copy()
        if observed.empty:
            raise ValueError("Training data have no observed targets")
        values = pd.to_numeric(observed[task.target], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Training targets must be finite where observed")
        observed[task.target] = values
        ordered = observed.sort_values(task.time_col, kind="stable")
        last = (ordered.groupby(list(task.group_cols), sort=False).tail(1)
                if task.group_cols else ordered.tail(1))
        self._last = last[[*task.group_cols, task.time_col, task.target]].rename(
            columns={task.time_col: "_last_time", task.target: PREDICTION_COLUMN}
        ).reset_index(drop=True)
        self._task = task
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._task is None or self._last is None:
            raise RuntimeError("Call fit() before predict()")
        task = self._task
        validate_frame(future, task, labelled=False)
        if task.target in future.columns:
            raise ValueError("future must not contain the target (avoid leakage)")
        request = future[task.keys].copy()
        if task.group_cols:
            joined = request.merge(
                self._last, on=list(task.group_cols), how="left",
                sort=False, validate="many_to_one", indicator=True,
            )
            if joined["_merge"].ne("both").any():
                raise ValueError("No training history for one or more forecast groups")
        else:
            joined = request.assign(
                _last_time=self._last["_last_time"].iloc[0],
                prediction=self._last[PREDICTION_COLUMN].iloc[0],
            )
        try:
            if joined[task.time_col].le(joined["_last_time"]).any():
                raise ValueError("Future timestamps must follow each group's last observation")
        except TypeError as exc:
            raise ValueError("Training and future timestamps must be comparable") from exc
        return joined[[*task.keys, PREDICTION_COLUMN]].copy()

"""One-year autoregressive MLP for a single BBS species across survey squares.

Every observed square-year in the training period is a target. Inputs are the
*immediately preceding calendar year's* count and a missing-input indicator;
an absent earlier square-year is not a zero count. The MLP shares weights
across squares, so the number of input neurons does not depend on site count.

Prediction uses a fixed origin: after the final training year, forecasts are
fed back as inputs through every calendar year (including unobserved years).
No actual test counts are read by ``predict``.

Requires the optional PyTorch dependency (``pip install torch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from .forecasting import ForecastTask, PREDICTION_COLUMN, validate_frame


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("MLP forecasting requires PyTorch: python -m pip install torch") from exc
    return torch, nn


def _validate_years(frame: pd.DataFrame, column: str) -> np.ndarray:
    years = frame[column].to_numpy(dtype=float)
    if not np.isfinite(years).all() or (years != np.floor(years)).any():
        raise ValueError("OneYearMLPForecaster requires integer calendar years")
    return years.astype(np.int64)


@dataclass(frozen=True)
class MLPConfig:
    """Small, shared feed-forward model on log1p(count) and missingness."""

    hidden_sizes: tuple[int, ...] = (64, 32, 16)
    epochs: int = 150
    learning_rate: float = 0.001
    batch_size: int = 512
    weight_decay: float = 0.0001
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(type(n) is not int or n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must be nonempty positive integers")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


class OneYearMLPForecaster:
    """Train one MLP per species across many sites, then forecast recursively.

    Uses only last year's numerical count (log1p transformed) and an explicit
    missing indicator. Training loss is computed only for observed targets:
    missing target years are NEVER assigned zero. Missing previous-year inputs
    use log-count=0 plus missing=1, distinguishing them from a real zero.

    Future predictions after the first step are fed back as numerical input
    (missing=0, meaning a model-generated value is available). This is a
    recursive *fixed-origin* forecast, not rolling updating with observations.

    The forecast universe consists of squares observed at least once in train.
    Additional metadata / site coordinates are not used by this first model.
    """

    def __init__(self, config: MLPConfig | None = None) -> None:
        self.config = config or MLPConfig()
        self._task: ForecastTask | None = None
        self._species: object | None = None
        self._origin: int | None = None
        self._site_states: dict[object, tuple[float, bool]] | None = None
        self._network = None
        self.history_: list[float] = []
        self.n_training_pairs_: int = 0
        self.n_missing_previous_: int = 0

    def fit(self, train: pd.DataFrame, task: ForecastTask) -> Self:
        torch, nn = _torch()
        validate_frame(train, task, labelled=True)
        if tuple(task.group_cols) != ("OS_GRID_REF", "species"):
            raise ValueError('MLP requires task.group_cols=("OS_GRID_REF", "species")')
        train = train.copy()
        train[task.time_col] = _validate_years(train, task.time_col)
        species = train["species"].unique()
        if len(species) != 1:
            raise ValueError("Train the MLP on exactly one species at a time")
        valid = train[task.target].notna()
        vals = pd.to_numeric(train.loc[valid, task.target], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError("Observed bird counts must be finite and nonnegative")
        train = train.loc[valid, [*task.keys, task.target]].copy()
        train[task.target] = vals
        if train.empty:
            raise ValueError("No observed training targets")
        origin = int(_validate_years(train, task.time_col).max())
        earliest = int(train[task.time_col].min())
        if origin == earliest:
            raise ValueError("Need at least two training calendar years")

        # Merge by calendar year (not positional shift): absent 2001/2020 etc.
        # remains missing instead of silently using the preceding available year.
        targets = train.loc[train[task.time_col].gt(earliest)].copy()
        previous = train[["OS_GRID_REF", task.time_col, task.target]].copy()
        previous[task.time_col] += 1
        previous = previous.rename(columns={task.target: "_previous_count"})
        pairs = targets.merge(previous, how="left", on=["OS_GRID_REF", task.time_col],
                              sort=False, validate="one_to_one")
        if pairs.empty:
            raise ValueError("Need observed targets after the first training year")

        prior = pairs["_previous_count"].to_numpy(dtype=float)
        missing = np.isnan(prior)
        x = np.column_stack([
            np.log1p(np.where(missing, 0.0, prior)), missing.astype(float),
        ]).astype("float32")
        y = np.log1p(pairs[task.target].to_numpy(dtype=float)).astype("float32").reshape(-1, 1)

        # Fork avoids resetting the caller's global RNG state, while giving
        # reproducible initial weights as well as minibatch ordering.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            layers: list = []
            input_dim = 2
            for width in self.config.hidden_sizes:
                layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
                input_dim = width
            layers.append(nn.Linear(input_dim, 1))
            model = nn.Sequential(*layers)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                               weight_decay=self.config.weight_decay)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        gen = torch.Generator().manual_seed(self.config.seed)
        self.history_ = []
        model.train()
        for _ in range(self.config.epochs):
            indices = torch.randperm(len(x_tensor), generator=gen)
            total_loss = 0.0
            for batch in indices.split(self.config.batch_size):
                opt.zero_grad()
                estimate = model(x_tensor[batch])
                loss = nn.functional.mse_loss(estimate, y_tensor[batch])
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * len(batch)
            self.history_.append(total_loss / len(x_tensor))
        model.eval()

        # Exactly the origin year's observations become known initial state.
        # Other squares remain genuinely unobserved at that year.
        all_sites = train["OS_GRID_REF"].unique().tolist()
        present = train.loc[train[task.time_col].eq(origin), ["OS_GRID_REF", task.target]]
        last_counts = dict(zip(present["OS_GRID_REF"], present[task.target]))
        self._site_states = {
            site: (float(last_counts.get(site, 0.0)), site not in last_counts)
            for site in all_sites
        }
        self._task = task
        self._species = species[0]
        self._origin = origin
        self._network = model
        self.n_training_pairs_ = len(pairs)
        self.n_missing_previous_ = int(missing.sum())
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        if self._network is None or self._task is None or self._site_states is None:
            raise RuntimeError("Call fit() before predict()")
        task = self._task
        validate_frame(future, task, labelled=False)
        if task.target in future.columns:
            raise ValueError("future must not contain target counts (test leakage)")
        future = future.copy()
        future[task.time_col] = _validate_years(future, task.time_col)
        if not future["species"].eq(self._species).all():
            raise ValueError("Future must contain the trained species only")
        if future[task.time_col].le(self._origin).any():
            raise ValueError("Forecast years must be after the training origin")
        sites = future["OS_GRID_REF"].unique().tolist()
        unknown = set(sites) - set(self._site_states)
        if unknown:
            raise ValueError(f"Unknown grid squares absent from training: {len(unknown)}")
        # All future square-years are computed using only frozen train state and
        # our predictions. No future labels or future covariates are consulted.
        import torch
        current = np.array([self._site_states[s][0] for s in sites], dtype=float)
        missing = np.array([self._site_states[s][1] for s in sites], dtype=float)
        wanted_years = set(future[task.time_col].tolist())
        forecasts: dict[tuple[object, int], float] = {}
        with torch.no_grad():
            for year in range(self._origin + 1, int(future[task.time_col].max()) + 1):
                inputs = np.column_stack([np.log1p(current), missing]).astype("float32")
                log_pred = self._network(torch.from_numpy(inputs)).numpy().reshape(-1)
                # log1p(count) target. Counts cannot be negative.
                current = np.expm1(np.clip(log_pred, 0.0, 20.0))
                missing = np.zeros(len(sites), dtype=float)
                if year in wanted_years:
                    forecasts.update(((site, year), float(current[i])) for i, site in enumerate(sites))
        result = future[task.keys].copy()
        result[PREDICTION_COLUMN] = [
            forecasts[(site, int(year))]
            for site, year in zip(future["OS_GRID_REF"], future[task.time_col])
        ]
        return result

"""Select illustrative UKBBS squares by *training-period* observation coverage.

This module reads the original processed observations for plotting historical counts.
It does not affect the fair, matched-row error calculations in prediction_comparison.
An observed zero counts as a surveyed target; an absent or NaN target does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def train_cutoff_from_runs(runs: Sequence, override: int | None = None) -> int:
    """Require a single explicit fixed training cutoff for every plotted model.

    If manifests omit the cutoff, require an explicit override. Do not guess from
    the first forecast year: excluded or unpredicted years make that unreliable.
    """
    cutoffs = {}
    for run in runs:
        config = run.run_config
        raw = config.get("train_through", config.get("forecast_origin"))
        if raw is not None:
            value = int(raw)
            if value != float(raw):
                raise ValueError(f"Non-integral train cutoff in {run.directory}: {raw!r}")
            cutoffs[run.directory] = value

    if len(set(cutoffs.values())) > 1:
        raise ValueError(
            f"Selected models have different training cutoffs: {cutoffs}. "
            "Restrict MODEL_FOLDERS to comparable runs."
        )
    inferred = next(iter(cutoffs.values()), None)
    if override is not None:
        chosen = int(override)
        if chosen != override:
            raise ValueError("TRAIN_THROUGH must be a calendar year")
        if inferred is not None and inferred != chosen:
            raise ValueError(
                f"TRAIN_THROUGH={chosen} disagrees with saved training cutoff {inferred}."
            )
        return chosen
    if inferred is None:
        raise ValueError(
            "Cannot determine the training cutoff from the selected manifests. "
            "Set TRAIN_THROUGH to the last observed training year in the notebook."
        )
    if len(cutoffs) != len(runs):
        missing = [run.directory for run in runs if run.directory not in cutoffs]
        raise ValueError(
            f"Training cutoff missing from {missing}. Set TRAIN_THROUGH explicitly "
            "after checking those model experiments."
        )
    return inferred


def load_species_training_history(
    path: str | Path,
    *,
    species: str,
    train_through: int,
    target_col: str,
    time_col: str = "year",
    square_col: str = "OS_GRID_REF",
) -> pd.DataFrame:
    """Load all observed pre-cutoff species counts from the processed dataset.

    Reads CSV in chunks so ranking one species does not retain every species.
    Returns columns [OS_GRID_REF, year, actual] with genuine missing targets as NaN.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing processed observations: {source}. "
            "Run preprocessing notebook 01 to create ukbbs_model_data.csv."
        )
    columns = [square_col, "species", time_col, target_col]
    frames = []
    try:
        chunks = pd.read_csv(
            source, usecols=columns,
            dtype={square_col: "string", "species": "string"},
            chunksize=150_000,
        )
        for chunk in chunks:
            chunk[time_col] = pd.to_numeric(chunk[time_col], errors="raise")
            keep = chunk["species"].eq(species) & chunk[time_col].le(train_through)
            if keep.any():
                frames.append(chunk.loc[keep, [square_col, time_col, target_col]])
    except ValueError as exc:
        raise ValueError(
            f"Could not read {source} with expected columns {columns}. "
            "Check the task target and processed observation format."
        ) from exc

    if not frames:
        raise ValueError(f"No pre-{train_through + 1} observations for species {species!r}")
    history = pd.concat(frames, ignore_index=True).rename(
        columns={square_col: "OS_GRID_REF", time_col: "year", target_col: "actual"}
    )
    if history[["OS_GRID_REF", "year"]].isna().any().any():
        raise ValueError("Historical observations contain missing square/year keys")
    if history.duplicated(["OS_GRID_REF", "year"]).any():
        raise ValueError("Historical species observations have duplicate square/year keys")
    history["actual"] = pd.to_numeric(history["actual"], errors="raise")
    numbers = history["actual"].dropna().to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Historical count values must be finite when observed")
    if not np.equal(history["year"].to_numpy(dtype=float), history["year"].to_numpy(dtype=int)).all():
        raise ValueError("Historical year must be an integer calendar year")
    history["year"] = history["year"].astype(int)
    return history.sort_values(["OS_GRID_REF", "year"]).reset_index(drop=True)


def select_squares_by_training_coverage(
    history: pd.DataFrame,
    matched_forecasts: pd.DataFrame,
    *,
    n: int = 5,
    squares: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Choose squares at 0%, 20%, ... down the *training-coverage ranking*.

    Rank **only** forecastable squares with a common scored test row, but use
    their *full* pre-cutoff observation history, not the test prediction period.
    Requiring >=1 observed training year ensures automatic plots show history.
    """
    if n < 1:
        raise ValueError("N_PLOT_SQUARES must be >= 1")
    if history.empty or matched_forecasts.empty:
        raise ValueError("Need training history and common observed forecast rows")
    eligible = matched_forecasts["OS_GRID_REF"].astype(str).dropna().unique()
    total_years = int(history["year"].nunique())
    if not total_years:
        raise ValueError("No training years found")
    observed = history.loc[history["actual"].notna()]
    counts = observed.groupby("OS_GRID_REF")["year"].nunique()
    ranking = pd.DataFrame({"OS_GRID_REF": sorted(eligible)})
    ranking["n_training_years"] = (
        ranking["OS_GRID_REF"].map(counts).fillna(0).astype(int)
    )
    ranking["possible_training_years"] = total_years
    ranking["training_coverage_pct"] = (
        100 * ranking["n_training_years"] / total_years
    )
    ranking = ranking.sort_values(
        ["n_training_years", "OS_GRID_REF"],
        ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    if squares is not None:
        selected = [str(x) for x in squares]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("PLOT_SQUARES must be distinct and nonempty")
        missing = sorted(set(selected) - set(ranking.OS_GRID_REF))
        if missing:
            raise ValueError(f"Requested squares have no common observed forecasts: {missing}")
        return ranking.set_index("OS_GRID_REF").loc[selected].reset_index()

    ranking = ranking.loc[ranking.n_training_years.gt(0)].reset_index(drop=True)
    if ranking.empty:
        raise ValueError("None of the forecastable squares has an observed training year")
    sample_count = min(n, len(ranking))
    # With n=5 and enough squares: ranks 0%, 20%, 40%, 60%, 80%.
    # Using floor(fraction * population), not floor(fraction * (N-1)), also
    # gives distinct ranks when the entire population is exactly five squares.
    indices = [min(int(i * len(ranking) / sample_count), len(ranking) - 1)
               for i in range(sample_count)]
    result = ranking.iloc[indices].copy()
    result.insert(1, "rank_percentile_from_top", [round(100 * i / sample_count) for i in range(sample_count)])
    return result.reset_index(drop=True)

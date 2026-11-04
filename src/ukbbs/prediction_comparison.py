"""Compare saved UKBBS prediction folders on exactly matched observed test rows.

No fitting or downloading. Discover model runs by their per-folder manifest and CSVs,
independent of predictions/index.json (which can be missing or outdated).
Missing actual counts are never interpreted as zero; signed error is actual - prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SavedRun:
    directory: str
    label: str
    model_name: str
    species: tuple[str, ...]
    task: dict
    run_config: dict


@dataclass
class Comparison:
    runs: list[SavedRun]
    species: list[str]
    overall: pd.DataFrame
    by_species: pd.DataFrame
    by_square: pd.DataFrame
    by_year: pd.DataFrame
    # Most useful for inspecting individual forecast trajectories. Kept for one species only.
    plot_species: str
    plot_rows: pd.DataFrame


def discover_runs(prediction_root: str | Path) -> list[SavedRun]:
    """Discover every immediate subdirectory containing a saved run.

    Ignore unrelated subdirectories without CSV files. Fail on malformed run folders
    rather than silently hiding saved forecasts from the comparison.
    """
    root = Path(prediction_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {root}")
    runs: list[SavedRun] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        csvs = sorted(p.stem for p in folder.glob("*.csv") if p.is_file())
        if not csvs:
            continue
        path = folder / "manifest.json"
        if not path.is_file():
            raise ValueError(f"Saved prediction folder has CSVs but no manifest: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            identity = manifest["identity"]
            task = identity["task"]
            model_name = identity["model_name"]
            group_cols = task["group_cols"]
            if not isinstance(group_cols, list) or "OS_GRID_REF" not in group_cols or "species" not in group_cols:
                raise ValueError("Expected task grouping by OS_GRID_REF and species")
            if not all(isinstance(task[field], str) for field in ("target", "time_col")):
                raise ValueError("Invalid task schema")
            if not isinstance(model_name, str):
                raise ValueError("Missing model name")
            run_config = identity.get("run_config", {})
            if not isinstance(run_config, dict):
                raise ValueError("Invalid run configuration")
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid saved manifest {path}: {exc}") from exc
        digest = str(manifest.get("hash", "")) or folder.name.rsplit("__", 1)[-1]
        runs.append(SavedRun(
            directory=folder.name,
            label=f"{model_name} ({digest[:8]})",
            model_name=model_name,
            species=tuple(csvs),
            task=task,
            run_config=run_config,
        ))
    if not runs:
        raise ValueError(f"No saved model folders with species CSV files found in {root}")
    return runs


def select_runs(runs: list[SavedRun], folders: Iterable[str] | None = None) -> list[SavedRun]:
    """Default to all discovered runs, or choose specific directory names."""
    if folders is None:
        return list(runs)
    names = list(folders)
    if not names or len(names) != len(set(names)):
        raise ValueError("MODEL_FOLDERS must be a nonempty list without duplicates")
    lookup = {run.directory: run for run in runs}
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown prediction folders: {unknown}")
    return [lookup[name] for name in names]


def common_species(runs: list[SavedRun], only: Iterable[str] | None = None) -> list[str]:
    if not runs:
        raise ValueError("Select at least one model run")
    common = set.intersection(*(set(run.species) for run in runs))
    if only is None:
        wanted = sorted(common)
    else:
        wanted = sorted(set(only))
        not_shared = set(wanted) - common
        if not_shared:
            raise ValueError(f"Species not present in every selected model: {sorted(not_shared)}")
    if not wanted:
        raise ValueError("Selected models have no species CSV files in common")
    return wanted


def _shared_task(runs: list[SavedRun]) -> tuple[list[str], str, str]:
    task = runs[0].task
    if any(run.task != task for run in runs[1:]):
        raise ValueError("Cannot compare runs with different target, time, or grouping columns")
    keys = [*task["group_cols"], task["time_col"]]
    if len(keys) != len(set(keys)) or task["target"] in keys:
        raise ValueError(f"Invalid task schema: {task}")
    return keys, task["time_col"], task["target"]


def matched_species_rows(
    prediction_root: str | Path, runs: list[SavedRun], species: str,
) -> pd.DataFrame:
    """Return one row per square-year scored by *every* model for this species.

    Prediction columns are pred_0, pred_1, ... in the order of ``runs``.
    Matched observed values must agree across all models. No missing truth is scored.
    """
    keys, time_col, target = _shared_task(runs)
    root = Path(prediction_root)
    merged: pd.DataFrame | None = None
    for i, run in enumerate(runs):
        file = root / run.directory / f"{species}.csv"
        if not file.is_file():
            raise FileNotFoundError(file)
        # Retain leading zeros or alphanumeric grid square identifiers.
        group_cols = run.task["group_cols"]
        frame = pd.read_csv(file, dtype={c: "string" for c in group_cols})
        required = set([*keys, target, "prediction"])
        if missing := required.difference(frame.columns):
            raise ValueError(f"{file} is missing columns: {sorted(missing)}")
        if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{file} contains empty, missing, or duplicate forecast keys")
        if not frame["species"].eq(species).all():
            raise ValueError(f"{file} contains rows for a different species")
        for column in [time_col, target, "prediction"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[time_col].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid forecast times in {file}")
        if not np.isfinite(frame["prediction"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite model predictions in {file}")
        actual = frame[target].to_numpy(dtype=float)
        if not np.isfinite(actual[~np.isnan(actual)]).all():
            raise ValueError(f"Non-finite actual counts in {file}")
        scored = frame.loc[frame[target].notna(), [*keys, target, "prediction"]]
        scored = scored.rename(columns={target: f"actual_{i}", "prediction": f"pred_{i}"})
        if merged is None:
            merged = scored.copy()
        else:
            merged = merged.merge(scored, on=keys, how="inner", validate="one_to_one")
    assert merged is not None
    if merged.empty:
        return pd.DataFrame(columns=[*keys, "actual", *[f"pred_{i}" for i in range(len(runs))]])
    for i in range(1, len(runs)):
        if not np.array_equal(merged["actual_0"].to_numpy(), merged[f"actual_{i}"].to_numpy()):
            raise ValueError(
                f"Observed counts disagree between saved runs for {species}. "
                "Check dataset releases, targets and preprocessing."
            )
    result = merged.rename(columns={"actual_0": "actual"}).drop(
        columns=[f"actual_{i}" for i in range(1, len(runs))]
    )
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _scores(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = actual - prediction
    return {
        "n_scored": len(residual),
        "MAE": float(np.abs(residual).mean()),
        "RMSE": float(np.sqrt(np.square(residual).mean())),
        "bias": float(residual.mean()),
    }


def compare_saved_predictions(
    prediction_root: str | Path,
    *,
    model_folders: Iterable[str] | None = None,
    species: Iterable[str] | None = None,
    plot_species: str | None = None,
) -> Comparison:
    """Score models on identical observations, pooling square-years within species.

    Overall MAE/RMSE pool observed square-years across species (micro-average),
    so abundant species may dominate. Per-species and per-square tables are also
    returned to inspect heterogeneity. Entire matched rows are held only for the
    chosen plot species to avoid retaining huge UKBBS matrices in memory.
    """
    runs = select_runs(discover_runs(prediction_root), model_folders)
    selected_species = common_species(runs, species)
    chosen = plot_species or selected_species[0]
    if chosen not in selected_species:
        raise ValueError(f"Plot species {chosen!r} must be among {selected_species}")
    keys, time_col, _ = _shared_task(runs)
    species_records: list[dict] = []
    square_records: list[dict] = []
    totals = [dict(n=0, sum_abs=0.0, sum_sq=0.0, sum_err=0.0, species=0,
                   squares=0) for _ in runs]
    # Aggregate year-level errors across all selected species without retaining
    # each species' entire prediction matrix in memory.
    annual: dict[tuple[int, object], dict[str, float | int]] = {}
    plot_rows = pd.DataFrame()
    species_with_overlap: list[str] = []
    for code in selected_species:
        common = matched_species_rows(prediction_root, runs, code)
        if common.empty:
            continue
        species_with_overlap.append(code)
        if code == chosen:
            plot_rows = common.copy()
        for i, run in enumerate(runs):
            actual = common["actual"].to_numpy(dtype=float)
            predicted = common[f"pred_{i}"].to_numpy(dtype=float)
            summary = _scores(actual, predicted)
            species_records.append({"model": run.label, "directory": run.directory,
                                    "species": code, "n_squares": common.OS_GRID_REF.nunique(),
                                    **summary})
            err = actual - predicted
            year_error = pd.DataFrame({
                "time": common[time_col], "abs": np.abs(err),
                "sq": np.square(err), "signed": err,
            }).groupby("time", as_index=False).agg(
                n=("signed", "size"), abs_sum=("abs", "sum"),
                sq_sum=("sq", "sum"), signed_sum=("signed", "sum"),
            )
            for row in year_error.itertuples(index=False):
                slot = annual.setdefault((i, row.time), dict(n=0, abs_sum=0.0,
                                        sq_sum=0.0, signed_sum=0.0))
                slot["n"] += row.n
                slot["abs_sum"] += row.abs_sum
                slot["sq_sum"] += row.sq_sum
                slot["signed_sum"] += row.signed_sum
            totals[i]["n"] += len(err)
            totals[i]["sum_abs"] += float(np.abs(err).sum())
            totals[i]["sum_sq"] += float(np.square(err).sum())
            totals[i]["sum_err"] += float(err.sum())
            totals[i]["species"] += 1
            totals[i]["squares"] += common.OS_GRID_REF.nunique()
            per_square = pd.DataFrame({
                "OS_GRID_REF": common["OS_GRID_REF"],
                "error": err,
                "absolute_error": np.abs(err),
                "squared_error": np.square(err),
            }).groupby("OS_GRID_REF", sort=True, as_index=False).agg(
                n_scored=("error", "size"),
                MAE=("absolute_error", "mean"),
                mean_squared_error=("squared_error", "mean"),
                bias=("error", "mean"),
            )
            per_square["RMSE"] = np.sqrt(per_square.pop("mean_squared_error"))
            per_square["model"] = run.label
            per_square["directory"] = run.directory
            per_square["species"] = code
            square_records.extend(per_square.to_dict("records"))
    if not species_with_overlap:
        raise ValueError(
            "No common observed square-year predictions across the selected models. "
            "Choose runs with overlapping forecast horizons, squares and species."
        )
    if chosen not in species_with_overlap:
        if plot_species is not None:
            raise ValueError(
                f"No common observed predictions for plot species {chosen!r}; "
                f"choose one of {species_with_overlap}"
            )
        chosen = species_with_overlap[0]
        plot_rows = matched_species_rows(prediction_root, runs, chosen)
    overall_records = []
    for run, total in zip(runs, totals):
        n = total["n"]
        overall_records.append({
            "model": run.label, "directory": run.directory,
            "n_species": total["species"], "n_species_squares": total["squares"],
            "n_scored": n,
            "MAE": total["sum_abs"] / n,
            "RMSE": float(np.sqrt(total["sum_sq"] / n)),
            "bias": total["sum_err"] / n,
        })
    year_records = []
    for (i, year), entry in annual.items():
        n = entry["n"]
        year_records.append({
            "model": runs[i].label, "directory": runs[i].directory,
            time_col: year, "n_scored": n,
            "MAE": entry["abs_sum"] / n,
            "RMSE": float(np.sqrt(entry["sq_sum"] / n)),
            "bias": entry["signed_sum"] / n,
        })
    return Comparison(
        runs=runs, species=species_with_overlap,
        overall=pd.DataFrame(overall_records).sort_values("MAE").reset_index(drop=True),
        by_species=pd.DataFrame(species_records),
        by_square=pd.DataFrame(square_records),
        by_year=pd.DataFrame(year_records),
        plot_species=chosen, plot_rows=plot_rows,
    )


def choose_squares(
    matched: pd.DataFrame, *, n: int = 5, seed: int = 42,
    squares: Iterable[str] | None = None, time_col: str = "year",
) -> list[str]:
    """Choose reproducible random squares with >=2 scored years when possible."""
    if matched.empty:
        raise ValueError("No common observed predictions for plot species")
    years = matched.groupby("OS_GRID_REF")[time_col].nunique()
    if squares is not None:
        selected = list(squares)
        unknown = sorted(set(selected) - set(years.index.astype(str)))
        if unknown:
            raise ValueError(f"Requested squares have no common observed forecasts: {unknown}")
        if len(selected) != len(set(selected)) or not selected:
            raise ValueError("PLOT_SQUARES must be distinct and nonempty")
        return selected
    if n < 1:
        raise ValueError("n must be >= 1")
    eligible = sorted(years[years >= 2].index.astype(str))
    if not eligible:
        eligible = sorted(years.index.astype(str))
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    return list(selected)

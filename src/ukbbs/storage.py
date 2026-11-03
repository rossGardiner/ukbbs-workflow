"""Save evaluated forecasts by experiment identity, then by species.

An experiment ID is a deterministic hash of the model name, model parameters,
task schema, and run configuration. It is NOT a hash of fitted weights. Include
data version, training cutoff, and other relevant settings in run_config to avoid
collisions between experiments.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .evaluation import EvaluationResult
from .forecasting import ForecastTask, PREDICTION_COLUMN, validate_frame

_SPECIES_CODE = re.compile(r"[A-Za-z0-9_-]+\Z")


def _json_default(value: Any) -> Any:
    """Support common scalar configuration values without repr-based hashes."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Configuration value {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False, default=_json_default)


def _write_atomic(path: Path, contents: bytes) -> None:
    """Replace a file only after its complete contents have been written."""
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_forecast_results(
    result: EvaluationResult,
    *,
    output_root: str | Path,
    model_name: str,
    task: ForecastTask,
    model_params: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    species_col: str = "species",
    overwrite: bool = False,
) -> Path:
    """Persist one CSV per species and maintain ``index.json`` at output_root.

    Layout::

        output_root/
          index.json                  # human names -> hashes/directories
          model-name__<hash>/
            manifest.json            # config defining the hash
            CU.csv                   # all rows for species CU
            ...

    Include dataset release, evaluation window and other relevant details in
    run_config. Two notebooks with the same model/config write to the same
    directory; conflicting existing species CSVs raise unless overwrite=True.
    This is intended for sequential notebook writes, not concurrent processes.
    """
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    model_name = model_name.strip()
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    if not slug:
        raise ValueError("model_name must contain letters or digits")
    if not isinstance(species_col, str) or not species_col:
        raise ValueError("species_col must be a nonempty column name")

    validate_frame(result.predictions, task, labelled=True)
    if PREDICTION_COLUMN not in result.predictions.columns:
        raise ValueError(f"Predictions must include {PREDICTION_COLUMN!r}")
    if species_col not in result.predictions.columns:
        raise ValueError(f"Predictions must include a {species_col!r} column")
    if result.predictions[species_col].isna().any():
        raise ValueError("Species codes cannot be missing")

    # Only strings with safe characters become filenames; no path traversal or
    # accidental collisions caused by replacing punctuation with underscores.
    codes = result.predictions[species_col].unique().tolist()
    if any(not isinstance(code, str) or not _SPECIES_CODE.fullmatch(code) for code in codes):
        raise ValueError("Species codes must contain only ASCII letters, digits, _ or -")

    identity = {
        "model_name": model_name,
        "model_params": dict(model_params or {}),
        "run_config": dict(run_config or {}),
        "task": {
            "target": task.target,
            "time_col": task.time_col,
            "group_cols": list(task.group_cols),
        },
    }
    try:
        canonical = _canonical_json(identity)
        identity = json.loads(canonical)  # normalized scalar types and ordering
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model configuration must be JSON serializable: {exc}") from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    folder_name = f"{slug}__{digest}"

    root = Path(output_root)
    folder = root / folder_name
    manifest_path = folder / "manifest.json"
    index_path = root / "index.json"
    manifest = {"schema_version": 1, "hash": digest, "identity": identity}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")

    # Validate the entire request before writing any species files.
    species_files = {
        f"{code}.csv": result.predictions.loc[result.predictions[species_col].eq(code)].to_csv(
            index=False
        ).encode("utf-8")
        for code in codes
    }

    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError(f"Existing model manifest does not match the experiment: {folder}")
    conflicting = [
        name for name, contents in species_files.items()
        if (folder / name).exists() and (folder / name).read_bytes() != contents
    ]
    if conflicting and not overwrite:
        raise FileExistsError(
            f"Different predictions already exist for {conflicting} in {folder}; "
            "use overwrite=True only if you intend to replace them"
        )

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("schema_version") != 1 or not isinstance(index.get("models"), dict):
            raise ValueError(f"Unrecognised index format: {index_path}")
    else:
        index = {"schema_version": 1, "models": {}}
    entries = index["models"].setdefault(model_name, [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid lookup entry for {model_name!r}")
    entry = {"hash": digest, "directory": folder_name}
    if entry not in entries:
        if any(item.get("hash") == digest for item in entries):
            raise ValueError("Model hash already exists with a different directory in index")
        entries.append(entry)
        entries.sort(key=lambda item: item["hash"])

    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in species_files.items():
        path = folder / name
        if not path.exists() or path.read_bytes() != contents:
            _write_atomic(path, contents)
    if not manifest_path.exists():
        _write_atomic(manifest_path, manifest_bytes)
    _write_atomic(index_path, (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return folder

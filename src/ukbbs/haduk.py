"""Extract local Met Office HadUK-Grid 1 km annual NetCDF data for UK BBS squares.

Only *gridded*, individual-year, annual files are accepted. CEDA 'region'
files and 20/30-year climatologies cannot provide values for a BBS square.
The output matches the annual schema used by :mod:`ukbbs.climate`, so its
existing lag builder and fixed-origin leakage checks can be reused.

Run after downloading the real, non-empty .nc files from CEDA. No network
calls or file downloads are made by this module.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .climate_features import LOCATION, YEAR

# Annual HadUK-Grid provides GROUND frost, NOT air frost.
VARIABLES = {
    "tas": "temp_mean_c",
    "rainfall": "precip_total_mm",
    "groundfrost": "ground_frost_days",
}
DEFAULT_VARIABLES = ("tas", "rainfall")
_PATTERN = re.compile(
    r"(?P<var>tas|rainfall|groundfrost)_hadukgrid_uk_1km_ann_"
    r"(?P<start>\d{4})01-(?P<end>\d{4})12\.nc$"
)
_X = "projection_x_coordinate"
_Y = "projection_y_coordinate"


def find_haduk_annual_files(
    had_dir: str | Path, *, start_year: int, end_year: int,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> dict[tuple[str, int], Path]:
    """Find requested annual 1km variables for every requested year.

    Accept any directory layout below `had_dir` (including extracted CEDA
    hierarchy). Fail loudly on 0-byte downloads, missing or duplicate files.
    """
    root = Path(had_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"HadUK directory not found: {root}")
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if not variables or len(variables) != len(set(variables)) or any(v not in VARIABLES for v in variables):
        raise ValueError(f"variables must be unique and drawn from {tuple(VARIABLES)}")

    matches: dict[tuple[str, int], Path] = {}
    for path in sorted(root.rglob("*.nc")):
        m = _PATTERN.fullmatch(path.name)
        if not m or m.group("start") != m.group("end"):
            continue
        year = int(m.group("start"))
        if not (start_year <= year <= end_year):
            continue
        key = (m.group("var"), year)
        if key[0] in variables:
            if path.stat().st_size == 0:
                raise ValueError(
                    f"Empty HadUK-Grid NetCDF: {path}. "
                    "CEDA bulk ZIP downloads may contain zero-byte placeholders "
                    "when authentication is required; download the real file after login."
                )
            if key in matches:
                raise ValueError(f"Duplicate HadUK-Grid data for {key}: {matches[key]} and {path}")
            matches[key] = path

    missing = [(variable, year) for variable in variables
               for year in range(start_year, end_year + 1)
               if (variable, year) not in matches]
    if missing:
        examples = ", ".join(f"{variable}/{year}" for variable, year in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} annual HadUK-Grid 1km files in {root} "
            f"(for example: {examples}). Required filenames look like "
            "tas_hadukgrid_uk_1km_ann_201901-201912.nc. "
            "Select CEDA '1km' → variable → 'ann', not 'region' or 'ann-30y'."
        )
    return matches


def _site_coordinates(locations: pd.DataFrame) -> pd.DataFrame:
    required = {LOCATION, "ETRS89Long", "ETRS89Lat"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"locations.csv missing columns: {sorted(missing)}")
    sites = locations[[LOCATION, "ETRS89Long", "ETRS89Lat"]].copy()
    if sites.empty or sites[LOCATION].isna().any() or sites[LOCATION].duplicated().any():
        raise ValueError("Need nonempty locations with unique, non-null square identifiers")
    sites["ETRS89Long"] = pd.to_numeric(sites["ETRS89Long"], errors="raise")
    sites["ETRS89Lat"] = pd.to_numeric(sites["ETRS89Lat"], errors="raise")
    if not (np.isfinite(sites[["ETRS89Long", "ETRS89Lat"]].to_numpy()).all()
            and sites["ETRS89Long"].between(-9, 3).all()
            and sites["ETRS89Lat"].between(49, 62).all()):
        raise ValueError("Invalid UK ETRS89 latitude/longitude in locations.csv")

    # IMPORTANT: preprocessing's easting/northing are EPSG:3035, not BNG.
    # Use original ETRS89 coordinates instead of those easting/northing values.
    transform = Transformer.from_crs("EPSG:4258", "EPSG:27700", always_xy=True)
    x, y = transform.transform(sites.ETRS89Long.to_numpy(), sites.ETRS89Lat.to_numpy())
    sites["_x"], sites["_y"] = x, y
    if not np.isfinite(sites[["_x", "_y"]].to_numpy()).all():
        raise ValueError("Coordinate transformation failed")
    return sites


def _extract_one(path: Path, variable: str, sites: pd.DataFrame) -> np.ndarray:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Install HadUK dependencies with: pip install -e '.[haduk]'") from exc

    # NetCDF4/HDF5 support is provided by the optional netCDF4 dependency.
    with xr.open_dataset(path) as dataset:
        if variable not in dataset.data_vars:
            raise ValueError(f"{path.name}: expected variable {variable!r}; got {list(dataset.data_vars)}")
        field = dataset[variable]
        if _X not in field.coords or _Y not in field.coords:
            raise ValueError(f"{path.name}: missing {_X}/{_Y} grid coordinates")
        if field[_X].ndim != 1 or field[_Y].ndim != 1:
            raise ValueError(f"{path.name}: expected 1D projection coordinates")
        for dimension in tuple(field.dims):
            if dimension not in (_X, _Y):
                if field.sizes[dimension] != 1:
                    raise ValueError(f"{path.name}: expected one annual value, {dimension} has {field.sizes[dimension]}")
                field = field.isel({dimension: 0})

        # Vectorised point lookup; read only selected cells, not the whole UK grid.
        x = xr.DataArray(sites["_x"].to_numpy(), dims="site")
        y = xr.DataArray(sites["_y"].to_numpy(), dims="site")
        # Bounds check prevents nearest() returning distant coastal/off-grid cells.
        gx = field[_X].to_numpy()
        gy = field[_Y].to_numpy()
        within = ((sites["_x"].to_numpy() >= gx.min() - 500)
                  & (sites["_x"].to_numpy() <= gx.max() + 500)
                  & (sites["_y"].to_numpy() >= gy.min() - 500)
                  & (sites["_y"].to_numpy() <= gy.max() + 500))
        sampled = field.sel({_X: x, _Y: y}, method="nearest").to_numpy()
        sampled = np.asarray(sampled, dtype=float)
        if sampled.shape != (len(sites),):
            raise ValueError(f"{path.name}: unexpected extracted shape {sampled.shape}")
        sampled[~within] = np.nan
        sampled[~np.isfinite(sampled)] = np.nan
        return sampled


def extract_haduk_annual(
    locations: pd.DataFrame, *, had_dir: str | Path, start_year: int, end_year: int,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> pd.DataFrame:
    """Read local annual 1km climate for each BBS square and calendar year.

    Returns OS_GRID_REF, year, temp_mean_c (°C), precip_total_mm (mm),
    optionally ground_frost_days (days). HadUK-Grid does not supply air frost.
    Values outside land coverage remain NaN; no spatial imputation is done.
    """
    sites = _site_coordinates(locations)
    sources = find_haduk_annual_files(had_dir, start_year=start_year, end_year=end_year, variables=variables)
    years = range(start_year, end_year + 1)
    records = []
    for year in years:
        frame = pd.DataFrame({LOCATION: sites[LOCATION].to_numpy(), YEAR: year})
        for variable in variables:
            frame[VARIABLES[variable]] = _extract_one(sources[(variable, year)], variable, sites)
        records.append(frame)
    result = pd.concat(records, ignore_index=True)
    return result[[LOCATION, YEAR, *(VARIABLES[v] for v in variables)]].sort_values([LOCATION, YEAR]).reset_index(drop=True)

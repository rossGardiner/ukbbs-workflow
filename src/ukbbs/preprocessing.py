"""Download a fixed BBS release, reproduce the R aggregation, and attach coordinates."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer

RECORD_ID = "18598453"
SOURCE_URL = f"https://zenodo.org/records/{RECORD_ID}"
EXCLUDED_YEARS = (2001, 2020)
SOURCE_CRS = "EPSG:4258"  # ETRS89 geographic coordinates in the publisher lookup
TARGET_CRS = "EPSG:3035"  # ETRS89-extended / LAEA Europe, metres
COUNT_COLUMN = "max_total_count_over_all_distance_bands_for_max_visit"
SECTIONS = [f"section_{i}" for i in range(1, 11)]
VISIT_KEYS = ["year", "square", "visit", "species_code"]
ANNUAL_KEYS = ["year", "square", "species_code"]
FILE_MD5 = {
    "BBS_bird_dataset.csv": "26a5044bda81b5c772f685425da0ccfc",
    "BBS_visit_dataset.csv": "1a1b0fb63a4fa3ea133d4c2e33f000fc",
    "grid_square_coordinates_lookup.csv": "66c0211c91cfac89537c095c0d2c2658",
    "species_lookup.csv": "8a75adf1409000c4662bb9d148bf6d61",
    "0_Metadata.pdf": "32d54f8e5772366c5c1364c56af9c8c7",
}


def file_md5(path: Path) -> str:
    """Publisher MD5 is an integrity check, not a security mechanism."""
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_release(raw_dir: Path) -> dict[str, Path]:
    """Stream fixed-version files to disk; verify cached and new files.

    Failed or interrupted downloads do not replace completed files. A corrupt
    cache raises an error instead of silently reusing or overwriting it.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in FILE_MD5.items():
        destination = raw_dir / name
        if destination.exists():
            if file_md5(destination) != expected:
                raise ValueError(f"Checksum mismatch: {destination}. Remove it and rerun.")
            print(f"Verified cached {name}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = Request(
                f"{SOURCE_URL}/files/{name}?download=1",
                headers={"User-Agent": "ukbbs-workflow/0.1"},
            )
            print(f"Downloading {name} ...", flush=True)
            try:
                with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                if file_md5(temporary) != expected:
                    raise ValueError(f"Downloaded checksum mismatch: {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        files[name] = destination
    return files


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_keys(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalise grid formatting, and reject missing grouping keys."""
    frame = frame.copy()
    require_columns(frame, columns)
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing grouping key: {column}")
    if "square" in columns:
        frame["square"] = frame["square"].astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    if "year" in columns:
        years = pd.to_numeric(frame["year"], errors="raise")
        if not np.isfinite(years).all() or (years % 1 != 0).any():
            raise ValueError("Years must be finite integers")
        frame["year"] = years.astype("int64")
    return frame


def preprocess_counts(
    path: Path,
    *,
    excluded_years: tuple[int, ...] = EXCLUDED_YEARS,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict]:
    """R-equivalent section sum -> visit sum -> annual maximum.

    All distance bands, including F (flight only), are retained. Missing section
    cells contribute zero, including all-missing rows, as in rowSums(na.rm=TRUE).
    No absent species, visits, or years are invented. Chunk partial totals are
    summed again before taking maxima, even when a visit spans chunk boundaries.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    header = pd.read_csv(path, nrows=0)
    require_columns(header, VISIT_KEYS + ["distance_band"] + SECTIONS)
    parts = []
    stats = {"raw_rows": 0, "excluded_rows": 0, "missing_section_cells": 0,
             "missing_distance_band_rows": 0,
             "all_missing_section_rows": 0, "distance_bands": []}
    bands = set()
    reader = pd.read_csv(
        path, usecols=VISIT_KEYS + ["distance_band"] + SECTIONS,
        dtype={key: "string" for key in ["square", "visit", "species_code", "distance_band"]},
        keep_default_na=False,
        na_values={column: ["", "NA", "NaN"] for column in SECTIONS + ["year"]},
        chunksize=chunksize,
    )
    with reader:
        for chunk in reader:
            chunk = clean_keys(chunk, VISIT_KEYS)
            counts = chunk[SECTIONS].apply(pd.to_numeric, errors="raise")
            values = counts.to_numpy(dtype=float)
            valid = values[~np.isnan(values)]
            if not np.isfinite(valid).all() or (valid < 0).any() or (valid % 1 != 0).any():
                raise ValueError("Section counts must be missing or finite nonnegative integers")
            stats["raw_rows"] += len(chunk)
            stats["missing_section_cells"] += int(counts.isna().sum().sum())
            stats["all_missing_section_rows"] += int(counts.isna().all(axis=1).sum())
            # Band labels are not grouping keys in the R workflow. Retain counts
            # with an unspecified band and expose them in the diagnostics.
            band_labels = chunk["distance_band"].fillna("").str.strip()
            missing_band = band_labels.isin(["", "NA", "NaN"])
            stats["missing_distance_band_rows"] += int(missing_band.sum())
            bands.update(band_labels.mask(missing_band, "(missing)").unique())
            # pandas skipna=True matches the original rowSums(..., na.rm=TRUE).
            chunk["band_count"] = counts.sum(axis=1).astype("int64")
            excluded = chunk["year"].isin(excluded_years)
            stats["excluded_rows"] += int(excluded.sum())
            chunk = chunk.loc[~excluded]
            if not chunk.empty:
                parts.append(chunk.groupby(VISIT_KEYS, as_index=False, sort=False)["band_count"].sum())
    if not parts:
        raise ValueError("No rows remain after preprocessing")
    # A visit may occur in several chunks. Combine those sums before max().
    visits = pd.concat(parts, ignore_index=True).groupby(
        VISIT_KEYS, as_index=False, sort=False
    )["band_count"].sum()
    annual = visits.groupby(ANNUAL_KEYS, as_index=False)["band_count"].max()
    annual = annual.rename(columns={"square": "OS_GRID_REF", "species_code": "species", "band_count": COUNT_COLUMN})
    annual = annual[["OS_GRID_REF", "species", COUNT_COLUMN, "year"]]
    annual = annual.sort_values(["year", "OS_GRID_REF", "species"]).reset_index(drop=True)
    stats.update(distance_bands=sorted(bands), output_rows=len(annual),
                 square_species_visits=len(visits), unique_squares=int(annual.OS_GRID_REF.nunique()),
                 unique_species=int(annual.species.nunique()))
    return annual, stats


def survey_coverage(path: Path, *, excluded_years=EXCLUDED_YEARS):
    """True survey coverage comes from visit records, not species detections."""
    visits = pd.read_csv(path, usecols=["square", "year"], dtype={"square": "string"}, keep_default_na=False)
    visits = clean_keys(visits, ["square", "year"])
    sampled = visits.loc[~visits.year.isin(excluded_years), ["square", "year"]].drop_duplicates()
    sampled = sampled.rename(columns={"square": "OS_GRID_REF"}).sort_values(["year", "OS_GRID_REF"]).reset_index(drop=True)
    if sampled.empty:
        raise ValueError("No survey visits remain")
    # Check against every calendar year in the retained span, except exclusions.
    expected_years = sorted(set(range(int(sampled.year.min()), int(sampled.year.max()) + 1)) - set(excluded_years))
    coverage = sampled.groupby("OS_GRID_REF").year.nunique().rename("n_years_sampled").reset_index()
    coverage["sampled_every_retained_year"] = coverage.n_years_sampled.eq(len(expected_years))
    return sampled, coverage, expected_years


def coordinate_lookup(path: Path) -> pd.DataFrame:
    """Transform publisher ETRS89 square centroids once per square.

    This deliberately replaces manual GB/Irish reference parsing and covers
    Channel Island squares too. No additional half-cell offset is applied.
    """
    lookup = pd.read_csv(path, dtype={"square": "string"}, keep_default_na=False)
    require_columns(lookup, ["square", "ETRS89Lat", "ETRS89Long", "BBS_region"])
    lookup = clean_keys(lookup, ["square"])
    if lookup.square.duplicated().any():
        raise ValueError("Coordinate lookup has duplicate square keys")
    xy = lookup[["ETRS89Long", "ETRS89Lat"]].apply(pd.to_numeric, errors="raise")
    if (not np.isfinite(xy).all().all() or not xy.ETRS89Long.between(-180, 180).all()
            or not xy.ETRS89Lat.between(-90, 90).all()):
        raise ValueError("Lookup contains invalid longitude/latitude")
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    east, north = transformer.transform(xy.ETRS89Long.tolist(), xy.ETRS89Lat.tolist(), errcheck=True)
    if not (np.isfinite(east).all() and np.isfinite(north).all()):
        raise ValueError("Non-finite projected coordinates")
    lookup = lookup.rename(columns={"square": "OS_GRID_REF"})
    lookup["easting"], lookup["northing"] = east, north
    lookup["easting_km"] = (lookup.easting / 1000).round(4)
    lookup["northing_km"] = (lookup.northing / 1000).round(4)
    lookup["coordinate_crs"] = TARGET_CRS
    lookup["coordinate_source"] = "publisher ETRS89 square centroid"
    lookup["grid_resolution_m"] = 1000  # 1-km squares in this fixed BBS release
    # Describe reference notation; projected easting/northing are ALL EPSG:3035.
    lookup["grid_system"] = np.select(
        [lookup.BBS_region.eq("CHANX"), lookup.OS_GRID_REF.str.match(r"I[A-HJ-Z]\d+")],
        ["Channel Islands grid", "Irish Grid"], default="British National Grid",
    )
    return lookup


def attach_coordinates(model: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    require_columns(model, ["OS_GRID_REF"])
    require_columns(locations, ["OS_GRID_REF", "easting", "northing"])
    result = model.merge(locations, on="OS_GRID_REF", how="left", validate="many_to_one", indicator=True)
    missing = result.loc[result._merge.ne("both"), "OS_GRID_REF"].unique().tolist()
    if missing:
        raise ValueError(f"No publisher coordinates for squares: {missing[:20]}")
    if result[["easting", "northing"]].isna().any().any():
        raise ValueError("Missing coordinates after join")
    return result.drop(columns="_merge")


def write_manifest(output: Path, files: dict[str, Path], stats: dict, excluded_years=EXCLUDED_YEARS):
    """Record source integrity, choices, software versions, and diagnostics."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_record": SOURCE_URL,
        "source_release": "2024 (1994-2023 data)",
        "input_md5": {name: file_md5(path) for name, path in files.items()},
        "excluded_years": list(excluded_years),
        "aggregation": "sum sections; sum all bands per visit; max visit per square/species/year",
        "missing_sections": "zero contribution; matches R na.rm=TRUE",
        "zero_imputation": False,
        "source_coordinate_crs": SOURCE_CRS,
        "target_coordinate_crs": TARGET_CRS,
        "coordinate_source": "publisher ETRS89 centroids; no manual grid parsing",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "pyproj": pyproj.__version__, "PROJ": pyproj.proj_version_str},
        "diagnostics": stats,
    }
    Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

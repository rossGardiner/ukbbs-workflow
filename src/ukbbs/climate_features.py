"""Calendar-year climate lags and leakage-safe fixed-origin enrichment.

Input is a location/year table; variables may include HadUK-Grid temperature,
rainfall, and ground-frost days. No API calls or dependencies on a climate
provider: all measurements are supplied explicitly by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCATION = "OS_GRID_REF"
YEAR = "year"


def climate_lag_features(
    annual: pd.DataFrame, *, history_years: int = 5,
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    """Produce one row per site/calendar year, with lag1 = previous year.

    Missing calendar years *do not* shift lag numbering: a missing 2018
    measurement means lag2 is NA for a 2020 target, not a value from 2017.
    Only the columns other than OS_GRID_REF/year are used as climate inputs.
    """
    if not isinstance(history_years, int) or history_years < 1:
        raise ValueError("history_years must be a positive integer")
    required = {LOCATION, YEAR}
    if not required.issubset(annual.columns):
        raise ValueError(f"Annual climate must contain {sorted(required)}")
    if annual.empty:
        raise ValueError("Annual climate table must not be empty")
    if annual[[LOCATION, YEAR]].isna().any().any():
        raise ValueError("Climate location/year keys cannot be missing")
    if annual[LOCATION].astype(str).str.strip().eq("").any():
        raise ValueError("Climate locations cannot be blank")
    if annual.duplicated([LOCATION, YEAR]).any():
        raise ValueError("Duplicate location/year climate records")
    fields = [c for c in annual.columns if c not in (LOCATION, YEAR)]
    if not fields or any(not isinstance(c, str) or not c for c in fields):
        raise ValueError("Annual climate needs at least one named numeric variable")
    years = pd.to_numeric(annual[YEAR], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(years).all() or (years % 1 != 0).any():
        raise ValueError("Climate years must be finite integers")
    targets = list(target_years or [])
    if any(not isinstance(y, (int, np.integer)) for y in targets):
        raise ValueError("target_years must be integer years")
    prepared = annual.copy()
    prepared[YEAR] = years.astype(int)
    for col in fields:
        prepared[col] = pd.to_numeric(prepared[col], errors="raise")
        values = prepared[col].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite climate values in {col}")
    min_year = min(prepared[YEAR].min(), *(targets or [int(prepared[YEAR].min())]))
    max_year = max(prepared[YEAR].max(), *(targets or [int(prepared[YEAR].max())]))
    # Reindex to every calendar year before shift(), even absent years.
    index = pd.MultiIndex.from_product(
        [prepared[LOCATION].drop_duplicates(), range(min_year, max_year + 1)],
        names=[LOCATION, YEAR],
    )
    filled = prepared.set_index([LOCATION, YEAR]).reindex(index)
    lagged = pd.DataFrame(index=index)
    grouped = filled.groupby(level=LOCATION, sort=False)
    for col in fields:
        for offset in range(1, history_years + 1):
            lagged[f"climate_{col}_lag{offset}"] = grouped[col].shift(offset)
    return lagged.reset_index()


def attach_fixed_origin_climate(
    train: pd.DataFrame, test: pd.DataFrame, annual: pd.DataFrame,
    *, forecast_origin: int, history_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach historical climate using a shared, immutable forecast origin.

    Training row for year Y gets measurements for Y-1, ..., Y-history_years.
    Each test row receives the same site's measurements for origin,
    origin-1, ..., origin-history_years+1, even for multi-year test periods.

    This avoids test-time climate leakage but does *not* by itself make a
    one-step model a valid multi-year forecasting model; build horizon-specific
    models or use a correct recursive strategy for multi-year forecasting.
    """
    if not isinstance(forecast_origin, int):
        raise ValueError("forecast_origin must be a calendar year (int)")
    if train.empty or test.empty:
        raise ValueError("Training and test data must be nonempty")
    for name, frame in (("train", train), ("test", test)):
        if not {LOCATION, YEAR}.issubset(frame.columns):
            raise ValueError(f"{name} is missing {LOCATION} or {YEAR}")
        if frame[[LOCATION, YEAR]].isna().any().any():
            raise ValueError(f"{name} has missing location/year keys")
    if not train[YEAR].le(forecast_origin).all() or not test[YEAR].gt(forecast_origin).all():
        raise ValueError("Forecast origin must separate all training and test years")
    # Filter BEFORE building features, not after, to prevent future data use.
    prior = annual.loc[annual[YEAR].le(forecast_origin)].copy()
    lagged = climate_lag_features(prior, history_years=history_years,
                                  target_years=[forecast_origin + 1])
    feature_names = [c for c in lagged.columns if c not in (LOCATION, YEAR)]
    for name, frame in (("train", train), ("test", test)):
        collisions = sorted(set(frame.columns) & set(feature_names))
        if collisions:
            raise ValueError(f"{name} already contains climate features: {collisions}")
    enriched_train = train.merge(lagged, on=[LOCATION, YEAR], how="left",
                                 validate="many_to_one", sort=False)
    origin_features = lagged.loc[lagged[YEAR].eq(forecast_origin + 1),
                                 [LOCATION, *feature_names]]
    enriched_test = test.merge(origin_features, on=LOCATION, how="left",
                               validate="many_to_one", sort=False)
    return enriched_train, enriched_test

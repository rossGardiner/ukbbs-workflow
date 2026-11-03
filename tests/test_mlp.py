"""Regression tests for the species-wide shared-weight autoregressive MLP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ukbbs import ForecastTask, LastValueForecaster, run_forecast
from ukbbs.mlp import MLPConfig, OneYearMLPForecaster

TARGET = ForecastTask().target


def example_data():
    # No 2001 measurements: this must NOT be treated as a 2000->2002 lag.
    # Observed zeros are deliberately distinct from absent square-years.
    rows = []
    for square, values in {
        "A": {1999: 3, 2000: 0, 2002: 5, 2003: 6, 2004: 7, 2005: 9},
        "B": {1999: 4, 2002: 0, 2003: 1, 2004: 3, 2005: 5},
    }.items():
        for year, count in values.items():
            rows.append({"OS_GRID_REF": square, "species": "CU", "year": year, TARGET: count})
    return pd.DataFrame(rows)


def small_config(seed=123):
    return MLPConfig(hidden_sizes=(8, 4), epochs=10, learning_rate=0.02,
                     batch_size=8, seed=seed)


def test_calendar_lags_missing_vs_zero():
    train = example_data().query("year <= 2003")
    model = OneYearMLPForecaster(small_config()).fit(train, ForecastTask())
    # At 2000 A has known previous data, but B's 2002 input is missing.
    # At 2002, neither square has a 2001 observation, even though 2000 has A.
    assert model.n_training_pairs_ == 5  # A: 2000,02,03 / B: 02,03
    assert model.n_missing_previous_ == 2  # A2002 and B2002 (2001 absent globally)


def test_fixed_origin_recursive_forecast_and_shared_api():
    df = example_data()
    task = ForecastTask()
    train, test = df[df.year <= 2003].copy(), df[df.year >= 2004].copy()
    model = OneYearMLPForecaster(small_config())
    evaluated = run_forecast(model, train, test, task)
    assert model.n_training_pairs_ == 5
    assert evaluated.metrics["n_scored"] == len(test)
    assert np.isfinite(evaluated.predictions["prediction"]).all()
    assert (evaluated.predictions["prediction"] >= 0).all()
    assert set(evaluated.predictions.columns) >= {TARGET, "prediction", "absolute_error"}
    assert np.isfinite(evaluated.metrics["mae"])
    # A standalone 2005 request must traverse the year 2004 internally.
    later = test.loc[test.year.eq(2005)].drop(columns=TARGET)
    predicted_later = model.predict(later)
    from_run = evaluated.predictions.loc[evaluated.predictions.year.eq(2005)]
    joined = predicted_later.merge(from_run, on=task.keys, suffixes=("_direct", "_run"))
    np.testing.assert_allclose(joined.prediction_direct, joined.prediction_run)
    # Changing test labels cannot affect the model's predictions.
    altered = test.copy()
    altered[TARGET] += 2000
    with_new_labels = run_forecast(OneYearMLPForecaster(small_config()), train, altered, task)
    np.testing.assert_allclose(evaluated.predictions.prediction,
                               with_new_labels.predictions.prediction)


def test_missing_origin_count_is_not_older_last_observation():
    df = example_data()
    # B has no 2003 observation now, even though B did in 2002.
    train = df.loc[(df.year <= 2003) & ~((df.OS_GRID_REF == "B") & (df.year == 2003))]
    model = OneYearMLPForecaster(small_config()).fit(train, ForecastTask())
    assert model._site_states["B"] == (0.0, True)
    assert model._site_states["A"] == (6.0, False)


def test_rejects_unknown_species_sites_target_and_past_year():
    task = ForecastTask()
    model = OneYearMLPForecaster(small_config()).fit(example_data().query("year <= 2003"), task)
    base = pd.DataFrame([{"OS_GRID_REF": "A", "species": "CU", "year": 2004}])
    for changed in [base.assign(species="BB"), base.assign(OS_GRID_REF="new"),
                    base.assign(year=2003), base.assign(**{TARGET: 5})]:
        with pytest.raises(ValueError):
            model.predict(changed)


def test_rejects_multiple_species_and_bad_targets():
    df = example_data().query("year <= 2003").copy()
    with pytest.raises(ValueError, match="one species"):
        OneYearMLPForecaster(small_config()).fit(pd.concat([df, df.assign(species="BB")]), ForecastTask())
    bad = df.copy()
    bad.loc[0, TARGET] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        OneYearMLPForecaster(small_config()).fit(bad, ForecastTask())


def test_predictions_deterministic_for_seed():
    df = example_data()
    task = ForecastTask()
    train, test = df[df.year <= 2003], df[df.year >= 2004]
    one = run_forecast(OneYearMLPForecaster(small_config()), train, test, task)
    two = run_forecast(OneYearMLPForecaster(small_config()), train, test, task)
    np.testing.assert_allclose(one.predictions.prediction, two.predictions.prediction)

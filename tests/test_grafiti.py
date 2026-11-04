"""Tests for the standalone GraFITi-style UKBBS forecaster."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ukbbs.forecasting import ForecastTask, chronological_split
from ukbbs.evaluation import run_forecast
from ukbbs.grafiti import GraFITiConfig, GraFITiForecaster, _SparsityGraphNetwork


@pytest.fixture
def small_data():
    rows = []
    for site, baseline in [("A1", 1.0), ("B2", 4.0), ("C3", 2.0), ("D4", 0.0), ("E5", 6.0)]:
        for year in range(2010, 2019):
            count = baseline + ((year - 2010) % 4)
            if site == "B2" and year == 2014:
                count = np.nan  # unsurveyed, not a zero
            if site == "D4" and year in (2011, 2015):
                count = np.nan
            rows.append({"OS_GRID_REF": site, "species": "CU", "year": year, "max_count": count})
    return pd.DataFrame(rows)


def config(**kw):
    defaults = dict(context_years=3, max_horizon=2, sites_per_graph=3, latent_dim=16,
                    attention_heads=2, layers=1, epochs=2, batch_size=8, seed=11, device="cpu")
    defaults.update(kw)
    return GraFITiConfig(**defaults)


def test_masked_graph_observed_zero_and_missing_are_distinct():
    net = _SparsityGraphNetwork(config())
    years = torch.tensor([[-1., 0., 1.]])
    values = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]]])
    observed = torch.tensor([[[True, False], [False, False], [False, False]]])
    query = torch.tensor([[[False, False], [False, False], [True, True]]])
    out = net(years, values, observed, query)
    assert out.shape == (1, 3, 2)
    assert torch.isfinite(out).all()
    # Site 1 is *truly missing*, whereas site 0 has an observed count of zero.
    assert observed[0, 0].tolist() == [True, False]


def test_train_and_evaluate_with_common_api(small_data):
    task = ForecastTask(target="max_count")
    train, test = chronological_split(small_data, train_through=2016, task=task)
    model = GraFITiForecaster(config())
    report = run_forecast(model, train, test, task)
    assert set(report.predictions["year"]) == {2017, 2018}
    assert len(report.predictions) == len(test)
    assert report.metrics["n_unscored"] == 0
    assert report.metrics["n_scored"] == len(test)
    assert np.isfinite(report.predictions["prediction"]).all()
    assert (report.predictions["prediction"] >= 0).all()
    assert len(model.history_) == 2
    assert model.n_training_graphs_ > 0


def test_predict_does_not_depend_on_test_rows_or_labels(small_data):
    task = ForecastTask(target="max_count")
    train, test = chronological_split(small_data, train_through=2016, task=task)
    model = GraFITiForecaster(config()).fit(train, task)
    future = test.drop(columns="max_count")
    first = model.predict(future)
    subset = model.predict(future.sample(n=3, random_state=2))
    merged = first.merge(subset, on=task.keys, suffixes=("_all", "_subset"))
    np.testing.assert_allclose(merged["prediction_all"], merged["prediction_subset"], rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="target"):
        model.predict(test)


def test_missing_history_is_absent_edge_not_zero(small_data):
    task = ForecastTask(target="max_count")
    train, _ = chronological_split(small_data, train_through=2016, task=task)
    model = GraFITiForecaster(config()).fit(train, task)
    grp = next(g for g in model._groups if "B2" in g)
    _, values, observed, *_ = model._make_graph(grp, 2015, labels=False)
    i = grp.index("B2")
    # context for origin 2015 is 2013, 2014, 2015
    assert observed[:3, i].tolist() == [True, False, True]
    assert not observed[1, i]
    assert values[1, i] == 0.0
    # Explicitly observed 0 at D4 in 2014 is not treated as missing.
    grp = next(g for g in model._groups if "D4" in g)
    _, _, observed, *_ = model._make_graph(grp, 2015, labels=False)
    i = grp.index("D4")
    assert observed[:3, i].tolist() == [True, True, False]  # 2013 and 2014 observed, 2015 absent


def test_calendar_gap_does_not_shift_observations(small_data):
    # 2020-style gap stays absent; 2015 is not mistaken for 2016.
    task = ForecastTask(target="max_count")
    train, _ = chronological_split(small_data, train_through=2016, task=task)
    train = train.loc[train.year.ne(2015)]
    model = GraFITiForecaster(config()).fit(train, task)
    grp = next(g for g in model._groups if "A1" in g)
    _, _, observed, *_ = model._make_graph(grp, 2016, labels=False)
    i = grp.index("A1")
    assert observed[:3, i].tolist() == [True, False, True]


def test_reject_unknown_species_location_and_horizon(small_data):
    task = ForecastTask(target="max_count")
    train, test = chronological_split(small_data, train_through=2016, task=task)
    model = GraFITiForecaster(config()).fit(train, task)
    future = test.drop(columns="max_count").head(1)
    with pytest.raises(ValueError, match="species"):
        model.predict(future.assign(species="LK"))
    with pytest.raises(ValueError, match="absent"):
        model.predict(future.assign(OS_GRID_REF="NEVER"))
    with pytest.raises(ValueError, match="Forecast years"):
        model.predict(future.assign(year=2020))


def test_configuration_validation():
    with pytest.raises(ValueError, match="divisible"):
        config(latent_dim=10, attention_heads=4)
    with pytest.raises(ValueError, match="positive"):
        config(epochs=0)
    with pytest.raises(ValueError, match="device"):
        config(device="not-a-device")

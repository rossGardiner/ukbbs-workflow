"""Offline tests for the shared model interface and evaluation contract."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ukbbs import (
    ForecastTask, LastValueForecaster, chronological_split,
    evaluate_forecasts, run_forecast,
)


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.task = ForecastTask(target="count")
        self.observations = pd.DataFrame([
            ("A", "CU", 2018, 0.0, 3.0),
            ("A", "CU", 2019, 4.0, 4.0),
            ("B", "CU", 2018, 2.0, 6.0),
            ("B", "CU", 2019, 6.0, 7.0),
            ("A", "CU", 2021, 3.0, 3.0),
            ("B", "CU", 2021, 0.0, 1.0),
        ], columns=["OS_GRID_REF", "species", "year", "count", "known_weather"])

    def test_end_to_end_last_value_metrics_and_rowwise_output(self):
        train, test = chronological_split(self.observations, train_through=2019, task=self.task)
        # No test label is passed to the model; labels exist only in the evaluator.
        result = run_forecast(LastValueForecaster(), train, test, self.task)
        self.assertEqual(result.metrics["n_scored"], 2)
        self.assertEqual(result.metrics["n_unscored"], 0)
        self.assertAlmostEqual(result.metrics["mae"], 3.5)
        self.assertAlmostEqual(result.metrics["rmse"], np.sqrt(18.5))
        self.assertAlmostEqual(result.metrics["bias"], -3.5)
        self.assertEqual(result.predictions["prediction"].tolist(), [4.0, 6.0])
        self.assertEqual(result.predictions["residual"].tolist(), [-1.0, -6.0])

    def test_evaluation_matches_keys_not_position(self):
        test = self.observations.query("year == 2021").copy()
        predictions = test[[*self.task.keys]].iloc[::-1].copy()
        predictions["prediction"] = [9.0, 2.0]  # B then A
        result = evaluate_forecasts(test, predictions, self.task)
        self.assertEqual(result.predictions["prediction"].tolist(), [2.0, 9.0])
        self.assertEqual(result.metrics["n_scored"], 2)

    def test_na_truth_is_not_zero_but_an_observed_zero_is_scored(self):
        test = self.observations.query("year == 2021").copy()
        test.loc[test.OS_GRID_REF.eq("A"), "count"] = np.nan
        preds = test[self.task.keys].copy()
        preds["prediction"] = [9.0, 2.0]
        result = evaluate_forecasts(test, preds, self.task)
        self.assertEqual(result.metrics["n_scored"], 1)
        self.assertEqual(result.metrics["n_unscored"], 1)
        self.assertEqual(result.metrics["mae"], 2)
        self.assertTrue(np.isnan(result.predictions["residual"].iloc[0]))
        self.assertEqual(result.predictions["count"].iloc[1], 0)

    def test_unknown_group_requires_explicit_policy(self):
        train, test = chronological_split(self.observations, train_through=2019, task=self.task)
        unseen = test.copy()
        unseen.loc[unseen.index[0], "OS_GRID_REF"] = "NEW"
        with self.assertRaisesRegex(ValueError, "No training history"):
            run_forecast(LastValueForecaster(), train, unseen, self.task)

    def test_baseline_uses_last_observed_not_null_or_zero(self):
        train = self.observations.query("year <= 2019").copy()
        train.loc[train.OS_GRID_REF.eq("A") & train.year.eq(2019), "count"] = np.nan
        test = self.observations.query("year == 2021").copy()
        result = run_forecast(LastValueForecaster(), train, test, self.task)
        self.assertEqual(result.predictions["prediction"].tolist(), [0.0, 6.0])

    def test_future_cannot_contain_target_or_past_timestamp(self):
        train, test = chronological_split(self.observations, train_through=2019, task=self.task)
        model = LastValueForecaster().fit(train, self.task)
        with self.assertRaisesRegex(ValueError, "target"):
            model.predict(test)
        with self.assertRaisesRegex(ValueError, "Future timestamps"):
            model.predict(train.drop(columns="count"))

    def test_custom_model_can_consume_metadata_and_custom_metric(self):
        class MetadataModel:
            def fit(self, train, task):
                self.task = task
                return self

            def predict(self, future):
                assert "count" not in future.columns  # guard against leakage
                return future[self.task.keys].assign(prediction=future.known_weather)

        train, test = chronological_split(self.observations, train_through=2019, task=self.task)
        result = run_forecast(
            MetadataModel(), train, test, self.task,
            metrics={"max_abs_error": lambda y, yhat: np.max(np.abs(y-yhat))},
        )
        self.assertEqual(result.predictions["prediction"].tolist(), [3.0, 1.0])
        self.assertEqual(result.metrics["max_abs_error"], 1)
        self.assertNotIn("mae", result.metrics)

    def test_custom_single_series_task(self):
        task = ForecastTask(target="index", time_col="date", group_cols=())
        observations = pd.DataFrame({
            "date": pd.to_datetime(["2018-01-01", "2019-01-01", "2021-01-01"]),
            "index": [5.0, 7.0, 8.0],
        })
        train, test = chronological_split(
            observations, train_through=pd.Timestamp("2019-01-01"), task=task,
        )
        result = run_forecast(LastValueForecaster(), train, test, task)
        self.assertEqual(result.metrics["mae"], 1.0)

    def test_reject_duplicate_missing_unmatched_keys_and_invalid_prediction(self):
        test = self.observations.query("year == 2021").copy()
        forecasts = test[self.task.keys].assign(prediction=[3.0, 1.0])
        with self.assertRaisesRegex(ValueError, "Duplicate forecast keys"):
            evaluate_forecasts(test, pd.concat([forecasts, forecasts.iloc[:1]]), self.task)
        with self.assertRaisesRegex(ValueError, "key sets differ"):
            evaluate_forecasts(test, forecasts.assign(year=2022), self.task)
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_forecasts(test, forecasts.assign(prediction=np.inf), self.task)
        with self.assertRaisesRegex(ValueError, "cannot be missing"):
            chronological_split(self.observations.assign(OS_GRID_REF=np.nan), train_through=2019, task=self.task)

    def test_temporal_leakage_and_empty_split_rejected(self):
        train = self.observations.query("year != 2018").copy()
        test = self.observations.query("year == 2019").copy()
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            run_forecast(LastValueForecaster(), train, test, self.task)
        with self.assertRaisesRegex(ValueError, "leave both"):
            chronological_split(self.observations, train_through=2030, task=self.task)

    def test_model_interface_without_runner(self):
        train, test = chronological_split(self.observations, train_through=2019, task=self.task)
        model = LastValueForecaster().fit(train, self.task)
        predicted = model.predict(test.drop(columns=self.task.target))
        direct = evaluate_forecasts(test, predicted, self.task)
        via_runner = run_forecast(LastValueForecaster(), train, test, self.task)
        assert_frame_equal(direct.predictions, via_runner.predictions)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for coverage-ranked square plotting and training history."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import pandas as pd

from ukbbs.coverage_selection import (
    load_species_training_history,
    select_squares_by_training_coverage,
    train_cutoff_from_runs,
)


class CoverageSelectionTests(unittest.TestCase):
    def test_cutoff_requires_shared_known_forecast_origin(self):
        a = SimpleNamespace(directory="a", run_config={"train_through": 2018})
        b = SimpleNamespace(directory="b", run_config={"forecast_origin": 2018})
        self.assertEqual(train_cutoff_from_runs([a, b]), 2018)
        with self.assertRaisesRegex(ValueError, "different training cutoffs"):
            train_cutoff_from_runs([a, SimpleNamespace(directory="c", run_config={"train_through": 2017})])
        with self.assertRaisesRegex(ValueError, "missing"):
            train_cutoff_from_runs([a, SimpleNamespace(directory="c", run_config={})])
        with self.assertRaisesRegex(ValueError, "Set TRAIN_THROUGH"):
            train_cutoff_from_runs([SimpleNamespace(directory="c", run_config={})])
        self.assertEqual(train_cutoff_from_runs([a], override=2018), 2018)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            train_cutoff_from_runs([a], override=2019)

    def test_loads_only_training_rows_and_real_zeros(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv"
            pd.DataFrame({
                "OS_GRID_REF": ["0001", "0001", "0001", "0002", "0002", "0003"],
                "species": ["CU", "CU", "CU", "CU", "LK", "CU"],
                "year": [2016, 2018, 2019, 2017, 2017, 2016],
                "long_count": [0, 4, 900, None, 10, 2],
            }).to_csv(path, index=False)
            loaded = load_species_training_history(path, species="CU", train_through=2018, target_col="long_count")
            self.assertEqual(len(loaded), 4)
            self.assertEqual(loaded.loc[loaded.OS_GRID_REF.eq("0001"), "year"].tolist(), [2016, 2018])
            self.assertEqual(loaded.loc[loaded.OS_GRID_REF.eq("0001"), "actual"].tolist(), [0, 4])
            forecast = pd.DataFrame({"OS_GRID_REF": ["0001", "0002", "0003"]})
            chosen = select_squares_by_training_coverage(loaded, forecast, n=3)
            self.assertEqual(chosen.OS_GRID_REF.tolist(), ["0001", "0003"])
            self.assertEqual(chosen.n_training_years.tolist(), [2, 1])

    def test_selects_five_rank_positions_using_full_training_history(self):
        rows = []
        for square, years in {
            "A": range(2010, 2020), "B": range(2011, 2020),
            "C": range(2012, 2020), "D": range(2013, 2020),
            "E": range(2014, 2020), "F": range(2015, 2020),
            "G": range(2016, 2020), "H": range(2017, 2020),
            "I": range(2018, 2020), "J": [2019],
        }.items():
            rows.extend({"OS_GRID_REF": square, "year": year, "actual": 0.0}
                        for year in years)
        history = pd.DataFrame(rows)
        predictions = pd.DataFrame({"OS_GRID_REF": list("ABCDEFGHIJ"), "year": [2020] * 10})
        selection = select_squares_by_training_coverage(history, predictions)
        self.assertEqual(selection.OS_GRID_REF.tolist(), ["A", "C", "E", "G", "I"])
        self.assertEqual(selection.rank_percentile_from_top.tolist(), [0, 20, 40, 60, 80])
        self.assertEqual(selection.n_training_years.tolist(), [10, 8, 6, 4, 2])
        self.assertEqual(selection.possible_training_years.tolist(), [10] * 5)

    def test_manual_selection_and_missing_test_records(self):
        history = pd.DataFrame({"OS_GRID_REF": ["A", "B", "C"], "year": [2018] * 3,
                                "actual": [2, 3, 4]})
        forecast = pd.DataFrame({"OS_GRID_REF": ["A", "B"], "year": [2019] * 2})
        chosen = select_squares_by_training_coverage(history, forecast, squares=["B", "A"])
        self.assertEqual(chosen.OS_GRID_REF.tolist(), ["B", "A"])
        with self.assertRaisesRegex(ValueError, "no common observed forecasts"):
            select_squares_by_training_coverage(history, forecast, squares=["C"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            select_squares_by_training_coverage(history, forecast, squares=["A", "A"])


if __name__ == "__main__":
    unittest.main()

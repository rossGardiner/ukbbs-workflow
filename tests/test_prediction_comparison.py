"""Regression tests for comparing per-species predictions saved by model/hash."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ukbbs.prediction_comparison import (
    discover_runs, common_species, choose_squares, compare_saved_predictions,
    matched_species_rows,
)


class SavedPredictionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "predictions"
        self.root.mkdir()
        self.task = {"target": "count", "group_cols": ["OS_GRID_REF", "species"],
                     "time_col": "year"}

    def add_run(self, folder, model, by_species, task=None, run_config=None):
        path = self.root / folder
        path.mkdir()
        manifest = {"hash": folder.rsplit("__", 1)[-1], "identity": {
            "task": task or self.task, "model_name": model, "run_config": run_config or {}}}
        (path / "manifest.json").write_text(json.dumps(manifest))
        for species, frame in by_species.items():
            frame.to_csv(path / f"{species}.csv", index=False)

    @staticmethod
    def table(species="CU", predictions=(1, 3, 6, 7), actuals=(0, 4, 2, None)):
        return pd.DataFrame({
            "OS_GRID_REF": ["A", "A", "B", "B"], "species": [species] * 4,
            "year": [2021, 2022, 2021, 2022],
            "count": actuals, "prediction": predictions,
        })

    def test_scan_all_folders_without_index_and_evaluate_common_rows(self):
        naive = self.table()
        mlp = self.table(predictions=(0, 2, 5, 7))
        # A model doesn't score a year it never predicted. Both must use exact
        # shared observations; a missing *actual* is not equivalent to zero.
        mlp = mlp.loc[~((mlp.OS_GRID_REF == "A") & (mlp.year == 2022))]
        self.add_run("baseline__111111111111", "baseline", {"CU": naive,
                                                              "LK": self.table("LK")})
        self.add_run("mlp__222222222222", "mlp", {"CU": mlp,
                                                  "LK": self.table("LK", predictions=(2, 2, 5, 8))})
        (self.root / "junk").mkdir()
        runs = discover_runs(self.root)
        self.assertEqual(len(runs), 2)
        self.assertEqual(common_species(runs), ["CU", "LK"])
        report = compare_saved_predictions(self.root, plot_species="CU")
        self.assertEqual(report.overall.n_scored.tolist(), [5, 5])  # CU 2 + LK 3
        cu = report.by_species.loc[report.by_species.species == "CU"]
        self.assertEqual(cu.n_scored.tolist(), [2, 2])
        baseline_mae = cu.loc[cu.directory == "baseline__111111111111", "MAE"].iloc[0]
        self.assertAlmostEqual(baseline_mae, 2.5)  # |0-1| + |2-6|
        self.assertEqual(len(report.plot_rows), 2)
        cu_2021 = report.by_year.loc[
            report.by_year["year"].eq(2021)
            & report.by_year.directory.eq("baseline__111111111111")
        ].iloc[0]
        self.assertEqual(cu_2021.n_scored, 4)  # CU A/B + LK A/B
        self.assertAlmostEqual(cu_2021.MAE, 2.5)
        # Real observed zero is not treated as missing.
        self.assertEqual(report.plot_rows.actual.min(), 0)
        self.assertEqual(len(report.by_square), 8)  # 2 squares × 2 species × 2 models

    def test_dataset_disagreement_is_detected(self):
        a = self.table()
        b = self.table(actuals=(9, 4, 2, None))
        self.add_run("a__111111111111", "A", {"CU": a})
        self.add_run("b__222222222222", "B", {"CU": b})
        with self.assertRaisesRegex(ValueError, "Observed counts disagree"):
            compare_saved_predictions(self.root)

    def test_duplicate_keys_and_invalid_predictions_rejected(self):
        a = self.table()
        b = pd.concat([a, a.iloc[:1]])
        self.add_run("a__111111111111", "A", {"CU": b})
        with self.assertRaisesRegex(ValueError, "duplicate forecast keys"):
            compare_saved_predictions(self.root)
        b = a.copy()
        b["prediction"] = b["prediction"].astype(float)
        b.loc[0, "prediction"] = np.inf
        b.to_csv(self.root / "a__111111111111" / "CU.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare_saved_predictions(self.root)

    def test_no_common_observed_years_fails(self):
        self.add_run("a__111111111111", "A", {"CU": self.table().query("year == 2021")})
        self.add_run("b__222222222222", "B", {"CU": self.table().query("year == 2022")})
        with self.assertRaisesRegex(ValueError, "No common observed"):
            compare_saved_predictions(self.root)

    def test_select_subset_and_species_validation(self):
        self.add_run("a__111111111111", "A", {"CU": self.table(), "LK": self.table("LK")})
        self.add_run("b__222222222222", "B", {"CU": self.table()})
        report = compare_saved_predictions(self.root, model_folders=["a__111111111111"],
                                           species=["LK"], plot_species="LK")
        self.assertEqual(report.species, ["LK"])
        self.assertEqual(len(report.runs), 1)
        with self.assertRaisesRegex(ValueError, "not present in every"):
            compare_saved_predictions(self.root, species=["LK"])

    def test_invalid_task_disallowed(self):
        self.add_run("a__111111111111", "A", {"CU": self.table()})
        task = dict(self.task, target="abundance")
        self.add_run("b__222222222222", "B", {"CU": self.table()}, task=task)
        with self.assertRaisesRegex(ValueError, "different target"):
            compare_saved_predictions(self.root)

    def test_square_selection_random_and_manual(self):
        frame = self.table().drop(columns="prediction").rename(columns={"count": "actual"})
        frame = frame.dropna(subset="actual")
        auto = choose_squares(frame, n=5, seed=12)
        self.assertEqual(auto, ["A"])  # only A has 2 years
        self.assertEqual(choose_squares(frame, squares=["B"]), ["B"])
        with self.assertRaisesRegex(ValueError, "no common observed"):
            choose_squares(frame, squares=["Z"])

    def test_empty_plot_species_is_clearly_reported(self):
        self.add_run("a__111111111111", "A", {"CU": self.table(), "LK": self.table("LK").query("year == 2021")})
        self.add_run("b__222222222222", "B", {"CU": self.table(), "LK": self.table("LK").query("year == 2022")})
        with self.assertRaisesRegex(ValueError, "No common observed predictions for plot species"):
            compare_saved_predictions(self.root, plot_species="LK")
        report = compare_saved_predictions(self.root)
        self.assertEqual(report.plot_species, "CU")


if __name__ == "__main__":
    unittest.main()

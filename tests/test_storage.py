"""Offline checks for portable, species-partitioned forecast storage."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ukbbs import (
    ForecastTask,
    LastValueForecaster,
    chronological_split,
    run_forecast,
    save_forecast_results,
)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "predictions"
        self.task = ForecastTask(target="count")
        obs = pd.DataFrame([
            ("A", "CU", 2019, 2.0),
            ("B", "LK", 2019, 4.0),
            ("A", "CU", 2021, 3.0),
            ("B", "LK", 2021, 1.0),
        ], columns=["OS_GRID_REF", "species", "year", "count"])
        train, test = chronological_split(obs, train_through=2019, task=self.task)
        self.result = run_forecast(LastValueForecaster(), train, test, self.task)

    def save(self, result=None, **kwargs):
        options = dict(output_root=self.root, model_name="Last Value", task=self.task,
                       model_params={}, run_config={"train_through": 2019, "dataset": "example"})
        options.update(kwargs)
        return save_forecast_results(self.result if result is None else result, **options)

    def test_species_csvs_manifest_and_lookup(self):
        folder = self.save()
        self.assertRegex(folder.name, r"^last-value__[a-f0-9]{12}$")
        self.assertEqual(set(p.name for p in folder.iterdir()), {"manifest.json", "CU.csv", "LK.csv"})
        for species in ("CU", "LK"):
            assert_frame_equal(
                pd.read_csv(folder / f"{species}.csv"),
                self.result.predictions.loc[self.result.predictions.species.eq(species)].reset_index(drop=True),
                check_dtype=False,
            )
        idx = json.loads((self.root / "index.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(idx["models"]["Last Value"], [
            {"hash": manifest["hash"], "directory": folder.name}
        ])
        self.assertEqual(manifest["identity"]["run_config"]["train_through"], 2019)

    def test_deterministic_and_idempotent(self):
        a = self.save(model_params={"a": 1, "b": 2})
        b = self.save(model_params={"b": 2, "a": 1})
        self.assertEqual(a, b)
        idx = json.loads((self.root / "index.json").read_text())
        self.assertEqual(len(idx["models"]["Last Value"]), 1)

    def test_different_params_run_configuration_and_model_name_get_new_hashes(self):
        base = self.save()
        params = self.save(model_params={"alpha": np.float64(1.5)})
        run = self.save(run_config={"train_through": 2018})
        name = self.save(model_name="Other Model")
        self.assertEqual(len({base, params, run, name}), 4)
        idx = json.loads((self.root / "index.json").read_text())
        self.assertEqual(len(idx["models"]["Last Value"]), 3)
        self.assertEqual(len(idx["models"]["Other Model"]), 1)

    def test_combining_species_in_multiple_writes(self):
        cu = self.result.predictions.query("species == 'CU'").copy()
        lk = self.result.predictions.query("species == 'LK'").copy()
        from ukbbs.evaluation import EvaluationResult
        first = self.save(EvaluationResult(self.result.metrics, cu))
        second = self.save(EvaluationResult(self.result.metrics, lk))
        self.assertEqual(first, second)
        self.assertTrue((first / "CU.csv").exists())
        self.assertTrue((first / "LK.csv").exists())

    def test_conflicting_species_needs_explicit_overwrite(self):
        folder = self.save()
        revised = self.result.predictions.copy()
        revised.loc[revised.species.eq("CU"), "prediction"] = 999.0
        from ukbbs.evaluation import EvaluationResult
        changed = EvaluationResult(self.result.metrics, revised)
        before = (folder / "CU.csv").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
            self.save(changed)
        self.assertEqual((folder / "CU.csv").read_bytes(), before)
        self.save(changed, overwrite=True)
        self.assertEqual(pd.read_csv(folder / "CU.csv")["prediction"].tolist(), [999.0])

    def test_invalid_species_and_configs_are_rejected(self):
        from ukbbs.evaluation import EvaluationResult
        invalid = self.result.predictions.copy()
        invalid.loc[invalid.species.eq("CU"), "species"] = "../../escape"
        with self.assertRaisesRegex(ValueError, "Species codes"):
            self.save(EvaluationResult(self.result.metrics, invalid))
        with self.assertRaisesRegex(ValueError, "JSON serializable"):
            self.save(model_params={"bad": object()})
        with self.assertRaisesRegex(ValueError, "species"):
            self.save(EvaluationResult(self.result.metrics, self.result.predictions.drop(columns="species")))


if __name__ == "__main__":
    unittest.main()

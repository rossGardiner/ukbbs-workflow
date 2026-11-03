"""Offline tests of HadUK-Grid extraction, lags and fixed-origin joins.

Run with: python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Transformer

from ukbbs import ForecastTask, LastValueForecaster, chronological_split, run_forecast
from ukbbs.haduk import extract_haduk_annual, find_haduk_annual_files
from ukbbs.climate_features import climate_lag_features, attach_fixed_origin_climate
from ukbbs.storage import save_forecast_results


class ClimateFeatureTests(unittest.TestCase):
    def test_lag_is_previous_calendar_year_even_with_gaps(self):
        data = pd.DataFrame({
            "OS_GRID_REF": ["A"] * 4, "year": [2015, 2016, 2017, 2019],
            "temp_mean_c": [15, 16, 17, 19],
        })
        features = climate_lag_features(data, target_years=[2020], history_years=5)
        year2020 = features.loc[features.year.eq(2020)].iloc[0]
        self.assertEqual(year2020.climate_temp_mean_c_lag1, 19)
        self.assertTrue(pd.isna(year2020.climate_temp_mean_c_lag2))
        self.assertEqual(year2020.climate_temp_mean_c_lag3, 17)
        self.assertEqual(year2020.climate_temp_mean_c_lag5, 15)

    def test_frozen_origin_cannot_use_later_climate(self):
        source = pd.DataFrame({"OS_GRID_REF": ["A"] * 6,
                               "year": [2015, 2016, 2017, 2018, 2019, 2020],
                               "rainfall_mm": [100, 101, 102, 103, 104, 9999]})
        train = pd.DataFrame({"OS_GRID_REF": ["A"], "species": ["CU"],
                              "year": [2019], "count": [4]})
        test = pd.DataFrame({"OS_GRID_REF": ["A", "A"], "species": ["CU", "CU"],
                             "year": [2020, 2022], "count": [6, 7]})
        tr, te = attach_fixed_origin_climate(train, test, source,
                                              forecast_origin=2019, history_years=5)
        self.assertEqual(tr.climate_rainfall_mm_lag1.tolist(), [103])
        self.assertEqual(te.climate_rainfall_mm_lag1.tolist(), [104, 104])
        self.assertEqual(te.climate_rainfall_mm_lag5.tolist(), [100, 100])
        _, te2 = attach_fixed_origin_climate(train, test, source[source.year.le(2019)],
                                              forecast_origin=2019, history_years=5)
        assert_frame_equal(te, te2)

    def test_invalid_climate_inputs(self):
        table = pd.DataFrame({"OS_GRID_REF": ["A", "A"],
                              "year": [2018, 2018], "temperature": [1., 2.]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            climate_lag_features(table)
        with self.assertRaisesRegex(ValueError, "history_years"):
            climate_lag_features(table.iloc[:1], history_years=0)
        with self.assertRaisesRegex(ValueError, "finite integers"):
            climate_lag_features(table.assign(year=[2018.5, 2019]))

    def test_invalid_forecast_origin(self):
        annual = pd.DataFrame({"OS_GRID_REF": ["A"], "year": [2018], "temp": [8.]})
        train = pd.DataFrame({"OS_GRID_REF": ["A"], "year": [2020]})
        test = pd.DataFrame({"OS_GRID_REF": ["A"], "year": [2021]})
        with self.assertRaisesRegex(ValueError, "Forecast origin"):
            attach_fixed_origin_climate(train, test, annual, forecast_origin=2019)


class HadUKNetCDFTests(unittest.TestCase):
    def setUp(self):
        try:
            import xarray  # noqa: F401
            import scipy  # noqa: F401  (create synthetic NetCDF files)
        except ImportError:
            self.skipTest("Install xarray and scipy to test synthetic NetCDF extraction")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.had = self.root / "data" / "had"
        converter = Transformer.from_crs("EPSG:27700", "EPSG:4258", always_xy=True)
        lon, lat = converter.transform([400500, 401500], [200500, 200500])
        self.locations = pd.DataFrame({"OS_GRID_REF": ["A", "B"],
                                       "ETRS89Long": lon, "ETRS89Lat": lat,
                                       # These are intentionally WRONG for HadUK.
                                       "easting": [4e6, 4e6],
                                       "northing": [2e6, 2e6]})

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def create_files(self, years=(2018, 2019), variables=("tas", "rainfall", "groundfrost")):
        import xarray as xr
        for year in years:
            for var in variables:
                path = (self.had / var / "ann" / "v20260512" /
                        f"{var}_hadukgrid_uk_1km_ann_{year}01-{year}12.nc")
                path.parent.mkdir(parents=True, exist_ok=True)
                base = {"tas": 8., "rainfall": 700., "groundfrost": 13.}[var]
                matrix = np.array([[base + year - 2018, base + 1 + year - 2018],
                                   [base + 2, np.nan]])
                ds = xr.Dataset(
                    {var: (("time", "projection_y_coordinate", "projection_x_coordinate"),
                           matrix[None, :, :])},
                    coords={"time": [np.datetime64(f"{year}-07-01")],
                            "projection_y_coordinate": [200500., 201500.],
                            "projection_x_coordinate": [400500., 401500.]},
                )
                ds.to_netcdf(path, engine="scipy")

    def test_extract_haduk_coords_years_and_optional_frost(self):
        self.create_files()
        annual = extract_haduk_annual(self.locations, had_dir=self.had,
                                      start_year=2018, end_year=2019,
                                      variables=("tas", "rainfall", "groundfrost"))
        self.assertEqual(len(annual), 4)
        self.assertEqual(annual.query("OS_GRID_REF == 'A' and year == 2019")
                         .temp_mean_c.iloc[0], 9.)
        self.assertEqual(annual.query("OS_GRID_REF == 'B' and year == 2019")
                         .precip_total_mm.iloc[0], 702.)
        self.assertIn("ground_frost_days", annual.columns)
        self.assertNotIn("air_frost_days", annual.columns)

    def test_only_requested_variables_need_be_downloaded(self):
        self.create_files(variables=("tas", "rainfall"))
        # Incomplete optional variable must not block the default two.
        (self.had / "groundfrost_hadukgrid_uk_1km_ann_201801-201812.nc").touch()
        found = find_haduk_annual_files(self.had, start_year=2018, end_year=2019)
        self.assertEqual(len(found), 4)

    def test_missing_year_and_empty_download(self):
        self.create_files(years=(2018,))
        with self.assertRaisesRegex(FileNotFoundError, "Missing 2 annual"):
            find_haduk_annual_files(self.had, start_year=2018, end_year=2019)
        (self.had / "tas" / "ann" / "v20260512" /
         "tas_hadukgrid_uk_1km_ann_201801-201812.nc").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty HadUK"):
            find_haduk_annual_files(self.had, start_year=2018, end_year=2018)

    def test_wrong_region_archive_rejected(self):
        self.had.mkdir(parents=True)
        (self.had / "tas_hadukgrid_uk_region_ann_201801-201812.nc").write_bytes(b"arbitrary")
        with self.assertRaisesRegex(FileNotFoundError, "Missing"):
            find_haduk_annual_files(self.had, start_year=2018, end_year=2018)

    def test_finite_grid_mask_off_domain(self):
        self.create_files(years=(2018,))
        far = self.locations.copy()
        far.loc[1, "ETRS89Lat"] = 61.9
        extracted = extract_haduk_annual(far, had_dir=self.had,
                                         start_year=2018, end_year=2018)
        self.assertTrue(pd.isna(extracted.query("OS_GRID_REF == 'B'").temp_mean_c.iloc[0]))

    def test_forecast_integration_and_export(self):
        self.create_files(years=(2016, 2017, 2018, 2019))
        annual = extract_haduk_annual(self.locations, had_dir=self.had,
                                      start_year=2016, end_year=2019)
        obs = pd.DataFrame({"OS_GRID_REF": ["A", "A", "A", "A"],
                            "species": ["CU"] * 4, "year": [2017, 2018, 2019, 2020],
                            "count": [4., 5., 6., 7.]})
        task = ForecastTask(target="count")
        train, test = chronological_split(obs, train_through=2019, task=task)
        tr, te = attach_fixed_origin_climate(train, test, annual,
                                             forecast_origin=2019, history_years=2)
        self.assertEqual(te.climate_temp_mean_c_lag1.tolist(), [9.])
        report = run_forecast(LastValueForecaster(), tr, te, task)
        self.assertEqual(report.predictions.prediction.tolist(), [6.])
        folder = save_forecast_results(report, output_root=self.root / "predictions",
                                       model_name="last_value", model_params={},
                                       run_config={"origin": 2019}, task=task)
        self.assertTrue((folder / "CU.csv").is_file())


if __name__ == "__main__":
    unittest.main()

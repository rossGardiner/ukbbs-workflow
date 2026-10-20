"""Offline checks of aggregation semantics, chunk boundaries, and joins."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Transformer

from ukbbs.preprocessing import (
    COUNT_COLUMN, SECTIONS, attach_coordinates, coordinate_lookup,
    download_release, file_md5, preprocess_counts, survey_coverage,
)
from unittest.mock import patch


def row(year=1999, square="HU3656", visit="E", species="CU", band="1", a=0, b=0):
    return dict(year=year, square=square, visit=visit, species_code=species,
                distance_band=band, **dict(zip(SECTIONS, [a, b] + [0] * 8)))


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, rows, name="test.csv"):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_sum_then_max_with_split_visits_flight_and_exclusions(self):
        # Early = 2+3+4+5=14; late = 12. Max of bands (12) is wrong.
        path = self.csv([
            row(a=2, b=3), row(band="2", a=4), row(band="F", a=5),
            row(visit="L", a=12), row(year=2001, a=100), row(year=2020, a=100),
            row(year=2002, a=7), row(species="GP", a=9),
        ])
        tiny, stats = preprocess_counts(path, chunksize=1)
        large, _ = preprocess_counts(path, chunksize=100)
        assert_frame_equal(tiny, large)
        self.assertEqual(tiny.loc[(tiny.species == "CU") & (tiny.year == 1999), COUNT_COLUMN].item(), 14)
        self.assertEqual(len(tiny), 3)
        self.assertEqual(stats["excluded_rows"], 2)
        self.assertIn("F", stats["distance_bands"])

    def test_missing_sections_match_r_na_rm(self):
        missing = row()
        missing.update(dict.fromkeys(SECTIONS, np.nan))
        path = self.csv([missing, row(year=2002, a=np.nan, b=3)])
        model, stats = preprocess_counts(path, chunksize=1)
        self.assertEqual(model[COUNT_COLUMN].tolist(), [0, 3])
        self.assertEqual(stats["all_missing_section_rows"], 1)
        self.assertEqual(stats["missing_section_cells"], 11)

    def test_unlabelled_distance_bands_still_contribute_counts(self):
        model, stats = preprocess_counts(self.csv([row(band="", a=3), row(band="F", a=4)]))
        self.assertEqual(model[COUNT_COLUMN].item(), 7)
        self.assertEqual(stats["missing_distance_band_rows"], 1)
        self.assertIn("(missing)", stats["distance_bands"])

    def test_reject_bad_schema_and_invalid_counts(self):
        for invalid in [-1, 1.5, float("inf"), "oops"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    preprocess_counts(self.csv([row(a=invalid)]))
        path = self.csv([{"year": 1999}])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            preprocess_counts(path)

    def test_reject_missing_key(self):
        with self.assertRaisesRegex(ValueError, "Missing grouping key"):
            preprocess_counts(self.csv([row(species="")]))

    def test_no_implicit_zero_completion(self):
        model, _ = preprocess_counts(self.csv([row(year=1999), row(year=2002)]))
        self.assertEqual(model.year.tolist(), [1999, 2002])

    def test_coverage_uses_visits_and_retains_globally_missing_years(self):
        path = self.csv([
            {"square": "HU3656", "year": 1999}, {"square": "HU3656", "year": 1999},
            {"square": "HU3656", "year": 2002}, {"square": "WV7052", "year": 1999},
            {"square": "HU3656", "year": 2001},
        ])
        sampled, coverage, years = survey_coverage(path)
        self.assertEqual(years, [1999, 2000, 2002])
        self.assertEqual(len(sampled), 3)
        self.assertEqual(coverage.n_years_sampled.tolist(), [2, 1])
        self.assertFalse(coverage.sampled_every_retained_year.any())

    def lookup_rows(self):
        return [
            dict(square="HU3656", ETRS89Long=-1.33972632299, ETRS89Lat=60.2915782197, BBS_region="SHETX"),
            dict(square="IJ3371", ETRS89Long=-5.9353324943, ETRS89Lat=54.5740717628, BBS_region="ANTRX"),
            dict(square="WV7052", ETRS89Long=-2.031827, ETRS89Lat=49.221226, BBS_region="CHANX"),
        ]

    def test_coordinates_cover_gb_ni_channel_islands_and_axis_order(self):
        locations = coordinate_lookup(self.csv(self.lookup_rows()))
        self.assertEqual(locations.grid_system.tolist(), ["British National Grid", "Irish Grid", "Channel Islands grid"])
        self.assertTrue(locations.easting.between(3e6, 4e6).all())
        self.assertTrue(locations.northing.between(2.5e6, 4.5e6).all())
        # Independent projection anchor: the LAEA origin maps to false E/N.
        origin = self.lookup_rows()[0] | dict(ETRS89Long=10.0, ETRS89Lat=52.0)
        projected = coordinate_lookup(self.csv([origin], "origin.csv"))
        self.assertAlmostEqual(projected.easting.item(), 4321000, places=5)
        self.assertAlmostEqual(projected.northing.item(), 3210000, places=5)
        inverse = Transformer.from_crs(3035, 4258, always_xy=True)
        lon, lat = inverse.transform(locations.easting, locations.northing)
        np.testing.assert_allclose(lon, [x["ETRS89Long"] for x in self.lookup_rows()], atol=1e-7)
        np.testing.assert_allclose(lat, [x["ETRS89Lat"] for x in self.lookup_rows()], atol=1e-7)

    def test_join_keeps_row_count_and_fails_missing_coordinates(self):
        locations = coordinate_lookup(self.csv(self.lookup_rows()))
        model = pd.DataFrame({"OS_GRID_REF": ["HU3656", "HU3656", "WV7052"]})
        self.assertEqual(len(attach_coordinates(model, locations)), 3)
        with self.assertRaisesRegex(ValueError, "No publisher coordinates"):
            attach_coordinates(pd.DataFrame({"OS_GRID_REF": ["UNKNOWN"]}), locations)

    def test_lookup_rejects_duplicate_or_bad_coordinates(self):
        with self.assertRaisesRegex(ValueError, "duplicate square"):
            coordinate_lookup(self.csv([self.lookup_rows()[0]] * 2))
        with self.assertRaisesRegex(ValueError, "invalid longitude/latitude"):
            coordinate_lookup(self.csv([self.lookup_rows()[0] | {"ETRS89Lat": 120}]))

    def test_download_verifies_cache_without_network(self):
        path = self.root / "cached.csv"
        path.write_text("known bytes")
        with patch("ukbbs.preprocessing.FILE_MD5", {path.name: file_md5(path)}), patch("ukbbs.preprocessing.urlopen") as request:
            self.assertEqual(download_release(self.root), {path.name: path})
            request.assert_not_called()
        with patch("ukbbs.preprocessing.FILE_MD5", {path.name: "incorrect"}):
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                download_release(self.root)


if __name__ == "__main__":
    unittest.main()

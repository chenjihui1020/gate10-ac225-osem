from __future__ import annotations

import math
import unittest

import numpy as np

from gate10_model.detector_response import (
    DetectorResponseConfig,
    SINGLES_DTYPE,
    build_compton_events,
    build_crystal_singles,
)


class DetectorResponseTests(unittest.TestCase):
    def test_build_crystal_singles_groups_hits_and_applies_mppc_response(self) -> None:
        config = DetectorResponseConfig(
            light_yield_per_mev=50_000.0,
            collection_efficiency=0.5,
            pde=0.4,
            microcell_count=10_000,
            electronics_noise_pe=0.0,
            energy_threshold_mev=0.0,
            random_seed=123,
        )
        arrays = {
            "EventID": np.array([1, 1, 1], dtype=np.int32),
            "PreStepUniqueVolumeID": np.array(
                [
                    b"scatter_thin_00_pixel_grid-0_0_0_5",
                    b"scatter_thin_00_pixel_grid-0_0_0_5",
                    b"absorber_00_pixel_grid-0_0_0_7",
                ]
            ),
            "PostPosition_X": np.array([10.0, 20.0, 40.0]),
            "PostPosition_Y": np.array([0.0, 0.0, 0.0]),
            "PostPosition_Z": np.array([0.0, 0.0, 0.0]),
            "TotalEnergyDeposit": np.array([0.1, 0.3, 0.2]),
            "GlobalTime": np.array([5.0, 4.0, 8.0]),
        }

        singles = build_crystal_singles(arrays, config)

        self.assertEqual(len(singles), 2)
        scatter = singles[singles["layer"] == "scatter"][0]
        self.assertEqual(scatter["energy_deposit_mev"], 0.4)
        self.assertEqual(scatter["hit_count"], 2)
        self.assertEqual(scatter["time"], 4.0)
        self.assertTrue(math.isclose(scatter["x"], 17.5))
        self.assertEqual(scatter["optical_photons_mean"], 20_000.0)
        self.assertEqual(scatter["photoelectrons_mean"], 4_000.0)
        self.assertTrue(scatter["passed_threshold"])

    def test_build_compton_events_pairs_scatter_and_absorber(self) -> None:
        singles = np.array(
            [
                (
                    1,
                    11,
                    0,
                    11,
                    "scatter_thin_00_pixel_grid-0_0_0_5",
                    "scatter",
                    0,
                    5,
                    0.20,
                    0.20,
                    10.0,
                    0.0,
                    0.0,
                    4.0,
                    1,
                    10_000.0,
                    2_000.0,
                    1800.0,
                    0.18,
                    0.18,
                    True,
                ),
                (
                    1,
                    11,
                    0,
                    11,
                    "absorber_00_pixel_grid-0_0_0_7",
                    "absorber",
                    0,
                    7,
                    0.30,
                    0.30,
                    40.0,
                    0.0,
                    0.0,
                    6.0,
                    1,
                    15_000.0,
                    3_000.0,
                    2700.0,
                    0.27,
                    0.27,
                    True,
                ),
            ],
            dtype=SINGLES_DTYPE,
        )

        events = build_compton_events(singles, coincidence_window=10.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], 1)
        self.assertEqual(events[0]["energy_total_mev"], 0.5)
        self.assertEqual(events[0]["delta_time"], 2.0)
        self.assertTrue(0.0 <= events[0]["compton_angle_deg"] <= 180.0)


if __name__ == "__main__":
    unittest.main()

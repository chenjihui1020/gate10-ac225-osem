from __future__ import annotations

import unittest

import numpy as np

from gate10_model.osem_reconstruction import (
    OSEMConfig,
    build_image_grid,
    compute_event_weights,
    run_osem,
    _profile_fwhm,
    _subset_indices,
)


class OSEMReconstructionTests(unittest.TestCase):
    def test_compute_event_weights_prefers_voxels_on_compton_cone(self) -> None:
        config = OSEMConfig(image_size=5, extent_mm=20.0, sigma_angle_deg=5.0)
        grid = build_image_grid(config)
        event = {
            "scatter_x": 10.0,
            "scatter_y": 0.0,
            "scatter_z": 0.0,
            "absorber_x": 20.0,
            "absorber_y": 0.0,
            "absorber_z": 0.0,
            "compton_angle_deg": 0.0,
        }

        weights = compute_event_weights(event, grid, config)
        on_axis_index = np.argmin((grid["x"] - 0.0) ** 2 + (grid["y"] - 0.0) ** 2)
        off_axis_index = np.argmin((grid["x"] - 0.0) ** 2 + (grid["y"] - 10.0) ** 2)

        self.assertGreater(weights[on_axis_index], weights[off_axis_index])

    def test_run_osem_returns_finite_normalized_image(self) -> None:
        config = OSEMConfig(image_size=16, extent_mm=40.0, iterations=2, subsets=2)
        grid = build_image_grid(config)
        events = [
            {
                "scatter_x": 10.0,
                "scatter_y": 0.0,
                "scatter_z": 0.0,
                "absorber_x": 20.0,
                "absorber_y": 0.0,
                "absorber_z": 0.0,
                "compton_angle_deg": 0.0,
            },
            {
                "scatter_x": 0.0,
                "scatter_y": 10.0,
                "scatter_z": 0.0,
                "absorber_x": 0.0,
                "absorber_y": 20.0,
                "absorber_z": 0.0,
                "compton_angle_deg": 0.0,
            },
        ]

        image = run_osem(events, grid, config)

        self.assertEqual(image.shape, (16, 16))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(float(image.max()), 0.0)
        self.assertAlmostEqual(float(image.sum()), 1.0, places=6)

    def test_subset_indices_use_strided_balanced_partitioning(self) -> None:
        subsets = list(_subset_indices(10, 3))

        self.assertTrue(np.array_equal(subsets[0], np.array([0, 3, 6, 9])))
        self.assertTrue(np.array_equal(subsets[1], np.array([1, 4, 7])))
        self.assertTrue(np.array_equal(subsets[2], np.array([2, 5, 8])))

    def test_profile_fwhm_interpolates_half_max_width(self) -> None:
        axis = np.linspace(-10.0, 10.0, 201)
        sigma = 2.0
        profile = np.exp(-0.5 * (axis / sigma) ** 2)

        width = _profile_fwhm(axis, profile)

        self.assertAlmostEqual(width, 2.355 * sigma, delta=0.05)

    def test_symmetric_synthetic_events_reconstruct_near_source(self) -> None:
        def angle(source, scatter, absorber):
            incident = np.array(scatter) - np.array(source)
            scattered = np.array(absorber) - np.array(scatter)
            cos_theta = np.dot(incident, scattered) / (
                np.linalg.norm(incident) * np.linalg.norm(scattered)
            )
            return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

        source = (0.0, 0.0, 0.0)
        events = []
        for phi in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
            scatter = (40.0 * np.cos(phi), 40.0 * np.sin(phi), 0.0)
            for delta_phi in (0.3, -0.35):
                absorber = (
                    65.0 * np.cos(phi + delta_phi),
                    65.0 * np.sin(phi + delta_phi),
                    0.0,
                )
                events.append(
                    {
                        "scatter_x": scatter[0],
                        "scatter_y": scatter[1],
                        "scatter_z": scatter[2],
                        "absorber_x": absorber[0],
                        "absorber_y": absorber[1],
                        "absorber_z": absorber[2],
                        "compton_angle_deg": angle(source, scatter, absorber),
                    }
                )

        config = OSEMConfig(image_size=65, extent_mm=50.0, iterations=8, subsets=8, sigma_angle_deg=3.0)
        grid = build_image_grid(config)
        image = run_osem(events, grid, config)
        axis = np.linspace(-config.extent_mm, config.extent_mm, config.image_size)
        peak = np.unravel_index(int(np.argmax(image)), image.shape)

        self.assertLessEqual(abs(axis[peak[1]]), 5.0)
        self.assertLessEqual(abs(axis[peak[0]]), 5.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "gate10_output"
    / "run_8threads_20min_test_detector_response"
    / "compton_events.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "run_8threads_20min_test_osem"


@dataclass(frozen=True)
class OSEMConfig:
    image_size: int = 128
    extent_mm: float = 80.0
    z_slice_mm: float = 0.0
    iterations: int = 8
    subsets: int = 8
    sigma_angle_deg: float = 6.0
    distance_power: float = 0.0
    min_event_energy_mev: float = 0.0
    max_event_energy_mev: float = math.inf
    epsilon: float = 1e-12


def build_image_grid(config: OSEMConfig) -> dict[str, np.ndarray]:
    if config.image_size <= 1:
        raise ValueError("image_size must be > 1.")
    if config.extent_mm <= 0:
        raise ValueError("extent_mm must be positive.")
    axis = np.linspace(-config.extent_mm, config.extent_mm, config.image_size)
    x, y = np.meshgrid(axis, axis)
    z = np.full_like(x, config.z_slice_mm, dtype=float)
    return {
        "x": x.ravel(),
        "y": y.ravel(),
        "z": z.ravel(),
        "axis": axis,
        "shape": np.array([config.image_size, config.image_size], dtype=int),
    }


def _event_value(event: dict | np.void, key: str) -> float:
    return float(event[key])


def compute_event_weights(event: dict | np.void, grid: dict[str, np.ndarray], config: OSEMConfig) -> np.ndarray:
    scatter = np.array(
        [
            _event_value(event, "scatter_x"),
            _event_value(event, "scatter_y"),
            _event_value(event, "scatter_z"),
        ],
        dtype=float,
    )
    absorber = np.array(
        [
            _event_value(event, "absorber_x"),
            _event_value(event, "absorber_y"),
            _event_value(event, "absorber_z"),
        ],
        dtype=float,
    )
    axis = absorber - scatter
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= config.epsilon:
        return np.zeros_like(grid["x"], dtype=float)
    scattered_direction = axis / axis_norm

    voxel_to_scatter = np.column_stack(
        [
            scatter[0] - grid["x"],
            scatter[1] - grid["y"],
            scatter[2] - grid["z"],
        ]
    )
    distances = np.linalg.norm(voxel_to_scatter, axis=1)
    valid = distances > config.epsilon
    directions = np.zeros_like(voxel_to_scatter)
    directions[valid] = voxel_to_scatter[valid] / distances[valid, None]

    cos_angles = np.clip(directions @ scattered_direction, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))
    residual = angles - _event_value(event, "compton_angle_deg")
    weights = np.exp(-0.5 * (residual / config.sigma_angle_deg) ** 2)
    weights[~valid] = 0.0
    if config.distance_power > 0:
        weights[valid] /= np.power(distances[valid], config.distance_power)
    return weights.astype(float)


def _event_passes_filters(event: dict[str, str], config: OSEMConfig) -> bool:
    if event.get("valid_compton_angle", "False") not in {"True", "true", "1", "yes"}:
        return False
    e_total = float(event["energy_total_mev"])
    return config.min_event_energy_mev <= e_total <= config.max_event_energy_mev


def read_compton_events(path: Path, config: OSEMConfig) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _event_passes_filters(row, config):
                continue
            events.append(
                {
                    "scatter_x": float(row["scatter_x"]),
                    "scatter_y": float(row["scatter_y"]),
                    "scatter_z": float(row["scatter_z"]),
                    "absorber_x": float(row["absorber_x"]),
                    "absorber_y": float(row["absorber_y"]),
                    "absorber_z": float(row["absorber_z"]),
                    "compton_angle_deg": float(row["compton_angle_deg"]),
                    "energy_total_mev": float(row["energy_total_mev"]),
                }
            )
    return events


def _subset_indices(n_events: int, subsets: int) -> Iterable[np.ndarray]:
    subsets = max(1, min(subsets, n_events))
    indices = np.arange(n_events)
    for offset in range(subsets):
        subset = indices[offset::subsets]
        if len(subset) > 0:
            yield subset


def run_osem(events: list[dict[str, float]], grid: dict[str, np.ndarray], config: OSEMConfig) -> np.ndarray:
    if not events:
        raise ValueError("No valid Compton events available for OSEM reconstruction.")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive.")
    if config.sigma_angle_deg <= 0:
        raise ValueError("sigma_angle_deg must be positive.")

    n_voxels = len(grid["x"])
    image = np.ones(n_voxels, dtype=float)
    image /= image.sum()

    event_weights = [compute_event_weights(event, grid, config) for event in events]
    event_weights = [w / (w.sum() + config.epsilon) for w in event_weights]
    full_sensitivity = np.vstack(event_weights).sum(axis=0) + config.epsilon

    for _ in range(config.iterations):
        for subset in _subset_indices(len(events), config.subsets):
            weights = np.vstack([event_weights[i] for i in subset])
            denominators = weights @ image + config.epsilon
            backprojection = (weights / denominators[:, None]).sum(axis=0)
            sensitivity = full_sensitivity * (len(subset) / len(events)) + config.epsilon
            image *= backprojection / sensitivity
            total = image.sum()
            if total > 0:
                image /= total

    return image.reshape((config.image_size, config.image_size))


def _save_reconstruction_image(image: np.ndarray, config: OSEMConfig, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [-config.extent_mm, config.extent_mm, -config.extent_mm, config.extent_mm]
    im = ax.imshow(image, origin="lower", extent=extent, cmap="inferno")
    ax.set_title("2D Compton-OSEM reconstruction")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, color="#00ccff", linewidth=1.5, label="2 mm source phantom"))
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative activity [a.u.]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _save_profile(image: np.ndarray, config: OSEMConfig, output_path: Path) -> None:
    axis = np.linspace(-config.extent_mm, config.extent_mm, config.image_size)
    center = config.image_size // 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(axis, image[center, :], label="X profile at Y=0")
    ax.plot(axis, image[:, center], label="Y profile at X=0")
    ax.axvspan(-2, 2, color="#00ccff", alpha=0.2, label="2 mm source radius")
    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Relative activity [a.u.]")
    ax.set_title("OSEM central profiles")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _profile_fwhm(axis: np.ndarray, profile: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(profile, dtype=float)
    if len(axis) != len(values) or len(axis) < 2 or not np.isfinite(values).all():
        return math.nan

    values = values - float(values.min())
    peak = float(values.max())
    if peak <= 0:
        return math.nan

    half_max = 0.5 * peak
    above = np.flatnonzero(values >= half_max)
    if len(above) == 0:
        return math.nan

    def crossing(i0: int, i1: int) -> float:
        x0, x1 = float(axis[i0]), float(axis[i1])
        y0, y1 = float(values[i0]), float(values[i1])
        if y0 == y1:
            return x1
        return x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    left_index = int(above[0])
    right_index = int(above[-1])
    left_x = float(axis[0]) if left_index == 0 else crossing(left_index - 1, left_index)
    right_x = float(axis[-1]) if right_index == len(axis) - 1 else crossing(right_index, right_index + 1)
    return max(0.0, right_x - left_x)


def _write_summary(path: Path, config: OSEMConfig, events: list[dict[str, float]], image: np.ndarray) -> None:
    axis = np.linspace(-config.extent_mm, config.extent_mm, config.image_size)
    peak_index = np.unravel_index(int(np.argmax(image)), image.shape)
    peak_x = axis[peak_index[1]]
    peak_y = axis[peak_index[0]]
    xx, yy = np.meshgrid(axis, axis)
    total = image.sum()
    centroid_x = float((image * xx).sum() / total)
    centroid_y = float((image * yy).sum() / total)
    radius = np.sqrt(xx**2 + yy**2)
    inside_2mm = float(image[radius <= 2.0].sum() / total)
    inside_5mm = float(image[radius <= 5.0].sum() / total)
    peak_fwhm_x = _profile_fwhm(axis, image[peak_index[0], :])
    peak_fwhm_y = _profile_fwhm(axis, image[:, peak_index[1]])
    center_index = int(np.argmin(np.abs(axis)))
    center_fwhm_x = _profile_fwhm(axis, image[center_index, :])
    center_fwhm_y = _profile_fwhm(axis, image[:, center_index])
    with path.open("w", encoding="utf-8") as f:
        f.write("Simplified 2D Compton-OSEM reconstruction.\n")
        f.write("This is not conventional PET LOR-OSEM and not a calibrated quantitative image.\n\n")
        f.write("Configuration:\n")
        for key, value in asdict(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\nData:\n")
        f.write(f"  valid_events_used: {len(events)}\n")
        f.write("\nImage metrics:\n")
        f.write(f"  image_sum: {float(image.sum()):.8g}\n")
        f.write(f"  peak_x_mm: {peak_x:.8g}\n")
        f.write(f"  peak_y_mm: {peak_y:.8g}\n")
        f.write(f"  centroid_x_mm: {centroid_x:.8g}\n")
        f.write(f"  centroid_y_mm: {centroid_y:.8g}\n")
        f.write(f"  activity_fraction_inside_2mm: {inside_2mm:.8g}\n")
        f.write(f"  activity_fraction_inside_5mm: {inside_5mm:.8g}\n")
        f.write(f"  peak_profile_fwhm_x_mm: {peak_fwhm_x:.8g}\n")
        f.write(f"  peak_profile_fwhm_y_mm: {peak_fwhm_y:.8g}\n")
        f.write(f"  source_center_profile_fwhm_x_mm: {center_fwhm_x:.8g}\n")
        f.write(f"  source_center_profile_fwhm_y_mm: {center_fwhm_y:.8g}\n")


def reconstruct(events_file: Path, output_dir: Path, config: OSEMConfig) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = read_compton_events(events_file, config)
    grid = build_image_grid(config)
    image = run_osem(events, grid, config)

    outputs = []
    image_npy = output_dir / "osem_reconstruction.npy"
    np.save(image_npy, image)
    outputs.append(image_npy)

    image_png = output_dir / "osem_reconstruction.png"
    _save_reconstruction_image(image, config, image_png)
    outputs.append(image_png)

    profile_png = output_dir / "osem_profiles.png"
    _save_profile(image, config, profile_png)
    outputs.append(profile_png)

    summary = output_dir / "osem_summary.txt"
    _write_summary(summary, config, events, image)
    outputs.append(summary)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified 2D Compton-OSEM reconstruction from ComptonEvents CSV.")
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=OSEMConfig.image_size)
    parser.add_argument("--extent-mm", type=float, default=OSEMConfig.extent_mm)
    parser.add_argument("--z-slice-mm", type=float, default=OSEMConfig.z_slice_mm)
    parser.add_argument("--iterations", type=int, default=OSEMConfig.iterations)
    parser.add_argument("--subsets", type=int, default=OSEMConfig.subsets)
    parser.add_argument("--sigma-angle-deg", type=float, default=OSEMConfig.sigma_angle_deg)
    parser.add_argument("--distance-power", type=float, default=OSEMConfig.distance_power)
    parser.add_argument("--min-event-energy-mev", type=float, default=OSEMConfig.min_event_energy_mev)
    parser.add_argument("--max-event-energy-mev", type=float, default=OSEMConfig.max_event_energy_mev)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = OSEMConfig(
        image_size=args.image_size,
        extent_mm=args.extent_mm,
        z_slice_mm=args.z_slice_mm,
        iterations=args.iterations,
        subsets=args.subsets,
        sigma_angle_deg=args.sigma_angle_deg,
        distance_power=args.distance_power,
        min_event_energy_mev=args.min_event_energy_mev,
        max_event_energy_mev=args.max_event_energy_mev,
    )
    paths = reconstruct(args.events_file, args.output_dir, cfg)
    for path in paths:
        print(path)
    if args.open:
        import subprocess

        subprocess.run(["open", *[str(p) for p in paths if p.suffix == ".png"]], check=False)

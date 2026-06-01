#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2:
        raise ValueError(f"{path} does not contain a 2D matrix")
    return data


def save_heatmap(
    matrix: np.ndarray,
    path: Path,
    *,
    title: str,
    fov_x_mm: float,
    fov_y_mm: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    extent = [-fov_x_mm / 2.0, fov_x_mm / 2.0, -fov_y_mm / 2.0, fov_y_mm / 2.0]
    im = ax.imshow(matrix, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.55)
    ax.axvline(0.0, color="white", linewidth=0.6, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="normalized activity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fwhm_profile(coords: np.ndarray, values: np.ndarray) -> float | None:
    if values.size == 0 or np.max(values) <= 0.0:
        return None
    peak = int(np.argmax(values))
    half = 0.5 * float(values[peak])
    left = peak
    while left > 0 and values[left] >= half:
        left -= 1
    right = peak
    while right < values.size - 1 and values[right] >= half:
        right += 1
    if left == peak or right == peak:
        return None

    def interp(i0: int, i1: int) -> float:
        y0 = float(values[i0])
        y1 = float(values[i1])
        if y1 == y0:
            return float(coords[i0])
        return float(coords[i0] + (half - y0) * (coords[i1] - coords[i0]) / (y1 - y0))

    return interp(left, left + 1) - interp(right, right - 1)


def save_profiles(matrix: np.ndarray, path: Path, *, fov_x_mm: float, fov_y_mm: float) -> None:
    ny, nx = matrix.shape
    x = np.linspace(-fov_x_mm / 2.0 + fov_x_mm / nx / 2.0, fov_x_mm / 2.0 - fov_x_mm / nx / 2.0, nx)
    y = np.linspace(-fov_y_mm / 2.0 + fov_y_mm / ny / 2.0, fov_y_mm / 2.0 - fov_y_mm / ny / 2.0, ny)
    cy = int(np.argmin(np.abs(y)))
    cx = int(np.argmin(np.abs(x)))
    prof_x = matrix[cy, :]
    prof_y = matrix[:, cx]
    fwhm_x = fwhm_profile(x, prof_x)
    fwhm_y = fwhm_profile(y, prof_y)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(x, prof_x, color="#2266aa")
    axes[0].set_title(f"x profile, FWHM={fwhm_x:.2f} mm" if fwhm_x else "x profile")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("normalized activity")
    axes[1].plot(y, prof_y, color="#aa4422")
    axes[1].set_title(f"y profile, FWHM={fwhm_y:.2f} mm" if fwhm_y else "y profile")
    axes[1].set_xlabel("y [mm]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CUDA OSEM output matrices.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fov-x-mm", type=float, default=100.0)
    parser.add_argument("--fov-y-mm", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    central = load_matrix(args.output_dir / "central_slice.csv")
    mip = load_matrix(args.output_dir / "mip_xy.csv")
    save_heatmap(
        central,
        args.output_dir / "central_slice.png",
        title="CUDA OSEM central slice",
        fov_x_mm=args.fov_x_mm,
        fov_y_mm=args.fov_y_mm,
    )
    save_heatmap(
        mip,
        args.output_dir / "mip_xy.png",
        title="CUDA OSEM XY maximum-intensity projection",
        fov_x_mm=args.fov_x_mm,
        fov_y_mm=args.fov_y_mm,
    )
    save_profiles(central, args.output_dir / "central_profiles.png", fov_x_mm=args.fov_x_mm, fov_y_mm=args.fov_y_mm)
    print(args.output_dir / "central_slice.png")
    print(args.output_dir / "mip_xy.png")
    print(args.output_dir / "central_profiles.png")


if __name__ == "__main__":
    main()

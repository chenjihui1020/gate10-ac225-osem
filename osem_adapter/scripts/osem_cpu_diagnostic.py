#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ELECTRON_REST_KEV = 510.99895


def coord(index: int, n: int, fov: float) -> float:
    return -fov / 2.0 + fov * (index + 0.5) / n


def read_events(path: Path, *, theta_mode: str, max_events: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if theta_mode == "computed":
                theta_deg = float(row["computed_theta_deg"])
            elif theta_mode == "source-geometry":
                theta_deg = float(row["geometric_theta_source_deg"])
            else:
                raise ValueError(f"Unknown theta_mode: {theta_mode}")
            if not math.isfinite(theta_deg):
                continue
            out.append(
                {
                    "sx": float(row["scatter_x_mm"]),
                    "sy": float(row["scatter_y_mm"]),
                    "sz": float(row["scatter_z_mm"]),
                    "ax": float(row["absorber_x_mm"]),
                    "ay": float(row["absorber_y_mm"]),
                    "az": float(row["absorber_z_mm"]),
                    "theta": math.radians(theta_deg),
                    "ei": float(row["incident_energy_keV"]),
                    "es": float(row["scatter_energy_keV"]),
                }
            )
            if max_events > 0 and len(out) >= max_events:
                break
    return out


def synthetic_events() -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for sector in range(8):
        phi = 2.0 * math.pi * sector / 8.0
        for i in range(8):
            z = -8.0 + 16.0 * (i + 0.5) / 8.0
            dphi = 0.25 if i % 2 == 0 else -0.32
            scatter = np.array([39.5 * math.cos(phi), 39.5 * math.sin(phi), z])
            absorber = np.array([67.0 * math.cos(phi + dphi), 67.0 * math.sin(phi + dphi), 0.5 * z])
            incoming = scatter
            outgoing = absorber - scatter
            cos_theta = float(np.dot(incoming, outgoing) / math.sqrt(np.dot(incoming, incoming) * np.dot(outgoing, outgoing)))
            events.append(
                {
                    "sx": float(scatter[0]),
                    "sy": float(scatter[1]),
                    "sz": float(scatter[2]),
                    "ax": float(absorber[0]),
                    "ay": float(absorber[1]),
                    "az": float(absorber[2]),
                    "theta": math.acos(max(-1.0, min(1.0, cos_theta))),
                    "ei": 440.446,
                    "es": 100.0,
                }
            )
    return events


def make_voxels(nx: int, ny: int, nz: int, fov_x: float, fov_y: float, fov_z: float) -> np.ndarray:
    voxels = []
    radius = min(fov_x, fov_y) / 2.0
    for k in range(nz):
        z = coord(k, nz, fov_z)
        for j in range(ny):
            y = coord(j, ny, fov_y)
            for i in range(nx):
                x = coord(i, nx, fov_x)
                if math.hypot(x, y) <= radius:
                    voxels.append((x, y, z))
    return np.array(voxels, dtype=np.float64)


def event_weights(event: dict[str, float], voxels: np.ndarray, sigma_rad: float, *, use_distance: bool) -> np.ndarray:
    scatter = np.array([event["sx"], event["sy"], event["sz"]], dtype=np.float64)
    absorber = np.array([event["ax"], event["ay"], event["az"]], dtype=np.float64)
    incoming = scatter[None, :] - voxels
    outgoing = absorber - scatter
    d1sq = np.sum(incoming * incoming, axis=1)
    d2sq = float(np.dot(outgoing, outgoing))
    valid = (d1sq > 1e-9) & (d2sq > 1e-9)
    out = np.zeros(len(voxels), dtype=np.float64)
    if not np.any(valid):
        return out
    cos_theta = np.clip(incoming[valid].dot(outgoing) / np.sqrt(d1sq[valid] * d2sq), -1.0, 1.0)
    delta = np.arccos(cos_theta) - event["theta"]
    weight = np.exp(-0.5 * delta * delta / (sigma_rad * sigma_rad))
    e_after = event["ei"] - event["es"]
    if event["ei"] > 0.0 and e_after > 0.0:
        ratio = e_after / event["ei"]
        weight *= np.maximum(0.0, ratio * ratio * (ratio + 1.0 / ratio - (1.0 - cos_theta * cos_theta)))
    if use_distance:
        weight /= d1sq[valid] * d2sq + 1e-12
    out[valid] = weight
    return out


def reconstruct(
    events: list[dict[str, float]],
    voxels: np.ndarray,
    *,
    iterations: int,
    subsets: int,
    sigma_deg: float,
    use_sensitivity: bool,
    use_distance: bool,
) -> np.ndarray:
    image = np.ones(len(voxels), dtype=np.float64)
    sigma_rad = math.radians(sigma_deg)
    weights = np.vstack([event_weights(e, voxels, sigma_rad, use_distance=use_distance) for e in events])
    for _ in range(iterations):
        for subset in range(subsets):
            w = weights[subset::subsets]
            if len(w) == 0:
                continue
            denom = w.dot(image) + 1e-300
            back = (w / denom[:, None]).sum(axis=0)
            if use_sensitivity:
                sens = w.sum(axis=0)
                update = np.where(sens > 1e-300, back / sens, 0.0)
            else:
                update = back
            image *= update
            maxv = float(image.max())
            if maxv > 0.0 and math.isfinite(maxv):
                image /= maxv
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU diagnostic for the Ac-225 direct-position OSEM adapter.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--theta-mode", choices=["computed", "source-geometry"], default="computed")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--nx", type=int, default=31)
    parser.add_argument("--ny", type=int, default=31)
    parser.add_argument("--nz", type=int, default=9)
    parser.add_argument("--fov-x-mm", type=float, default=100.0)
    parser.add_argument("--fov-y-mm", type=float, default=100.0)
    parser.add_argument("--fov-z-mm", type=float, default=50.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--subsets", type=int, default=8)
    parser.add_argument("--sigma-deg", type=float, default=6.0)
    parser.add_argument("--use-sensitivity", action="store_true")
    parser.add_argument("--no-distance", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        events = synthetic_events()
    elif args.input:
        events = read_events(args.input, theta_mode=args.theta_mode, max_events=args.max_events)
    else:
        raise SystemExit("Provide --input or --synthetic.")

    if not events:
        raise SystemExit("No events available for reconstruction.")

    voxels = make_voxels(args.nx, args.ny, args.nz, args.fov_x_mm, args.fov_y_mm, args.fov_z_mm)
    image = reconstruct(
        events,
        voxels,
        iterations=args.iterations,
        subsets=args.subsets,
        sigma_deg=args.sigma_deg,
        use_sensitivity=args.use_sensitivity,
        use_distance=not args.no_distance,
    )
    peak = voxels[int(np.argmax(image))]
    centroid = (voxels * image[:, None]).sum(axis=0) / image.sum()
    print(f"events: {len(events)}")
    print(f"use_sensitivity: {args.use_sensitivity}")
    print(f"use_distance: {not args.no_distance}")
    print(f"peak_x_mm: {peak[0]:.8g}")
    print(f"peak_y_mm: {peak[1]:.8g}")
    print(f"peak_z_mm: {peak[2]:.8g}")
    print(f"centroid_x_mm: {centroid[0]:.8g}")
    print(f"centroid_y_mm: {centroid[1]:.8g}")
    print(f"centroid_z_mm: {centroid[2]:.8g}")


if __name__ == "__main__":
    main()

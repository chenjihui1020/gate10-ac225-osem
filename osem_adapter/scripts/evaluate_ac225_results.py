#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ELECTRON_REST_KEV = 510.99895
PAPER_REFERENCE = [
    {"energy_keV": 356.0, "label": "paper 133Ba", "fwhm_x_mm": 14.7, "fwhm_y_mm": 15.3},
    {"energy_keV": 511.0, "label": "paper 22Na", "fwhm_x_mm": 12.0, "fwhm_y_mm": 13.5},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_summary(path: Path) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value
    return out


def energy_resolution_fwhm_frac(energy_kev: np.ndarray | float) -> np.ndarray | float:
    """Empirical GAGG-SiPM/dTOT-like model from the paper: 15.1% at 122 keV and 10.8% at 245 keV."""

    a1 = 0.151 * math.sqrt(122.0)
    a2 = 0.108 * math.sqrt(245.0)
    a = 0.5 * (a1 + a2)
    return a / np.sqrt(np.maximum(energy_kev, 1.0))


def calculated_resolution_from_events(rows: list[dict[str, str]], scatter_radius_mm: float = 39.5) -> float | None:
    values = []
    for row in rows:
        e_inc = float(row["incident_energy_keV"])
        e_s = float(row["scatter_energy_keV"])
        e_a = float(row["absorber_energy_keV"])
        if e_inc <= e_s or e_a <= 0.0:
            continue
        cos_theta = 1.0 - ELECTRON_REST_KEV * (1.0 / (e_inc - e_s) - 1.0 / e_inc)
        if cos_theta <= -1.0 or cos_theta >= 1.0:
            continue
        theta = math.acos(cos_theta)
        sin_theta = abs(math.sin(theta))
        if sin_theta < 1e-6:
            continue
        frac = float(energy_resolution_fwhm_frac(e_a))
        sigma_e = (frac * e_a) / 2.355
        derivative = ELECTRON_REST_KEV / (sin_theta * e_a * e_a)
        sigma_theta = derivative * sigma_e
        fwhm_theta = 2.355 * sigma_theta
        angular_mm = scatter_radius_mm * math.tan(min(fwhm_theta, math.radians(80.0)))
        pixel_mm = 2.5
        values.append(math.sqrt(pixel_mm * pixel_mm + angular_mm * angular_mm))
    if not values:
        return None
    return float(np.median(values))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_fig10(
    rows: list[dict[str, str]],
    output_dir: Path,
    lines_kev: list[float],
    energy_window_frac: float,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scatter = np.array([float(r["scatter_energy_keV"]) for r in rows], dtype=float)
    absorber = np.array([float(r["absorber_energy_keV"]) for r in rows], dtype=float)
    total = scatter + absorber

    max_energy = max(600.0, float(np.percentile(total, 99.0)) if total.size else 600.0)
    bins = np.linspace(0.0, max_energy, 121)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if scatter.size:
        h = ax.hist2d(scatter, absorber, bins=[bins, bins], cmap="magma", cmin=1)
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("counts")
    x = np.linspace(0.0, max_energy, 500)
    colors = ["#20a4f3", "#2ec4b6", "#ff9f1c", "#e71d36"]
    counts: list[dict[str, object]] = []
    for i, line in enumerate(lines_kev):
        color = colors[i % len(colors)]
        y = line - x
        mask = y >= 0.0
        ax.plot(x[mask], y[mask], color=color, linewidth=1.8, label=f"{line:g} keV")
        low = line * (1.0 - energy_window_frac)
        high = line * (1.0 + energy_window_frac)
        y_low = low - x
        y_high = high - x
        mask_low = y_low >= 0.0
        mask_high = y_high >= 0.0
        ax.plot(x[mask_low], y_low[mask_low], color=color, linewidth=0.8, linestyle=":")
        ax.plot(x[mask_high], y_high[mask_high], color=color, linewidth=0.8, linestyle=":")
        n = int(np.count_nonzero((total >= low) & (total <= high)))
        counts.append(
            {
                "line_keV": line,
                "window_low_keV": low,
                "window_high_keV": high,
                "events_in_window": n,
                "fraction_of_all": n / len(rows) if rows else 0.0,
            }
        )

    ax.set_xlim(0.0, max_energy)
    ax.set_ylim(0.0, max_energy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("scatterer energy [keV]")
    ax.set_ylabel("absorber energy [keV]")
    ax.set_title("Scatterer vs absorber energy distribution")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "fig10_scatter_absorber_energy.png", dpi=240)
    plt.close(fig)

    write_rows(
        output_dir / "fig10_energy_window_counts.csv",
        counts,
        ["line_keV", "window_low_keV", "window_high_keV", "events_in_window", "fraction_of_all"],
    )
    return counts


def select_fwhm(summary: dict[str, float | str], axis: str) -> tuple[float | None, str]:
    source_key = f"source_center_profile_fwhm_{axis}_mm"
    peak_key = f"peak_profile_fwhm_{axis}_mm"
    for key, label in [(source_key, "source_center"), (peak_key, "peak")]:
        value = summary.get(key)
        if isinstance(value, float) and value > 0.0 and math.isfinite(value):
            return value, label
    return None, "missing"


def make_fig7(
    converted_dir: Path,
    cuda_output_root: Path,
    output_dir: Path,
    lines_kev: list[float],
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for line in lines_kev:
        line_name = f"line{int(round(line))}"
        sourceqa_summary_path = cuda_output_root / f"ac225_{line_name}_sourceqa20" / "summary.txt"
        sourceqa_events_path = converted_dir / f"ac225_osem_events_{line_name}_sourceqa20.csv"
        if sourceqa_summary_path.exists() and sourceqa_events_path.exists():
            summary = read_summary(sourceqa_summary_path)
            event_rows = read_csv(sourceqa_events_path)
            dataset = f"Ac225 {line_name} source-QA ARM<=20 deg"
        else:
            summary = read_summary(cuda_output_root / f"ac225_{line_name}" / "summary.txt")
            event_rows = read_csv(converted_dir / f"ac225_osem_events_{line_name}.csv")
            dataset = f"Ac225 {line_name}"
        fwhm_x, label_x = select_fwhm(summary, "x")
        fwhm_y, label_y = select_fwhm(summary, "y")
        calc = calculated_resolution_from_events(event_rows)
        rows.append(
            {
                "energy_keV": line,
                "dataset": dataset,
                "valid_events_used": summary.get("valid_events_used", len(event_rows)),
                "fwhm_x_mm": fwhm_x,
                "fwhm_y_mm": fwhm_y,
                "fwhm_mean_mm": (
                    None
                    if fwhm_x is None or fwhm_y is None
                    else 0.5 * (float(fwhm_x) + float(fwhm_y))
                ),
                "profile_x": label_x,
                "profile_y": label_y,
                "calculated_fwhm_mm": calc,
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xvals = [float(r["energy_keV"]) for r in rows if r["fwhm_mean_mm"] is not None]
    yvals = [float(r["fwhm_mean_mm"]) for r in rows if r["fwhm_mean_mm"] is not None]
    if xvals:
        ax.plot(xvals, yvals, "o-", color="#1f77b4", label="current Ac-225 OSEM mean FWHM")
        for r in rows:
            if r["fwhm_x_mm"] is None or r["fwhm_y_mm"] is None:
                continue
            ax.errorbar(
                [float(r["energy_keV"])],
                [float(r["fwhm_mean_mm"])],
                yerr=[
                    [
                        abs(float(r["fwhm_mean_mm"]) - float(r["fwhm_x_mm"])),
                    ],
                    [
                        abs(float(r["fwhm_y_mm"]) - float(r["fwhm_mean_mm"])),
                    ],
                ],
                color="#1f77b4",
                capsize=3,
            )
    calc_x = [float(r["energy_keV"]) for r in rows if r["calculated_fwhm_mm"] is not None]
    calc_y = [float(r["calculated_fwhm_mm"]) for r in rows if r["calculated_fwhm_mm"] is not None]
    if calc_x:
        ax.plot(calc_x, calc_y, "s--", color="#ff7f0e", label="current calculated estimate")

    paper_x = [r["energy_keV"] for r in PAPER_REFERENCE]
    paper_mean = [0.5 * (r["fwhm_x_mm"] + r["fwhm_y_mm"]) for r in PAPER_REFERENCE]
    ax.plot(paper_x, paper_mean, "d:", color="#666666", label="paper measured reference")
    for ref in PAPER_REFERENCE:
        ax.annotate(ref["label"], (ref["energy_keV"], 0.5 * (ref["fwhm_x_mm"] + ref["fwhm_y_mm"])), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.set_xlabel("input gamma-ray energy [keV]")
    ax.set_ylabel("spatial resolution FWHM [mm]")
    ax.set_title("Spatial resolution as a function of gamma-ray energy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_spatial_resolution_vs_energy.png", dpi=240)
    plt.close(fig)

    write_rows(
        output_dir / "fig7_spatial_resolution_summary.csv",
        rows,
        [
            "energy_keV",
            "dataset",
            "valid_events_used",
            "fwhm_x_mm",
            "fwhm_y_mm",
            "fwhm_mean_mm",
            "profile_x",
            "profile_y",
            "calculated_fwhm_mm",
        ],
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style evaluation plots for Ac-225 Gate10/OSEM results.")
    parser.add_argument("--converted-dir", type=Path, default=Path("data"))
    parser.add_argument("--cuda-output-root", type=Path, default=Path("cuda_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--line-kev", type=float, action="append", default=[218.0, 440.446])
    parser.add_argument("--energy-window-frac", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv(args.converted_dir / "ac225_osem_events_all.csv")
    fig10_counts = make_fig10(all_rows, args.output_dir, args.line_kev, args.energy_window_frac)
    fig7_rows = make_fig7(args.converted_dir, args.cuda_output_root, args.output_dir, args.line_kev)
    payload = {
        "converted_dir": str(args.converted_dir),
        "cuda_output_root": str(args.cuda_output_root),
        "fig10": {
            "input_events": len(all_rows),
            "energy_window_counts": fig10_counts,
            "output": str(args.output_dir / "fig10_scatter_absorber_energy.png"),
        },
        "fig7": {
            "rows": fig7_rows,
            "output": str(args.output_dir / "fig7_spatial_resolution_vs_energy.png"),
            "paper_reference": PAPER_REFERENCE,
        },
        "calculated_resolution_note": (
            "Calculated estimate uses the event energy distribution and an empirical GAGG-SiPM energy "
            "resolution model fitted to 15.1% FWHM at 122 keV and 10.8% FWHM at 245 keV from the paper. "
            "It is a current-model diagnostic, not the paper's exact calculation."
        ),
    }
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output_dir / "fig10_scatter_absorber_energy.png")
    print(args.output_dir / "fig7_spatial_resolution_vs_energy.png")
    print(summary_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

ELECTRON_REST_KEV = 510.99895

GEOMETRY_MM = {
    "sectors": 8,
    "sector_angle_deg": 45.0,
    "scatter_center_radius_mm": 39.5,
    "scatter_inner_radius_thin_mm": 38.75,
    "scatter_inner_radius_thick_mm": 37.5,
    "scatter_thin_thickness_mm": 1.5,
    "scatter_thick_thickness_mm": 4.0,
    "absorber_inner_radius_mm": 62.5,
    "absorber_center_radius_mm": 67.0,
    "absorber_thickness_mm": 9.0,
    "pixel_pitch_mm": 3.2,
    "pixel_size_mm": 2.5,
    "pixel_grid": "8x8",
    "gap_material": "BaSO4",
}

DIRECT_HEADER = [
    "event_id",
    "gamma_track_id",
    "scatter_x_mm",
    "scatter_y_mm",
    "scatter_z_mm",
    "absorber_x_mm",
    "absorber_y_mm",
    "absorber_z_mm",
    "scatter_energy_keV",
    "absorber_energy_keV",
    "total_energy_keV",
    "incident_energy_keV",
    "computed_theta_deg",
    "geometric_theta_source_deg",
    "arm_source_deg",
    "scatter_sector",
    "absorber_sector",
    "scatter_pixel",
    "absorber_pixel",
    "delta_time",
]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_pixel(crystal_id: str) -> int:
    match = re.search(r"_(\d+)$", crystal_id)
    return int(match.group(1)) if match else -1


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def theta_from_scatter_energy(scatter_kev: float, incident_kev: float) -> float | None:
    if scatter_kev <= 0.0 or incident_kev <= scatter_kev:
        return None
    scattered_after_kev = incident_kev - scatter_kev
    if scattered_after_kev <= 0.0:
        return None
    cos_theta = 1.0 - ELECTRON_REST_KEV * (
        1.0 / scattered_after_kev - 1.0 / incident_kev
    )
    if cos_theta < -1.0 or cos_theta > 1.0:
        return None
    return math.degrees(math.acos(clamp(cos_theta)))


def geometric_theta(
    scatter: tuple[float, float, float],
    absorber: tuple[float, float, float],
    source: tuple[float, float, float],
) -> float | None:
    vin = tuple(scatter[i] - source[i] for i in range(3))
    vout = tuple(absorber[i] - scatter[i] for i in range(3))
    nin = math.sqrt(sum(v * v for v in vin))
    nout = math.sqrt(sum(v * v for v in vout))
    if nin == 0.0 or nout == 0.0:
        return None
    dot = sum(vin[i] * vout[i] for i in range(3))
    return math.degrees(math.acos(clamp(dot / nin / nout)))


def read_gate_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_row(
    row: dict[str, str],
    *,
    incident_kev: float | None,
    source_xyz: tuple[float, float, float],
) -> dict[str, float | int] | None:
    scatter_kev = float(row["energy_scatter_mev"]) * 1000.0
    absorber_kev = float(row["energy_absorber_mev"]) * 1000.0
    total_kev = scatter_kev + absorber_kev
    ein_kev = total_kev if incident_kev is None else incident_kev
    theta = theta_from_scatter_energy(scatter_kev, ein_kev)
    if theta is None:
        return None

    scatter = (
        float(row["scatter_x"]),
        float(row["scatter_y"]),
        float(row["scatter_z"]),
    )
    absorber = (
        float(row["absorber_x"]),
        float(row["absorber_y"]),
        float(row["absorber_z"]),
    )
    theta_geo = geometric_theta(scatter, absorber, source_xyz)
    arm = math.nan if theta_geo is None else theta - theta_geo

    return {
        "event_id": int(row["event_id"]),
        "gamma_track_id": int(row.get("gamma_track_id", -1)),
        "scatter_x_mm": scatter[0],
        "scatter_y_mm": scatter[1],
        "scatter_z_mm": scatter[2],
        "absorber_x_mm": absorber[0],
        "absorber_y_mm": absorber[1],
        "absorber_z_mm": absorber[2],
        "scatter_energy_keV": scatter_kev,
        "absorber_energy_keV": absorber_kev,
        "total_energy_keV": total_kev,
        "incident_energy_keV": ein_kev,
        "computed_theta_deg": theta,
        "geometric_theta_source_deg": math.nan if theta_geo is None else theta_geo,
        "arm_source_deg": arm,
        "scatter_sector": int(row["scatter_sector"]),
        "absorber_sector": int(row["absorber_sector"]),
        "scatter_pixel": parse_pixel(row["scatter_crystal_id"]),
        "absorber_pixel": parse_pixel(row["absorber_crystal_id"]),
        "delta_time": float(row["delta_time"]),
    }


def select_events(
    rows: Iterable[dict[str, str]],
    *,
    incident_kev: float | None,
    energy_window_frac: float | None,
    min_scatter_kev: float,
    max_scatter_kev: float,
    only_gate_valid_angle: bool,
    source_xyz: tuple[float, float, float],
) -> list[dict[str, float | int]]:
    selected: list[dict[str, float | int]] = []
    for row in rows:
        if only_gate_valid_angle and not parse_bool(row.get("valid_compton_angle", "false")):
            continue
        scatter_kev = float(row["energy_scatter_mev"]) * 1000.0
        if scatter_kev < min_scatter_kev or scatter_kev > max_scatter_kev:
            continue
        total_kev = (float(row["energy_scatter_mev"]) + float(row["energy_absorber_mev"])) * 1000.0
        if incident_kev is not None and energy_window_frac is not None:
            low = incident_kev * (1.0 - energy_window_frac)
            high = incident_kev * (1.0 + energy_window_frac)
            if total_kev < low or total_kev > high:
                continue
        converted = convert_row(row, incident_kev=incident_kev, source_xyz=source_xyz)
        if converted is not None:
            selected.append(converted)
    return selected


def write_direct_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIRECT_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_legacy_coin(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"{int(row['scatter_sector'])} {int(row['scatter_pixel'])} "
                f"{int(row['absorber_sector']) + 8} {int(row['absorber_pixel'])} "
                f"{float(row['scatter_energy_keV']):.6f} "
                f"{float(row['absorber_energy_keV']):.6f} "
                f"{float(row['delta_time']):.9g} {int(row['event_id'])}\n"
            )


def summarize(name: str, rows: list[dict[str, float | int]]) -> dict[str, float | int | str]:
    if not rows:
        return {"name": name, "events": 0}
    total = [float(r["total_energy_keV"]) for r in rows]
    scatter = [float(r["scatter_energy_keV"]) for r in rows]
    arm = [float(r["arm_source_deg"]) for r in rows if math.isfinite(float(r["arm_source_deg"]))]
    out: dict[str, float | int | str] = {
        "name": name,
        "events": len(rows),
        "total_energy_mean_keV": sum(total) / len(total),
        "scatter_energy_mean_keV": sum(scatter) / len(scatter),
        "total_energy_min_keV": min(total),
        "total_energy_max_keV": max(total),
    }
    if arm:
        abs_arm = [abs(v) for v in arm]
        out["arm_source_mean_deg"] = sum(arm) / len(arm)
        out["arm_source_abs_median_deg"] = sorted(abs_arm)[len(abs_arm) // 2]
        out["events_inside_abs_arm_20deg"] = sum(v <= 20.0 for v in abs_arm)
    return out


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gate10 detector-response Compton CSV into Ac-225 CUDA/OSEM inputs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="ac225")
    parser.add_argument("--energy-line-kev", type=float, action="append", default=[218.0, 440.446])
    parser.add_argument("--energy-window-frac", type=float, default=0.10)
    parser.add_argument("--min-scatter-kev", type=float, default=5.0)
    parser.add_argument("--max-scatter-kev", type=float, default=float("inf"))
    parser.add_argument("--source-x-mm", type=float, default=0.0)
    parser.add_argument("--source-y-mm", type=float, default=0.0)
    parser.add_argument("--source-z-mm", type=float, default=0.0)
    parser.add_argument("--source-qa-arm-deg", type=float, default=20.0)
    parser.add_argument("--keep-invalid-gate-angle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_xyz = (args.source_x_mm, args.source_y_mm, args.source_z_mm)

    raw_rows = read_gate_events(args.input)
    only_gate_valid = not args.keep_invalid_gate_angle

    outputs: dict[str, str] = {}
    summaries: list[dict[str, float | int | str]] = []

    all_rows = select_events(
        raw_rows,
        incident_kev=None,
        energy_window_frac=None,
        min_scatter_kev=args.min_scatter_kev,
        max_scatter_kev=args.max_scatter_kev,
        only_gate_valid_angle=only_gate_valid,
        source_xyz=source_xyz,
    )
    all_path = args.output_dir / f"{args.prefix}_osem_events_all.csv"
    write_direct_csv(all_path, all_rows)
    outputs["all"] = str(all_path)
    summaries.append(summarize("all_variable_energy", all_rows))

    legacy_path = args.output_dir / f"{args.prefix}_coin_legacy_reference.txt"
    write_legacy_coin(legacy_path, all_rows)
    outputs["legacy_reference"] = str(legacy_path)

    qa_rows = [
        row for row in all_rows
        if math.isfinite(float(row["arm_source_deg"]))
        and abs(float(row["arm_source_deg"])) <= args.source_qa_arm_deg
    ]
    qa_path = args.output_dir / f"{args.prefix}_osem_events_sourceqa{int(args.source_qa_arm_deg)}.csv"
    write_direct_csv(qa_path, qa_rows)
    outputs["source_qa"] = str(qa_path)
    summaries.append(summarize(f"source_qa_abs_arm_le_{args.source_qa_arm_deg:g}deg", qa_rows))

    for line_keV in args.energy_line_kev:
        line_rows = select_events(
            raw_rows,
            incident_kev=line_keV,
            energy_window_frac=args.energy_window_frac,
            min_scatter_kev=args.min_scatter_kev,
            max_scatter_kev=args.max_scatter_kev,
            only_gate_valid_angle=only_gate_valid,
            source_xyz=source_xyz,
        )
        line_name = f"line{int(round(line_keV))}"
        line_path = args.output_dir / f"{args.prefix}_osem_events_{line_name}.csv"
        write_direct_csv(line_path, line_rows)
        outputs[line_name] = str(line_path)
        summaries.append(summarize(f"{line_keV:g}keV_window", line_rows))

        line_qa_rows = [
            row for row in line_rows
            if math.isfinite(float(row["arm_source_deg"]))
            and abs(float(row["arm_source_deg"])) <= args.source_qa_arm_deg
        ]
        line_qa_path = args.output_dir / (
            f"{args.prefix}_osem_events_{line_name}_sourceqa{int(args.source_qa_arm_deg)}.csv"
        )
        write_direct_csv(line_qa_path, line_qa_rows)
        outputs[f"{line_name}_source_qa"] = str(line_qa_path)
        summaries.append(summarize(f"{line_keV:g}keV_window_source_qa_abs_arm_le_{args.source_qa_arm_deg:g}deg", line_qa_rows))

    summary = {
        "gate_input": str(args.input),
        "input_rows": len(raw_rows),
        "only_gate_valid_angle": only_gate_valid,
        "energy_window_frac_for_lines": args.energy_window_frac,
        "scatter_energy_window_keV": [
            args.min_scatter_kev,
            args.max_scatter_kev if math.isfinite(args.max_scatter_kev) else None,
        ],
        "source_xyz_mm_for_qa_only": list(source_xyz),
        "geometry_mm": GEOMETRY_MM,
        "outputs": outputs,
        "event_sets": summaries,
        "notes": [
            "The all file uses measured scatter+absorber energy as per-event incident energy.",
            "Line files use fixed incident energies and require total reconstructed energy inside the configured window.",
            "source_qa output uses the known simulated source at the origin and is for validation, not blind reconstruction.",
        ],
    }
    summary_path = args.output_dir / f"{args.prefix}_osem_conversion_summary.json"
    write_summary(summary_path, summary)
    print(summary_path)
    for item in summaries:
        print(f"{item['name']}: {item['events']} events")


if __name__ == "__main__":
    main()

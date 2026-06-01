from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "gate10_output" / "small_run" / "ac225_hits.root"
DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "small_run_detector_response"

ELECTRON_REST_MEV = 0.51099895

SINGLES_DTYPE = [
    ("event_id", "i8"),
    ("track_id", "i8"),
    ("parent_id", "i8"),
    ("gamma_track_id", "i8"),
    ("crystal_id", "U160"),
    ("layer", "U16"),
    ("sector", "i4"),
    ("pixel_index", "i4"),
    ("energy_deposit_mev", "f8"),
    ("energy_reco_mev", "f8"),
    ("x", "f8"),
    ("y", "f8"),
    ("z", "f8"),
    ("time", "f8"),
    ("hit_count", "i4"),
    ("optical_photons_mean", "f8"),
    ("collected_photons_mean", "f8"),
    ("photoelectrons_mean", "f8"),
    ("photoelectrons", "f8"),
    ("charge", "f8"),
    ("passed_threshold", "?"),
]

COMPTON_DTYPE = [
    ("event_id", "i8"),
    ("gamma_track_id", "i8"),
    ("scatter_crystal_id", "U160"),
    ("absorber_crystal_id", "U160"),
    ("scatter_sector", "i4"),
    ("absorber_sector", "i4"),
    ("scatter_track_id", "i8"),
    ("absorber_track_id", "i8"),
    ("energy_scatter_mev", "f8"),
    ("energy_absorber_mev", "f8"),
    ("energy_total_mev", "f8"),
    ("scatter_x", "f8"),
    ("scatter_y", "f8"),
    ("scatter_z", "f8"),
    ("absorber_x", "f8"),
    ("absorber_y", "f8"),
    ("absorber_z", "f8"),
    ("delta_time", "f8"),
    ("distance_mm", "f8"),
    ("compton_cos_theta", "f8"),
    ("compton_angle_deg", "f8"),
    ("valid_compton_angle", "?"),
]


@dataclass(frozen=True)
class DetectorResponseConfig:
    light_yield_per_mev: float = 50_000.0
    collection_efficiency: float = 0.40
    pde: float = 0.35
    microcell_count: int = 14_400
    gain: float = 1.0
    electronics_noise_pe: float = 10.0
    energy_threshold_mev: float = 0.02
    random_seed: int = 123456


def _as_text(values: np.ndarray) -> np.ndarray:
    return np.array(
        [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v) for v in values]
    )


def _optional_int_array(arrays: dict[str, np.ndarray], name: str, size: int, default: int = -1) -> np.ndarray:
    if name in arrays:
        return arrays[name].astype(np.int64)
    return np.full(size, default, dtype=np.int64)


def _optional_text_array(arrays: dict[str, np.ndarray], name: str, size: int, default: str = "") -> np.ndarray:
    if name in arrays:
        return _as_text(arrays[name])
    return np.full(size, default, dtype=f"U{max(1, len(default))}")


def _energy_deposit_array(arrays: dict[str, np.ndarray], energy_mode: str) -> np.ndarray:
    if energy_mode == "edep":
        return arrays["TotalEnergyDeposit"].astype(float)
    if energy_mode != "ideal-gamma":
        raise ValueError(f"Unknown energy_mode: {energy_mode}")
    missing = {"PreKineticEnergy", "PostKineticEnergy", "ParticleName", "ProcessDefinedStep"} - set(arrays)
    if missing:
        raise ValueError(
            "ideal-gamma mode requires branches: "
            + ", ".join(sorted(missing))
            + ". Rerun the Gate simulation after updating the Hits actor attributes."
        )
    pre = arrays["PreKineticEnergy"].astype(float)
    post = arrays["PostKineticEnergy"].astype(float)
    particle = _as_text(arrays["ParticleName"])
    process = _as_text(arrays["ProcessDefinedStep"])
    edep = pre - post
    reject_processes = {"Transportation", "Rayl"}
    keep = (
        (particle == "gamma")
        & np.isfinite(edep)
        & (edep > 0.0)
        & np.array([p not in reject_processes for p in process], dtype=bool)
    )
    out = np.zeros_like(edep, dtype=float)
    out[keep] = edep[keep]
    return out


def _parse_crystal_id(unique_id: str) -> tuple[str, str, int, int]:
    module = unique_id.split("-", 1)[0]
    layer = "scatter" if module.startswith("scatter") else "absorber"
    sector_match = re.search(r"_(\d{2})_pixel_grid$", module)
    sector = int(sector_match.group(1)) if sector_match else -1
    pixel_match = re.search(r"_(\d+)$", unique_id)
    pixel_index = int(pixel_match.group(1)) if pixel_match else -1
    return module, layer, sector, pixel_index


def _saturate_photoelectrons(photoelectrons: np.ndarray, microcell_count: int) -> np.ndarray:
    if microcell_count <= 0:
        return photoelectrons.astype(float)
    return microcell_count * (1.0 - np.exp(-photoelectrons / microcell_count))


def _validate_config(config: DetectorResponseConfig) -> None:
    if config.light_yield_per_mev <= 0:
        raise ValueError("light_yield_per_mev must be positive.")
    if not 0 <= config.collection_efficiency <= 1:
        raise ValueError("collection_efficiency must be in [0, 1].")
    if not 0 <= config.pde <= 1:
        raise ValueError("pde must be in [0, 1].")
    if config.microcell_count < 0:
        raise ValueError("microcell_count must be >= 0.")
    if config.gain <= 0:
        raise ValueError("gain must be positive.")
    if config.electronics_noise_pe < 0:
        raise ValueError("electronics_noise_pe must be >= 0.")
    if config.energy_threshold_mev < 0:
        raise ValueError("energy_threshold_mev must be >= 0.")


def build_crystal_singles(
    arrays: dict[str, np.ndarray],
    config: DetectorResponseConfig,
    *,
    grouping_mode: str = "event",
    energy_mode: str = "edep",
) -> np.ndarray:
    _validate_config(config)
    rng = np.random.default_rng(config.random_seed)

    event_id = arrays["EventID"].astype(np.int64)
    n = len(event_id)
    track_id = _optional_int_array(arrays, "TrackID", n)
    parent_id = _optional_int_array(arrays, "ParentID", n)
    particle = _optional_text_array(arrays, "ParticleName", n)
    crystal_ids = _as_text(arrays["PreStepUniqueVolumeID"])
    edep = _energy_deposit_array(arrays, energy_mode)
    x = arrays["PostPosition_X"].astype(float)
    y = arrays["PostPosition_Y"].astype(float)
    z = arrays["PostPosition_Z"].astype(float)
    time = arrays["GlobalTime"].astype(float)

    if grouping_mode not in {"event", "gamma-track"}:
        raise ValueError(f"Unknown grouping_mode: {grouping_mode}")

    gamma_track_id = np.where(particle == "gamma", track_id, parent_id)
    gamma_track_id = np.where(gamma_track_id > 0, gamma_track_id, -1)

    groups: dict[tuple[int, int, str], list[int]] = {}
    for i, key in enumerate(zip(event_id, gamma_track_id, crystal_ids, strict=True)):
        if edep[i] <= 0:
            continue
        if grouping_mode == "event":
            key = (int(key[0]), -1, str(key[2]))
        elif int(key[1]) < 0:
            continue
        groups.setdefault(key, []).append(i)

    rows = []
    for (evt, gamma_id, crystal_id), indices in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        idx = np.array(indices)
        total_edep = float(edep[idx].sum())
        if total_edep <= 0:
            continue
        track_values = track_id[idx]
        track_value = int(track_values[0]) if len(set(track_values.tolist())) == 1 else -1
        parent_values = parent_id[idx]
        parent_value = int(parent_values[0]) if len(set(parent_values.tolist())) == 1 else -1
        module, layer, sector, pixel_index = _parse_crystal_id(crystal_id)
        weights = edep[idx] / total_edep
        optical_mean = total_edep * config.light_yield_per_mev
        collected_mean = optical_mean * config.collection_efficiency
        pe_mean = collected_mean * config.pde
        pe_sample = float(rng.poisson(pe_mean)) if pe_mean > 0 else 0.0
        fired_cells = float(_saturate_photoelectrons(np.array([pe_sample]), config.microcell_count)[0])
        if config.electronics_noise_pe > 0:
            fired_cells += float(rng.normal(0.0, config.electronics_noise_pe))
        fired_cells = max(fired_cells, 0.0)
        denom = config.light_yield_per_mev * config.collection_efficiency * config.pde
        energy_reco = fired_cells / denom if denom > 0 else 0.0
        rows.append(
            (
                int(evt),
                track_value,
                parent_value,
                int(gamma_id),
                crystal_id,
                layer,
                sector,
                pixel_index,
                total_edep,
                energy_reco,
                float(np.sum(x[idx] * weights)),
                float(np.sum(y[idx] * weights)),
                float(np.sum(z[idx] * weights)),
                float(time[idx].min()),
                int(len(idx)),
                optical_mean,
                collected_mean,
                pe_mean,
                fired_cells,
                fired_cells * config.gain,
                bool(energy_reco >= config.energy_threshold_mev),
            )
        )

    return np.array(rows, dtype=SINGLES_DTYPE)


def _compton_angle(e_scatter: float, e_absorber: float) -> tuple[float, float, bool]:
    if e_scatter <= 0 or e_absorber <= 0:
        return math.nan, math.nan, False
    e_total = e_scatter + e_absorber
    cos_theta = 1.0 - ELECTRON_REST_MEV * (1.0 / e_absorber - 1.0 / e_total)
    valid = -1.0 <= cos_theta <= 1.0
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_theta)))) if valid else math.nan
    return cos_theta, angle, valid


def build_compton_events(
    singles: np.ndarray,
    coincidence_window: float,
    *,
    pairing_mode: str = "event",
) -> np.ndarray:
    if pairing_mode not in {"event", "gamma-track"}:
        raise ValueError(f"Unknown pairing_mode: {pairing_mode}")
    selected = singles[singles["passed_threshold"]]
    event_ids = np.unique(selected["event_id"])
    rows = []

    for evt in event_ids:
        event_singles = selected[selected["event_id"] == evt]
        scatter_singles = event_singles[event_singles["layer"] == "scatter"]
        absorber_singles = event_singles[event_singles["layer"] == "absorber"]
        for scatter in scatter_singles:
            for absorber in absorber_singles:
                if pairing_mode == "gamma-track":
                    if int(scatter["gamma_track_id"]) < 0:
                        continue
                    if int(scatter["gamma_track_id"]) != int(absorber["gamma_track_id"]):
                        continue
                delta_time = abs(float(absorber["time"]) - float(scatter["time"]))
                if delta_time > coincidence_window:
                    continue
                e_scatter = float(scatter["energy_reco_mev"])
                e_absorber = float(absorber["energy_reco_mev"])
                cos_theta, angle, valid = _compton_angle(e_scatter, e_absorber)
                distance = math.dist(
                    (float(scatter["x"]), float(scatter["y"]), float(scatter["z"])),
                    (float(absorber["x"]), float(absorber["y"]), float(absorber["z"])),
                )
                rows.append(
                    (
                        int(evt),
                        int(scatter["gamma_track_id"]),
                        str(scatter["crystal_id"]),
                        str(absorber["crystal_id"]),
                        int(scatter["sector"]),
                        int(absorber["sector"]),
                        int(scatter["track_id"]),
                        int(absorber["track_id"]),
                        e_scatter,
                        e_absorber,
                        e_scatter + e_absorber,
                        float(scatter["x"]),
                        float(scatter["y"]),
                        float(scatter["z"]),
                        float(absorber["x"]),
                        float(absorber["y"]),
                        float(absorber["z"]),
                        delta_time,
                        distance,
                        cos_theta,
                        angle,
                        valid,
                    )
                )

    return np.array(rows, dtype=COMPTON_DTYPE)


def _write_csv(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data.dtype.names or [])
        for row in data:
            writer.writerow([row[name].item() if hasattr(row[name], "item") else row[name] for name in data.dtype.names or []])


def _root_safe_arrays(data: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name in data.dtype.names or []:
        arr = data[name]
        if arr.dtype.kind in {"U", "O"}:
            out[name] = arr.astype(str)
        else:
            out[name] = arr
    return out


def _save_hist(values: np.ndarray, path: Path, title: str, xlabel: str, bins: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(values) > 0:
        ax.hist(values, bins=bins, color="#2f78ff", edgecolor="black", linewidth=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_scatter(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(x) > 0:
        ax.scatter(x, y, s=18, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_layer_bar(singles: np.ndarray, path: Path) -> None:
    layers = ["scatter", "absorber"]
    counts = [int(np.count_nonzero(singles["layer"] == layer)) for layer in layers]
    energy = [
        float(singles[singles["layer"] == layer]["energy_reco_mev"].sum())
        for layer in layers
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(layers, counts, color=["#f4c430", "#e24a33"])
    axes[0].set_title("Singles count by layer")
    axes[0].set_ylabel("Count")
    axes[1].bar(layers, energy, color=["#f4c430", "#e24a33"])
    axes[1].set_title("Reconstructed energy by layer")
    axes[1].set_ylabel("Energy [MeV]")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_summary(path: Path, config: DetectorResponseConfig, singles: np.ndarray, events: np.ndarray) -> None:
    selected = singles[singles["passed_threshold"]]
    with path.open("w", encoding="utf-8") as f:
        f.write("Detector-response model: simplified post-processing, not full optical photon tracking.\n")
        f.write("Configuration:\n")
        for key, value in asdict(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\nCrystal singles:\n")
        f.write(f"  total_singles: {len(singles)}\n")
        f.write(f"  thresholded_singles: {len(selected)}\n")
        f.write(f"  scatter_singles: {int(np.count_nonzero(singles['layer'] == 'scatter'))}\n")
        f.write(f"  absorber_singles: {int(np.count_nonzero(singles['layer'] == 'absorber'))}\n")
        if len(singles) > 0:
            f.write(f"  edep_sum_mev: {float(singles['energy_deposit_mev'].sum()):.8g}\n")
            f.write(f"  ereco_sum_mev: {float(singles['energy_reco_mev'].sum()):.8g}\n")
            f.write(f"  max_ereco_mev: {float(singles['energy_reco_mev'].max()):.8g}\n")
            f.write(f"  max_photoelectrons: {float(singles['photoelectrons'].max()):.8g}\n")
        f.write("\nCompton candidates:\n")
        f.write(f"  pairs: {len(events)}\n")
        if len(events) > 0:
            valid = events[events["valid_compton_angle"]]
            f.write(f"  valid_angle_pairs: {len(valid)}\n")
            f.write(f"  mean_total_energy_mev: {float(events['energy_total_mev'].mean()):.8g}\n")
            if len(valid) > 0:
                f.write(f"  mean_compton_angle_deg: {float(valid['compton_angle_deg'].mean()):.8g}\n")


def analyze_detector_response(
    root_file: Path,
    output_dir: Path,
    config: DetectorResponseConfig,
    coincidence_window: float,
    *,
    grouping_mode: str,
    pairing_mode: str,
    energy_mode: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = uproot.open(root_file)["Hits"]
    requested = [
        "EventID",
        "TrackID",
        "ParentID",
        "ParticleName",
        "PreStepUniqueVolumeID",
        "PostPosition_X",
        "PostPosition_Y",
        "PostPosition_Z",
        "TotalEnergyDeposit",
        "GlobalTime",
    ]
    if energy_mode == "ideal-gamma":
        requested.extend(["PreKineticEnergy", "PostKineticEnergy", "ProcessDefinedStep"])
    available = set(tree.keys())
    missing_required = [name for name in requested if name not in available and name in {"EventID", "PreStepUniqueVolumeID", "PostPosition_X", "PostPosition_Y", "PostPosition_Z", "TotalEnergyDeposit", "GlobalTime"}]
    if missing_required:
        raise ValueError(f"Missing required ROOT branches: {missing_required}")
    arrays = tree.arrays([name for name in requested if name in available], library="np")

    singles = build_crystal_singles(arrays, config, grouping_mode=grouping_mode, energy_mode=energy_mode)
    events = build_compton_events(singles, coincidence_window, pairing_mode=pairing_mode)

    outputs: list[Path] = []
    singles_csv = output_dir / "crystal_singles.csv"
    events_csv = output_dir / "compton_events.csv"
    _write_csv(singles_csv, singles)
    _write_csv(events_csv, events)
    outputs.extend([singles_csv, events_csv])

    root_out = output_dir / "detector_response.root"
    with uproot.recreate(root_out) as f:
        f["CrystalSingles"] = _root_safe_arrays(singles)
        f["ComptonEvents"] = _root_safe_arrays(events)
    outputs.append(root_out)

    plot_paths = [
        output_dir / "reconstructed_energy_spectrum.png",
        output_dir / "photoelectron_spectrum.png",
        output_dir / "scatter_absorber_energy.png",
        output_dir / "compton_angle_spectrum.png",
        output_dir / "compton_total_energy_spectrum.png",
        output_dir / "layer_summary.png",
    ]
    _save_hist(singles["energy_reco_mev"], plot_paths[0], "Crystal reconstructed energy", "Energy [MeV]")
    _save_hist(singles["photoelectrons"], plot_paths[1], "MPPC photoelectron / fired-cell signal", "Signal [pe or fired cells]")
    _save_scatter(
        events["energy_scatter_mev"] if len(events) else np.array([]),
        events["energy_absorber_mev"] if len(events) else np.array([]),
        plot_paths[2],
        "Scatter vs absorber reconstructed energy",
        "Scatter energy [MeV]",
        "Absorber energy [MeV]",
    )
    valid_events = events[events["valid_compton_angle"]] if len(events) else events
    _save_hist(
        valid_events["compton_angle_deg"] if len(valid_events) else np.array([]),
        plot_paths[3],
        "Compton angle candidates",
        "Compton angle [deg]",
    )
    _save_hist(
        events["energy_total_mev"] if len(events) else np.array([]),
        plot_paths[4],
        "Compton candidate total reconstructed energy",
        "Scatter + absorber energy [MeV]",
    )
    _save_layer_bar(singles, plot_paths[5])
    outputs.extend(plot_paths)

    summary = output_dir / "detector_response_summary.txt"
    _write_summary(summary, config, singles, events)
    with summary.open("a", encoding="utf-8") as f:
        f.write("\nEvent building:\n")
        f.write(f"  energy_mode: {energy_mode}\n")
        f.write(f"  grouping_mode: {grouping_mode}\n")
        f.write(f"  pairing_mode: {pairing_mode}\n")
        f.write(f"  coincidence_window: {coincidence_window}\n")
    outputs.append(summary)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process GATE hits into MPPC-like detector response outputs.")
    parser.add_argument("--root-file", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--light-yield-per-mev", type=float, default=DetectorResponseConfig.light_yield_per_mev)
    parser.add_argument("--collection-efficiency", type=float, default=DetectorResponseConfig.collection_efficiency)
    parser.add_argument("--pde", type=float, default=DetectorResponseConfig.pde)
    parser.add_argument("--microcell-count", type=int, default=DetectorResponseConfig.microcell_count)
    parser.add_argument("--gain", type=float, default=DetectorResponseConfig.gain)
    parser.add_argument("--electronics-noise-pe", type=float, default=DetectorResponseConfig.electronics_noise_pe)
    parser.add_argument("--energy-threshold-mev", type=float, default=DetectorResponseConfig.energy_threshold_mev)
    parser.add_argument("--coincidence-window", type=float, default=float("inf"), help="Maximum time difference in the same units as GlobalTime.")
    parser.add_argument("--energy-mode", choices=["edep", "ideal-gamma"], default="edep")
    parser.add_argument("--grouping-mode", choices=["event", "gamma-track"], default="event")
    parser.add_argument("--pairing-mode", choices=["event", "gamma-track"], default="event")
    parser.add_argument("--random-seed", type=int, default=DetectorResponseConfig.random_seed)
    parser.add_argument("--open", action="store_true", help="Open generated PNG files in macOS Preview.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DetectorResponseConfig(
        light_yield_per_mev=args.light_yield_per_mev,
        collection_efficiency=args.collection_efficiency,
        pde=args.pde,
        microcell_count=args.microcell_count,
        gain=args.gain,
        electronics_noise_pe=args.electronics_noise_pe,
        energy_threshold_mev=args.energy_threshold_mev,
        random_seed=args.random_seed,
    )
    output_paths = analyze_detector_response(
        args.root_file,
        args.output_dir,
        cfg,
        args.coincidence_window,
        grouping_mode=args.grouping_mode,
        pairing_mode=args.pairing_mode,
        energy_mode=args.energy_mode,
    )
    for output_path in output_paths:
        print(output_path)
    if args.open:
        import subprocess

        images = [str(p) for p in output_paths if p.suffix.lower() == ".png"]
        subprocess.run(["open", *images], check=False)

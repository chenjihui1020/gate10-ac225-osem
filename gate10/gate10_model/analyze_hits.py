from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "gate10_output" / "small_run" / "ac225_hits.root"
DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "small_run_analysis"


def _as_text(values: np.ndarray) -> np.ndarray:
    return np.array(
        [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v) for v in values]
    )


def _save_hist(
    values: np.ndarray,
    *,
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int,
    log_y: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#2f78ff", edgecolor="black", linewidth=0.25)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_position_hist2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    weight: np.ndarray,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    hist = ax.hist2d(x, y, bins=bins, weights=weight, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.set_label("Energy deposit [MeV]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_particle_counts(particles: np.ndarray, output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    names, counts = np.unique(particles, return_counts=True)
    order = np.argsort(counts)[::-1]
    names = names[order]
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, counts, color="#d95f02")
    ax.set_yticks(y, labels=names)
    ax.invert_yaxis()
    ax.set_xlabel("Hit rows")
    ax.set_title("Particle names in Hits tree")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return names, counts


def analyze_hits(root_file: Path, output_dir: Path, bins: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = uproot.open(root_file)["Hits"]
    arrays = tree.arrays(
        [
            "EventID",
            "ParticleName",
            "PostPosition_X",
            "PostPosition_Y",
            "PostPosition_Z",
            "TotalEnergyDeposit",
            "GlobalTime",
        ],
        library="np",
    )

    event_id = arrays["EventID"].astype(int)
    particles = _as_text(arrays["ParticleName"])
    x = arrays["PostPosition_X"].astype(float)
    y = arrays["PostPosition_Y"].astype(float)
    z = arrays["PostPosition_Z"].astype(float)
    edep = arrays["TotalEnergyDeposit"].astype(float)
    time = arrays["GlobalTime"].astype(float)

    positive = edep > 0
    edep_positive = edep[positive]
    if edep_positive.size == 0:
        raise RuntimeError("No positive TotalEnergyDeposit entries were found in the Hits tree.")

    max_event_id = int(event_id.max()) if event_id.size else 0
    event_edep = np.bincount(event_id, weights=edep, minlength=max_event_id + 1)
    event_edep_positive = event_edep[event_edep > 0]

    outputs: list[Path] = []

    out = output_dir / "hit_energy_spectrum.png"
    _save_hist(
        edep_positive,
        output_path=out,
        title="Hit energy deposit spectrum",
        xlabel="TotalEnergyDeposit per hit [MeV]",
        bins=bins,
    )
    outputs.append(out)

    out = output_dir / "event_energy_spectrum.png"
    _save_hist(
        event_edep_positive,
        output_path=out,
        title="Event summed energy deposit spectrum",
        xlabel="Summed TotalEnergyDeposit per event [MeV]",
        bins=bins,
    )
    outputs.append(out)

    out = output_dir / "hit_xy_energy_map.png"
    _save_position_hist2d(
        x[positive],
        y[positive],
        output_path=out,
        title="XY hit energy map",
        xlabel="X [mm]",
        ylabel="Y [mm]",
        weight=edep_positive,
        bins=80,
    )
    outputs.append(out)

    out = output_dir / "hit_xz_energy_map.png"
    _save_position_hist2d(
        x[positive],
        z[positive],
        output_path=out,
        title="XZ hit energy map",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        weight=edep_positive,
        bins=80,
    )
    outputs.append(out)

    out = output_dir / "hit_yz_energy_map.png"
    _save_position_hist2d(
        y[positive],
        z[positive],
        output_path=out,
        title="YZ hit energy map",
        xlabel="Y [mm]",
        ylabel="Z [mm]",
        weight=edep_positive,
        bins=80,
    )
    outputs.append(out)

    out = output_dir / "particle_hit_counts.png"
    particle_names, particle_counts = _save_particle_counts(particles, out)
    outputs.append(out)

    out = output_dir / "hit_time_spectrum.png"
    _save_hist(
        time[positive],
        output_path=out,
        title="Hit time spectrum",
        xlabel="GlobalTime",
        bins=bins,
    )
    outputs.append(out)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"root_file: {root_file}\n")
        f.write(f"tree: Hits\n")
        f.write(f"entries: {tree.num_entries}\n")
        f.write(f"positive_edep_entries: {edep_positive.size}\n")
        f.write(f"events_with_hits: {event_edep_positive.size}\n")
        f.write(f"total_energy_deposit_mev: {edep.sum():.8g}\n")
        f.write(f"hit_edep_mean_mev: {edep_positive.mean():.8g}\n")
        f.write(f"hit_edep_max_mev: {edep_positive.max():.8g}\n")
        f.write(f"event_edep_mean_mev: {event_edep_positive.mean():.8g}\n")
        f.write(f"event_edep_max_mev: {event_edep_positive.max():.8g}\n")
        f.write("particle_hit_counts:\n")
        for name, count in zip(particle_names, particle_counts):
            f.write(f"  {name}: {int(count)}\n")
    outputs.append(summary_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create spectra and hit maps from a GATE 10 ROOT hits file.")
    parser.add_argument("--root-file", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--open", action="store_true", help="Open generated images in macOS Preview.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = analyze_hits(args.root_file, args.output_dir, args.bins)
    for output in outputs:
        print(output)
    if args.open:
        import subprocess

        images = [str(p) for p in outputs if p.suffix.lower() == ".png"]
        subprocess.run(["open", *images], check=False)

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "geometry_plots"


def module_specs():
    for sector in range(8):
        phi = 2.0 * math.pi * sector / 8.0
        use_thin = sector % 2 == 0
        scatter_thickness = 1.5 if use_thin else 4.0
        scatter_center_radius = 37.5 + 4.0 / 2.0
        scatter_inner_radius = scatter_center_radius - scatter_thickness / 2.0
        yield {
            "name": f"scatter_{'thin' if use_thin else 'thick'}_{sector:02d}",
            "layer": "scatter",
            "sector": sector,
            "phi": phi,
            "pixel_thickness": scatter_thickness,
            "mppc_thickness": 2.0,
            "inner_radius": scatter_inner_radius,
            "face": 26.3,
            "color": "#f4c430" if use_thin else "#d99a00",
        }
        yield {
            "name": f"absorber_{sector:02d}",
            "layer": "absorber",
            "sector": sector,
            "phi": phi,
            "pixel_thickness": 9.0,
            "mppc_thickness": 2.0,
            "inner_radius": 62.5,
            "face": 26.3,
            "color": "#e24a33",
        }


def rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_corners(spec: dict) -> np.ndarray:
    pixel_t = spec["pixel_thickness"]
    mppc_t = spec["mppc_thickness"]
    base_t = pixel_t + mppc_t
    half = spec["face"] / 2.0
    center_r = spec["inner_radius"] + pixel_t / 2.0 + mppc_t / 2.0
    center = np.array(
        [center_r * math.cos(spec["phi"]), center_r * math.sin(spec["phi"]), 0.0]
    )
    # Local +x points outward. This rotation matches the GATE 10 model.
    rot = rotation_z(spec["phi"])
    local = np.array(
        [
            [sx * base_t / 2.0, sy * half, sz * half]
            for sx in [-1, 1]
            for sy in [-1, 1]
            for sz in [-1, 1]
        ]
    )
    return local @ rot.T + center


def box_faces(corners: np.ndarray) -> list[list[np.ndarray]]:
    idx = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    return [[corners[i] for i in face] for face in idx]


def plot_xy(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    for spec in module_specs():
        corners = box_corners(spec)
        xy = corners[[0, 2, 6, 4], :2]
        patch = plt.Polygon(
            xy,
            closed=True,
            facecolor=spec["color"],
            edgecolor="black",
            alpha=0.75 if spec["layer"] == "scatter" else 0.55,
            linewidth=0.8,
        )
        ax.add_patch(patch)
        r = spec["inner_radius"] + spec["pixel_thickness"] + spec["mppc_thickness"] + 6
        ax.text(
            r * math.cos(spec["phi"]),
            r * math.sin(spec["phi"]),
            str(spec["sector"]),
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.add_patch(plt.Circle((0, 0), 2.0, color="#2f78ff", alpha=0.8, label="2 mm water phantom"))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Ac-225 Compton PET geometry, XY view")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#f4c430", lw=8, label="scatter GAGG"),
        plt.Line2D([0], [0], color="#e24a33", lw=8, label="absorber GAGG"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f78ff", markersize=8, label="water phantom"),
    ], loc="upper right")
    out = output_dir / "geometry_xy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_3d(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for spec in module_specs():
        corners = box_corners(spec)
        poly = Poly3DCollection(
            box_faces(corners),
            facecolors=spec["color"],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.7 if spec["layer"] == "scatter" else 0.5,
        )
        ax.add_collection3d(poly)

    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x = 2.0 * np.outer(np.cos(u), np.sin(v))
    y = 2.0 * np.outer(np.sin(u), np.sin(v))
    z = 2.0 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#2f78ff", alpha=0.8, linewidth=0)

    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_zlim(-25, 25)
    ax.set_box_aspect((180, 180, 50))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Ac-225 Compton PET geometry, 3D overview")
    ax.view_init(elev=24, azim=35)
    out = output_dir / "geometry_3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_pixel_faces(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    cases = [
        ("scatter thin", 1.5, "#f4c430"),
        ("scatter thick", 4.0, "#d99a00"),
        ("absorber", 9.0, "#e24a33"),
    ]
    pixel_size = 2.5
    pitch = 3.2
    edge_gap = 0.7
    pixels_per_side = 8
    face = pixels_per_side * pitch + edge_gap
    start = -((pixels_per_side - 1) * pitch) / 2.0

    for ax, (title, thickness, color) in zip(axes, cases):
        ax.add_patch(
            plt.Rectangle(
                (-face / 2.0, -face / 2.0),
                face,
                face,
                facecolor="#dddddd",
                edgecolor="black",
                alpha=0.35,
                linewidth=1.2,
            )
        )
        for iy in range(pixels_per_side):
            for iz in range(pixels_per_side):
                y = start + iy * pitch
                z = start + iz * pitch
                ax.add_patch(
                    plt.Rectangle(
                        (y - pixel_size / 2.0, z - pixel_size / 2.0),
                        pixel_size,
                        pixel_size,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.45,
                    )
                )
        ax.set_title(f"{title}\n{pixels_per_side}x{pixels_per_side}, thickness {thickness:g} mm")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("local Y [mm]")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("local Z [mm]")
    axes[0].set_xlim(-15, 15)
    axes[0].set_ylim(-15, 15)
    out = output_dir / "geometry_pixel_faces_8x8.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable non-Qt geometry preview.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true", help="Open the generated PNG files in macOS Preview.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_xy(args.output_dir),
        plot_3d(args.output_dir),
        plot_pixel_faces(args.output_dir),
    ]
    for out in outputs:
        print(out)
    if args.open:
        import subprocess

        subprocess.run(["open", *map(str, outputs)], check=False)

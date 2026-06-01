from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import opengate as gate
from opengate.geometry.volumes import RepeatParametrisedVolume
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIAL_DB = PROJECT_ROOT / "config" / "GateMaterials.db"


@dataclass(frozen=True)
class GeometryHandles:
    scintillator_volumes: list[str]
    scatter_scintillator_volumes: list[str]
    absorber_scintillator_volumes: list[str]


def add_ac225_compton_pet_geometry(sim: gate.Simulation) -> GeometryHandles:
    """Build the 8-sector GAGG Compton PET geometry with GATE 10 Python objects."""

    mm = gate.g4_units.mm
    m = gate.g4_units.m

    sim.volume_manager.add_material_database(MATERIAL_DB)

    sim.world.size = [3.0 * m, 3.0 * m, 3.0 * m]
    sim.world.material = "Air"
    sim.world.color = [0.0, 0.0, 0.0, 0.0]

    source_phantom = sim.add_volume("Sphere", "source_phantom")
    source_phantom.mother = "world"
    source_phantom.rmin = 0 * mm
    source_phantom.rmax = 2.0 * mm
    source_phantom.material = "Water"
    source_phantom.translation = [0, 0, 0]
    source_phantom.color = [0.0, 0.35, 1.0, 0.45]

    pixel_names: list[str] = []
    scatter_pixel_names: list[str] = []
    absorber_pixel_names: list[str] = []

    for sector in range(8):
        phi = 2.0 * math.pi * sector / 8.0
        phi_deg = math.degrees(phi)
        use_thin = sector % 2 == 0
        scatter_thickness = 1.5 if use_thin else 4.0
        scatter_center_radius = 37.5 + 4.0 / 2.0
        scatter_inner_radius = scatter_center_radius - scatter_thickness / 2.0
        scatter_tag = "thin" if use_thin else "thick"
        name = f"scatter_{scatter_tag}_{sector:02d}"
        pixel = _add_detector_module(
            sim=sim,
            name=name,
            layer="scatter",
            sector=sector,
            pixel_thickness_mm=scatter_thickness,
            inner_surface_radius_mm=scatter_inner_radius,
            phi_rad=phi,
            phi_deg=phi_deg,
        )
        pixel_names.append(pixel)
        scatter_pixel_names.append(pixel)

        name = f"absorber_{sector:02d}"
        pixel = _add_detector_module(
            sim=sim,
            name=name,
            layer="absorber",
            sector=sector,
            pixel_thickness_mm=9.0,
            inner_surface_radius_mm=62.5,
            phi_rad=phi,
            phi_deg=phi_deg,
        )
        pixel_names.append(pixel)
        absorber_pixel_names.append(pixel)

    return GeometryHandles(
        scintillator_volumes=pixel_names,
        scatter_scintillator_volumes=scatter_pixel_names,
        absorber_scintillator_volumes=absorber_pixel_names,
    )


def _add_detector_module(
    *,
    sim: gate.Simulation,
    name: str,
    layer: str,
    sector: int,
    pixel_thickness_mm: float,
    inner_surface_radius_mm: float,
    phi_rad: float,
    phi_deg: float,
) -> str:
    mm = gate.g4_units.mm

    mppc_thickness = 2.0 * mm
    pixel_size = 2.5 * mm
    pitch = 3.2 * mm
    edge_gap = 0.7 * mm
    num_pixels = 8
    half_size = (num_pixels * pitch + edge_gap) / 2.0
    pixel_thickness = pixel_thickness_mm * mm
    base_thickness = pixel_thickness + mppc_thickness
    module_radius = inner_surface_radius_mm * mm + pixel_thickness / 2.0 + mppc_thickness / 2.0

    module = sim.add_volume("Box", name)
    module.mother = "world"
    module.size = [base_thickness, 2.0 * half_size, 2.0 * half_size]
    module.translation = [
        module_radius * math.cos(phi_rad),
        module_radius * math.sin(phi_rad),
        0.0,
    ]
    module.rotation = Rotation.from_euler("z", phi_deg, degrees=True).as_matrix()
    module.material = "Air"
    module.color = [0.8, 0.8, 0.8, 0.08]

    frame = sim.add_volume("Box", f"{name}_frame")
    frame.mother = module.name
    frame.size = [pixel_thickness, 2.0 * half_size, 2.0 * half_size]
    frame.translation = [-base_thickness / 2.0 + pixel_thickness / 2.0, 0, 0]
    frame.material = "BaSO4"
    frame.color = [0.95, 0.95, 0.9, 0.25]

    pixel = sim.add_volume("Box", f"{name}_pixel")
    pixel.mother = frame.name
    pixel.size = [pixel_thickness, pixel_size, pixel_size]
    pixel.material = "CeGAGG"
    pixel.color = [1.0, 0.85, 0.05, 1.0] if layer == "scatter" else [1.0, 0.25, 0.05, 1.0]

    pixel_grid = RepeatParametrisedVolume(repeated_volume=pixel, name=f"{name}_pixel_grid")
    pixel_grid.translation = [0.0, pitch, pitch]
    pixel_grid.linear_repeat = [1, num_pixels, num_pixels]
    sim.volume_manager.add_volume(pixel_grid)

    mppc = sim.add_volume("Box", f"{name}_mppc")
    mppc.mother = module.name
    mppc.size = [mppc_thickness, 2.0 * half_size, 2.0 * half_size]
    mppc.translation = [base_thickness / 2.0 - mppc_thickness / 2.0, 0, 0]
    mppc.material = "Silicon"
    mppc.color = [0.95, 0.95, 1.0, 0.35]

    module.user_info["sector"] = sector
    module.user_info["layer"] = layer
    module.user_info["inner_surface_radius_mm"] = inner_surface_radius_mm
    module.user_info["pixel_thickness_mm"] = pixel_thickness_mm

    return pixel.name

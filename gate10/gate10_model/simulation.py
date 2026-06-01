from __future__ import annotations

from pathlib import Path

import opengate as gate

from gate10_model.geometry import add_ac225_compton_pet_geometry
from gate10_model.source import add_ac225_point_source


def build_simulation(
    *,
    output_dir: Path,
    n_events: int | None = 1000,
    activity_bq: float | None = None,
    duration_s: float = 1.0,
    threads: int = 1,
    seed: int | str = 123456,
    visu: str = "none",
    hits: bool = True,
    overlap_check: bool = True,
) -> gate.Simulation:
    """Create but do not run the Ac-225 Compton PET simulation."""

    sim = gate.Simulation()
    sim.output_dir = output_dir
    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.number_of_threads = int(threads)
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.progress_bar = True
    sim.check_volumes_overlap = overlap_check

    if visu != "none":
        sim.visu = True
        sim.visu_type = visu
        sim.visu_verbose = True
        if "gdml" in visu:
            sim.visu_filename = str(output_dir / "ac225_compton_pet_geometry.gdml")
        elif "vrml" in visu:
            sim.visu_filename = str(output_dir / "ac225_compton_pet_geometry.wrl")
    else:
        sim.visu = False

    handles = add_ac225_compton_pet_geometry(sim)
    _configure_physics(sim)

    if n_events is None:
        sec = gate.g4_units.s
        sim.run_timing_intervals = [[0, float(duration_s) * sec]]
    add_ac225_point_source(sim, activity_bq=activity_bq, n_events=n_events)

    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output_filename = "stats.txt"

    if hits:
        hit_actor = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
        hit_actor.attached_to = handles.scintillator_volumes
        hit_actor.authorize_repeated_volumes = True
        hit_actor.output_filename = output_dir / "ac225_hits.root"
        hit_actor.attributes = [
            "EventID",
            "RunID",
            "ThreadID",
            "TrackID",
            "ParentID",
            "PDGCode",
            "ParticleName",
            "TrackCreatorProcess",
            "TrackVolumeName",
            "PreStepUniqueVolumeID",
            "PostPosition",
            "TotalEnergyDeposit",
            "PreKineticEnergy",
            "PostKineticEnergy",
            "KineticEnergy",
            "ProcessDefinedStep",
            "GlobalTime",
        ]

    return sim


def _configure_physics(sim: gate.Simulation) -> None:
    mm = gate.g4_units.mm

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
   
    sim.physics_manager.enable_decay = True

    sim.physics_manager.em_parameters.fluo = True
    sim.physics_manager.em_parameters.auger = True
    sim.physics_manager.em_parameters.auger_cascade = True
    sim.physics_manager.em_parameters.pixe = True
    sim.physics_manager.em_parameters.deexcitation_ignore_cut = True
##判断截断距离，若二次粒子能量预期运动距离小于该值，则将能量直接沉积在当前位置
    sim.physics_manager.global_production_cuts.gamma = 0.005 * mm
    sim.physics_manager.global_production_cuts.electron = 0.01 * mm
    sim.physics_manager.global_production_cuts.positron = 0.01 * mm
    sim.physics_manager.global_production_cuts.proton = 0.1 * mm

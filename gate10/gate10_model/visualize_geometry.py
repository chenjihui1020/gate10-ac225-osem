from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gate10_model.simulation import build_simulation


DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open or export the GATE 10 detector geometry.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--type",
        choices=["qt", "gdml", "vrml_file_only"],
        default="qt",
        help="qt opens Geant4 Qt; gdml/vrml_file_only export geometry files.",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run visualization in a subprocess. On macOS Qt this can crash in GATE 10, so it is off by default.",
    )
    parser.add_argument(
        "--check-overlaps",
        action="store_true",
        help="Enable overlap checks. Parameterised pixel grids may print non-fatal warnings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = build_simulation(
        output_dir=args.output_dir,
        n_events=1,
        threads=1,
        visu=args.type,
        hits=False,
        overlap_check=args.check_overlaps,
    )
    sim.run(start_new_process=args.subprocess)
    print(f"Visualization mode finished: {args.type}")
    if args.type in {"gdml", "vrml_file_only"}:
        print(f"Geometry file: {sim.visu_filename}")

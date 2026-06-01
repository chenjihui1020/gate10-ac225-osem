from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gate10_model.simulation import build_simulation


DEFAULT_OUTPUT = PROJECT_ROOT / "gate10_output" / "smoke"


def _parse_seed(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer or 'auto'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GATE 10 Ac-225 Compton PET model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events", type=int, default=1000, help="Fixed number of Ac-225 ion events. Use 0 for activity/time mode.")
    parser.add_argument("--activity-bq", type=float, default=1_000_000.0)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=_parse_seed, default=123456)
    parser.add_argument(
        "--visu",
        choices=["none", "qt", "gdml", "vrml_file_only"],
        default="none",
    )
    parser.add_argument("--no-hits", action="store_true", help="Disable ROOT hit output.")
    parser.add_argument("--no-overlap-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_events = None if args.events == 0 else args.events
    sim = build_simulation(
        output_dir=args.output_dir,
        n_events=n_events,
        activity_bq=args.activity_bq,
        duration_s=args.duration_s,
        threads=args.threads,
        seed=args.seed,
        visu=args.visu,
        hits=not args.no_hits,
        overlap_check=not args.no_overlap_check,
    )

    sim.run(start_new_process=False)
    print(sim.get_actor("Stats"))

# Gate10 Ac-225 Compton PET and CUDA/OSEM Adapter

This repository contains the Gate10/OpenGATE Python model, detector-response post-processing, and CUDA/OSEM reconstruction adapter used for an Ac-225 point-source Compton PET study.

## Contents

- `gate10/gate10_model/`: Gate10/OpenGATE geometry, Ac-225 source, simulation runner, hit analysis, detector response, geometry visualization, and CPU OSEM helper code.
- `gate10/tests/`: Python unit tests for detector response and OSEM utilities.
- `gate10/config/GateMaterials.db`: Material definitions used by the Gate10 model.
- `osem_adapter/src/ac225_osem_cuda.cu`: CUDA OSEM adapter using Gate10 direct scatter/absorber positions.
- `osem_adapter/scripts/`: Gate10-to-OSEM conversion, CPU diagnostic, CUDA plotting, and paper-style evaluation scripts.
- `osem_adapter/data/`: Small converted OSEM input CSV files from the latest formal run.
- `osem_adapter/evaluation/`: Fig.7/Fig.10-style evaluation outputs.
- `original_osem_reference/`: Original experiment-oriented OSEM reference files copied for comparison.
- `results/`: Text summaries from the latest 40 min Gate10 simulation and detector-response stage.

Large ROOT files, detector-response ROOT outputs, local Python environments, and transient Gate output folders are intentionally excluded from Git.

## Latest Formal Run

The latest formal run used:

- Gate simulated time: `40 min`
- Threads: `8`
- Initial events: `864105`
- Detector response mode: `ideal-gamma + gamma-track`
- OSEM grid: `40 x 40 x 25`
- OSEM iterations/subsets: `48 / 4`

The important summaries are in:

- `results/gate10_40min_stats.txt`
- `results/gate10_hit_analysis_summary.txt`
- `results/detector_response_summary.txt`
- `osem_adapter/data/ac225_osem_conversion_summary.json`
- `osem_adapter/evaluation/evaluation_summary.json`

## Minimal Local Workflow

Activate the Gate10 environment first:

```bash
source /Users/chen/Documents/gate10/activate_gate10.sh
```

Run a small Gate10 simulation:

```bash
cd gate10
python -m gate10_model.run_ac225 \
  --output-dir gate10_output/smoke \
  --events 2000 --threads 4 --seed 2400703 --no-overlap-check
```

Convert a detector-response `compton_events.csv` file to CUDA/OSEM input:

```bash
cd osem_adapter
python scripts/gate_to_osem_input.py \
  --input /path/to/compton_events.csv \
  --output-dir data --prefix ac225 \
  --energy-window-frac 0.15 --min-scatter-kev 1
```

Run CPU diagnostics:

```bash
python scripts/osem_cpu_diagnostic.py \
  --input data/ac225_osem_events_sourceqa20.csv \
  --iterations 48 --subsets 4 --sigma-deg 6 \
  --nx 40 --ny 40 --nz 25
```

## CUDA / Colab Workflow

Upload this repository to Colab with a T4 runtime and run:

```bash
cd osem_adapter
bash run_colab_ac225_osem.sh
```

The script compiles `src/ac225_osem_cuda.cu`, runs six reconstruction datasets, creates central slices and MIP images, and generates paper-style evaluation plots.

## Important Notes

- `sourceqa20` files use the known simulated source position for ARM filtering. They are useful for validating geometry and reconstruction closure, but they are not blind reconstruction inputs.
- MPPC response is a post-processing approximation, not optical photon transport.
- The current Ac-225 blind reconstruction is sensitive to low-statistics and multi-gamma event contamination. The source-QA datasets reconstruct near the origin, confirming that the geometry and coordinate adapter are consistent.
- See `osem_adapter/README.md` for detailed Chinese documentation of the simulation, conversion, CUDA/OSEM behavior, and limitations.

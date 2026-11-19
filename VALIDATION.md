# Validation — 2026-11-19

- Python 3.11.18, macOS arm64. Editable package installation and environment check passed.
- Core regression selection (`scripts/smoke.sh`): **73 passed in 4.72s**, no skips. Raw output: validation/cpu-smoke.log.
- CLI migration and baseline execution checks: **43 passed in 0.31s**, no skips. Raw output: validation/baseline-cli.log.
- Baseline/fusion configurations checked against a temporary stub benchmark: kind=baseline vs v17_ablation/fusion, separate policy/output directories, matching model/runtime/robot/action/budget settings, and refusal to overwrite existing policies. This is a configuration/copy check, not a simulator run.
- Experimental dense-geometry extension and its migration registration removed. Core execution and PoS/OpenVL behavior covered by the regression checks above.
- Software versions tested on this machine are in requirements/cpu-tested-macos311.txt; this is not a CUDA lockfile.
- No real GPU perception, API generation or full RoboDojo evaluation was performed in this delivery task.
- ZIP CRC and per-file SHA256 verified after packaging.

- Historical fusion episodes are archived under logs/robodojo-fusion-20261114 with original/published hashes, official-score checks and source-difference records. They are not fresh validation of the current source.

- Expanded historical log catalog: 20 project ZIP archives with original-file SHA256 manifests and 44 indexed task records. ZIP CRC checked; score/error indexes retain batch and condition identities. No fresh task runs performed.

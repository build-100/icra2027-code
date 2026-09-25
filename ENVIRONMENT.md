# Runtime environment

Use a fresh **Python 3.12** environment created inside this repository. GEVD and
its baselines resolve their inputs from the repository; they do not require a
previous project checkout, a system-specific interpreter path, or an external
training workspace.

## Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The direct runtime dependencies are NumPy, NetworkX, PyYAML, PyTorch, and
Matplotlib. Pytest is required for the scientific regression suite.

## Recorded validation environment

The independently created release environment uses the following versions:

| Component | Version |
|---|---|
| CPython | 3.12.14 |
| PyTorch | 2.11.0+cu130 |
| NumPy | 2.5.2 |
| NetworkX | 3.6.1 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.11.1 |
| Pytest | 9.1.1 |

`requirements-lock.txt` records the exact installed dependency set. Its PyTorch
build is CUDA-capable, while release checks run on CPU. Use the lock with its
specified package sources when recreating this exact environment; use
`requirements.txt` for a portable installation with an appropriate PyTorch build.
The environment uses a standard CPython base installation and has its own
installed packages; it does not depend on an earlier project environment.

## CPU and CUDA

CPU is the default device and the target for release validation. No ROS
installation is required for graph-level simulation. Optional CUDA execution
requires a compatible PyTorch build and driver; select it through the command
line only after the environment recognizes the device.

Dependency ranges in `requirements.txt` express supported installation bounds;
an exact lock records the packages used for a particular validated environment.
A package lock does not guarantee identical optimization trajectories across
operating systems, accelerators, or PyTorch versions. Preserve the seed,
configuration, version report, and budget counters with each new experiment.

## Verification

```bash
python main.py --config configs/simulation/seven_node/gevd.yaml
python scripts/verify_release.py
python -m pytest tests --basetemp=results/generated/pytest_tmp -q
```

The first command performs a preflight without training. Add `--run` only when
starting an experiment. Output directories are local and excluded from Git.

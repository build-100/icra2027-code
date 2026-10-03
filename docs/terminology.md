# Manuscript Terminology

The repository follows the terminology of the GEVD manuscript. Prose and figures
use **Env1**, **Env2**, and **Env3**, as in Fig. 4 and Table I. Paths, configuration
identifiers, and command arguments use `env1`, `env2`, and `env3`.

| Manuscript designation | Identifier | Regions | Robot counts | Configuration directory |
|---|---|---:|---|---|
| Env1 | `env1` | 36 | 2, 3, 4 | `configs/simulation/comparison/env1/` |
| Env2 | `env2` | 23 | 2, 3, 4 | `configs/simulation/comparison/env2/` |
| Env3 | `env3` | 39 | 2, 3, 4 | `configs/simulation/comparison/env3/` |

## Method Components

| Manuscript section | Term |
|---|---|
| IV-A | Gauge-Aware Structural Modeling |
| IV-A.1 | Local Mapping Evaluation |
| IV-A.2 | Gauge-Nullity Accounting |
| IV-A.3 | Gauge-Aware Task Objective |
| IV-B | Event-Aligned Value Decomposition |
| IV-B.1 | Collective Value Decomposition |
| IV-B.2 | Retrospective Utility Augmentation |

The evaluation quantities are mapping return G, structural score S, coverage
n(p)/|V|, connected-component count c(p), and cumulative team travel D(p).
Full coverage and terminal fusion require n(p) = |V| and c(p) = 1 within T_max.

## Experiments

| Manuscript section | Title | Repository configuration |
|---|---|---|
| V-A | Performance in Simple Use Case | `configs/simulation/seven_node/` |
| V-B | Comparative Simulation in Canonical Environments | `configs/simulation/comparison/` |
| V-C | GEVD Ablation Study | `configs/simulation/ablation/env3_N3/` |
| VI | Experiments with a Real Multi-Robot System | `configs/real/paper7_external/` |

Section V-A compares GEVD with **QMIX coverage-only (QCO)** and **QMIX
coverage-first (QCF)**. Section V-B uses **CMRE**, **sGre**, **dGre**, **QMIX**,
and **COMA**. Table II labels are **w/o Retrospective Utility**, **w/o
Gauge-Nullity Term**, **w/o Value Decomposition**, and **w/o Representative-Pose**.
See [methods](methods.md) for their configuration identifiers.

The office video is a supplementary demonstration supplied with the presentation.
It is distinct from the constructed environment in Section VI. Terminology
alignment does not change experimental measurements, sample counts, or the
[evidence limitations](reproducibility.md).

# GEVD Methods and Common Evaluation

## Gauge-Aware Structural Modeling

Following Section IV-A, Local Mapping Evaluation computes the representative-pose
structural score within each connected component. Gauge-Nullity Accounting
records unresolved relative frames through the component count. The
Gauge-Aware Task Objective combines structure, coverage, fusion, and travel cost:

```text
G(p) = S(p) + rho_V [n(p) - N] + rho_g [N - c(p)] - beta D(p).
```

This is Equation (9): S is the structural score, n is the number of visited
regions, N is the number of robots, c is the number of connected components,
and D is cumulative team travel. The initial isolated representatives give
S_0 = 0. Full coverage and terminal fusion require n(p) = |V| and c(p) = 1
within T_max. Incremental rewards follow Equation (11).

## Event-Aligned Value Decomposition

Section IV-B combines **Collective Value Decomposition**, Q_tot = sum_i Q_i,
with **Retrospective Utility Augmentation**, whose auxiliary utilities Q'_i learn
from delayed fusion feedback associated with earlier visits. Decentralized
selection uses Q_i + lambda Q'_i as in Equation (14).

The implementation uses shared graph encoders, Double DQN targets, separate
primary and auxiliary parameters, an auxiliary amplitude bound of 0.1 |Q_i|,
and an execution multiplier lambda = 0.1. See [parameters](parameters.md) for
implementation settings beyond the manuscript's high-level definitions.

## Comparison Baselines

| Method | Manuscript terminology and released implementation |
|---|---|
| CMRE | Coordinated multi-robot exploration; constructs coverage routes |
| sGre | Sequential greedy loop selection |
| dGre | Ordered double-greedy loop selection; candidate cap 32 |
| QMIX | QMIX with local MLP utilities; Adam learning rate 5e-5 |
| COMA | Counterfactual multi-agent policy gradients; actor/critic RMSprop learning rate 5e-4 |
| QCO | QMIX coverage-only; coverage, travel cost, and intra-robot information with coefficient 7 |
| QCF | QMIX coverage-first; QCO with an additional 0.5 success bonus |

QCO and QCF are the QMIX variants in Section V-A. CMRE, sGre, dGre, QMIX,
and COMA are the baselines in Section V-B. Planning methods are adapted to the
common graph task. Each MARL baseline retains its own network and optimizer.
Evaluation uses the common mapping return G rather than each baseline's training
return. The physical coverage-only baseline in Section VI is labeled separately.

## GEVD Ablation Study

The labels below follow Table II; configuration identifiers select each variant.

| Table II label | Configuration identifier | Component removed |
|---|---|---|
| w/o Retrospective Utility | `no_retrospective` | Retrospective Utility Augmentation |
| w/o Gauge-Nullity Term | `no_gauge` | Explicit gauge-nullity reward term in primary TD learning |
| w/o Value Decomposition | `no_vdn` | Collective Value Decomposition; use shared independent TD with team reward |
| w/o Representative-Pose | `raw_structure` | Representative-pose evaluation; use componentwise raw structural increments |

All variants retain the common simulator, evaluation objective, and success
definition. The last three retain the auxiliary utilities. The configurations
use Env3 with N = 3 under `configs/simulation/ablation/env3_N3`. See
[reproducibility notes](reproducibility.md) for the available experiment evidence.

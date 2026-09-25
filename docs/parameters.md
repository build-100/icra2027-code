# Task parameters and calibration

All file paths in the configurations are relative to the repository. CPU is the
default device. The parameter tables below define executable simulation tasks;
they do not establish that every manuscript table entry was produced by the
current GEVD implementation.

## Comparison tasks

| Paper environment | Map | Robots | Start regions | T_max | beta | rho_v | rho_g |
|---|---|---:|---|---:|---:|---:|---:|
| Environment 1 | map3 | 2 | [8, 29] | 23 | 0.00098840812174339713 | 0.1 | 1 |
| Environment 1 | map3 | 3 | [3, 8, 29] | 17 | 0.00048883587743437924 | 0.1 | 1 |
| Environment 1 | map3 | 4 | [3, 8, 29, 34] | 13 | 0.00036878842250259708 | 0.1 | 1 |
| Environment 2 | map7 | 2 | [0, 16] | 14 | 0.00065700197705703916 | 0.4 | 1 |
| Environment 2 | map7 | 3 | [0, 11, 16] | 11 | 0.00035512450575036835 | 0.4 | 1 |
| Environment 2 | map7 | 4 | [0, 11, 16, 21] | 9 | 0.00068233117855083579 | 0.4 | 1 |
| Environment 3 | map8 | 2 | [9, 30] | 35 | 0.00035396566065758791 | 0.3 | 1 |
| Environment 3 | map8 | 3 | [9, 12, 30] | 23 | 0.00022829998760764592 | 0.3 | 1 |
| Environment 3 | map8 | 4 | [9, 12, 30, 3] | 18 | 0.00022600981996761401 | 0.3 | 1 |

Map widths, edge geometry, and all remaining settings are recorded in the graph
assets and the task YAML files. Alpha is the reciprocal of the region count.

### Horizon rule

Construct coverage-and-fusion reference routes on the prior graph by enumerating
fixed robot priority orders. Visit the nearest uncovered regions and connect
components that have not yet fused. Among feasible reference candidates, choose
lexicographically by joint duration T, total distance D, and route sequence.
The reference is not selected by a learned score or by a post-training result.

```text
T_max = T_ref + max(2, ceil(0.15 * T_ref)).
```

T_ref is the joint step count of the constructed reference route. This heuristic
provides a reproducible task horizon; it is not a proof of minimum completion
time.

### Distance-penalty rule

Insert two- or four-edge closed detours before the reference route terminates;
other robots make legal return traversals. Replay each candidate with the actual
environment termination rule. Keep unique successful routes for which both
Delta S and Delta D are positive.

Prefer candidates satisfying Delta D / D_ref <= 0.20. If none satisfy that bound,
use all positive feasible detours and record the fallback. For the resulting
slopes Delta S / Delta D, compute their 25th and 75th percentiles:

```text
beta = sqrt(q25 * q75).
```

With no positive candidate, this rule does not yield a calibrated beta.
The nine calibration records retain the reference routes, detours, quantiles,
fallback flags, and final task parameters under `configs/simulation/calibration`.

## Seven-node illustration

The graph has seven regions and nine unit-length edges. Two robots start at
regions [0, 4], with T_max = 3 and rho_v = rho_g = 1.
Beta = 0.00022829998760764592 is an inherited fixed value; it was not independently
calibrated for the seven-node graph. Its horizon is also a preset and does not
use the nine-task horizon rule above.

GEVD uses a 10,000-call analysis interval; the QCO/QCF training runs use 20,000
simulator calls. The saved seed set is
[906300, 906301, 906410, 906411, 906412, 906510, 906511, 906512, 906513, 906514].
The manuscript's 5,000-episode label is not equivalent to these call budgets;
see [reproducibility notes](reproducibility.md).

## Learning settings

The main learner has hidden dimension 128, three graph message-passing layers,
gamma = 1, batch size 64, replay capacity 20,000, Double DQN targets, and internal
TD reward scale 0.1.

For the nine comparison tasks, the budget is 25,000 simulator calls. Learning rate
is 5e-5 through call 12,500, then decays linearly to 5e-6 at call 20,000. Epsilon
decays from 1 to 0.05 by call 15,000.

The seven-node GEVD configuration starts at learning rate 5e-6. Its stored
schedule begins decay at call 10,000 and reaches 5e-7 at call 16,000; epsilon
reaches 0.05 at call 12,000. Thus the supplied 10,000-call run does not traverse
the entire stored schedule.

The independent retrospective branch uses an amplitude cap of 0.1 |Q|,
execution multiplier 0.1, trace decay 0.9, auxiliary batch size 8, episode replay
capacity 256, event sampling fraction 0.5, and a trace horizon matching T_max.
QCO/QCF use intra-component information coefficient 7, with an additional success
bonus of 0.5 for QCF. QMIX and COMA retain their separate optimizer settings.

## Evaluation and budget accounting

```text
G = (S_T - S_0) + rho_v * (K_T - N) + rho_g * (N - c_T) - beta * D.
```

Training return, common evaluation G, structural score S, and success are separate
quantities. Budgeted factual transitions and candidate verification count toward
the simulator-call budget. Read-only evaluations and post-run audits are tracked
separately. A retained verified route and the final greedy policy must be
reported separately when they differ.

## Physical environment

The author-supplied image depicts seven regions and two starting regions, 0 and
4. It does not supply calibrated metric coordinates, T_max, or beta. Available
metadata and explicitly unknown fields live under `configs/real`. Do not transfer
parameters from the distinct later 58-node office environment to this setup.

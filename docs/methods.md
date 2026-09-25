# Methods and common evaluation

## GEVD

GEVD uses a shared graph-based value learner with Double DQN targets and a value
decomposition objective. An independent retrospective utility branch Q-prime
augments action selection. Its magnitude is capped at 0.1 times the magnitude of
the corresponding primary utility, and its execution multiplier is 0.1.

The common task score is

```text
G = (S_T - S_0) + rho_v (K_T - N) + rho_g (N - c_T) - beta D.
```

Here S is the structural score, K is the number of covered regions, N is the
number of robots, c is the number of unregistered components, and D is total
travel distance. Region coverage and component fusion are separate contributions.
The shared physical success test is retained across learning-method ablations.

## Comparison baselines

| Method | Task implementation |
|---|---|
| CMRE | Adapted coordinated coverage-route planner |
| sGre | CGE-style sequential greedy loop insertion |
| dGre | Adapted ordered double-greedy planner, candidate cap 32 |
| QMIX | Isolated QMIX with local MLP, Adam learning rate 5e-5 |
| COMA | Isolated COMA with actor/critic RMSprop learning rate 5e-4 |
| QCO | Seven-node QMIX using coverage, distance, and intra-component information; information coefficient 7 |
| QCF | QCO with an additional 0.5 success bonus |

QMIX and COMA use their own learners and optimizer settings. QCO and QCF have
different training rewards from GEVD; comparison uses the common evaluation
score rather than substituting each learner's training return for G. Planning
baselines are adaptations to the graph task, not full upstream navigation stacks.

## GEVD component ablations

| Variant | Change from GEVD |
|---|---|
| `no_retrospective` | Disable the independent auxiliary Q-prime branch |
| `no_gauge` | Remove the explicit gauge reward from primary factual TD replay |
| `no_vdn` | Use parameter-shared independent TD with the full team reward per local target |
| `raw_structure` | Use full-pose component structural increments in primary replay without representative marginalization |

The last three variants retain Q-prime. All four retain the common simulator,
evaluation score, and success definition. A training-reward ablation does not
remove the physical possibility of fusion. The supplied ablation configurations
use `map8/N3`; see [parameters](parameters.md) and
[reproducibility notes](reproducibility.md) for experiment scope.

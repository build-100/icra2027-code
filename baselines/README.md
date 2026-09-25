# Baselines

The implementations share the GEVD task environment and its common evaluation
metric. Each learning baseline uses its own network and optimizer. Planning
baselines adapt the cited methods to the same graph, action space and horizon;
they are not verbatim executions of the upstream ROS systems.

| Entry and method | Implementation |
| --- | --- |
| `run.py --method qmix` | Independent local MLP with a monotonic mixer; Adam, learning rate `5e-5` |
| `run.py --method coma` | Independent actor and counterfactual critic; RMSprop, learning rate `5e-4` |
| `run.py --method cmre` | Coordinated coverage-route planner |
| `run.py --method sgre` | CGE-style sequential greedy loop insertion |
| `run.py --method dgre` | Ordered randomized double-greedy loop insertion, candidate cap 32 |
| `run_coverage.py --method coverage_only` | QCO: seven-node QMIX with coverage, travel cost and intra-robot information |
| `run_coverage.py --method coverage_first` | QCF: the QCO objective plus a `0.5` success bonus |

From the repository root, validate the configuration without simulator calls:

```bash
python baselines/run.py --method qmix --map map8 --robots 3
python baselines/run_coverage.py --method coverage_only
```

Add `--run` to execute an experiment. Use `--budget` for the number of training
or planning simulator calls, and `--output results/generated/run_name` to select
a new output directory. Evaluation calls are recorded separately for learning
baselines. Existing output directories are protected against overwriting.

The comparison configurations are in `configs/simulation/comparison/`; the
seven-node configurations are in `configs/simulation/seven_node/`. QCO and QCF
retain their separate training objective: `training_return` is not the common
evaluation score `common_G`. Their intra-information coefficient is `7`.

`marl/` contains the comparison learners, `coverage/` contains the seven-node
learners and their original schedules, and `planning/` contains the route
planners. `budget.py` and `replay.py` provide shared bookkeeping. The two QMIX
implementations preserve their distinct checkpoint formats and settings.

For a trusted historical optimizer checkpoint, use
`baselines.checkpoints.load_checkpoint(path)` before passing its `learner`
payload to the matching learner's `load_checkpoint` method. The loader remaps
the old replay tuple module without requiring the previous source tree.

See the repository's third-party notices and references for upstream attribution.

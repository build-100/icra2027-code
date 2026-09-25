# Experiments and manuscript illustrations

The figures on this page were supplied with the manuscript. Raster originals are
preserved without redrawing; the method overview is rendered from the supplied
vector PDF. They illustrate the study and are not newly generated measurements.

## Cooperative mapping concept

![Two robots exploring with initially unregistered frames](assets/multi_robot_concept.png)

*Region-level planning couples local exploration with opportunities for inter-robot
closure. The two panels illustrate different route choices in the same setting.*

## Simulation environments

| Manuscript environment | Code map | Regions | Robot counts | Horizons for 2 / 3 / 4 robots |
|---|---|---:|---|---|
| Environment 1 | `map3` | 36 | 2, 3, 4 | 23 / 17 / 13 |
| Environment 2 | `map7` | 23 | 2, 3, 4 | 14 / 11 / 9 |
| Environment 3 | `map8` | 39 | 2, 3, 4 | 35 / 23 / 18 |

| Environment 1 (`map3`) | Environment 2 (`map7`) | Environment 3 (`map8`) |
|---|---|---|
| ![Environment 1](assets/environment_map3.png) | ![Environment 2](assets/environment_map7.png) | ![Environment 3](assets/environment_map8.png) |

*Green markers denote graph regions and the overlaid connections define the
planning structure. The YAML files and graph assets specify the executable tasks;
the raster figures are illustrations.*

### Seven-node example

![Seven-node routes and manuscript learning-curve illustration](assets/seven_node_manuscript.png)

*The supplied figure contrasts the GEVD routes with coverage-oriented routes and
shows the manuscript learning-curve illustration. Its horizontal axis is labeled
Episodes. Available experiment logs use budgeted simulator-call coordinates, and
the exact mapping to this plotted axis has not been verified. This image is not a
claim of a newly reproduced curve or a multi-seed mean.*

The seven-node task uses two robots starting at regions 0 and 4, nine unit-length
edges, and a horizon of three joint steps. See [parameters](parameters.md) for the
fixed beta and the learner budgets.

## Single-run ablation example

The following compact display corresponds to the saved routes behind manuscript
Table II on map8 with three robots (seed 906305). It reports a retained verified
successful route, or a saved complete failure when no success was retained.
These are not multi-seed means or final-policy scores. See the
[reproducibility notes](reproducibility.md) for the selection rule.

| Method | G | S | Complete fused map |
|---|---:|---:|---|
| GEVD | 12.865 | 0.292 | Yes |
| Without retrospective utility | 12.601 | 0.317 | No |
| Without value decomposition | 12.859 | 0.286 | Yes |
| Without fusion reward | 12.807 | 0.236 | Yes |
| Raw structural score | 12.561 | 0.291 | No |

## Physical setup

| Experiment arena | Robot platform |
|---|---|
| ![Two robots in the physical arena](assets/physical_experiment.jpg) | ![Robot platform used in the manuscript](assets/robot_platform.png) |

*Author-supplied photographs of the physical experiment and robot platform. The
portable release contains the high-level graph simulator; it does not include a
ROS navigation or sensor-processing deployment stack.*

![Seven-region abstraction of the physical environment](assets/physical_seven_regions.png)

*Seven-region abstraction: region 0 is the left passage; regions 1, 2, and 3 are
on the upper row, with regions 6, 5, and 4 below. The depicted starting regions
are 0 and 4. The image does not provide an executable metric topology or determine
T_max, beta, or map scale; unspecified configuration fields remain unspecified.*

### Structural-score comparison

| Separate exploration | Coordinated exploration |
|---|---|
| ![Separate routes, annotated structural score minus 0.100](assets/separate_routes.png) | ![Coordinated routes, annotated structural score 0.157](assets/coordinated_routes.png) |

*The supplied manuscript illustrations annotate S = -0.100 for the separate
routes and S = 0.157 for the coordinated routes. These are image annotations,
not scores recomputed from released sensor data. Physical recordings are held
separately and have not been replayed as part of this release.*

See [reproducibility and data availability](reproducibility.md) before interpreting
these illustrations as quantitative reproduction evidence.

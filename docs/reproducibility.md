# Reproducibility and data availability

This repository provides an executable GEVD implementation, comparison and
ablation code, task configurations, and author-supplied manuscript illustrations.
It does not distribute experiment logs, learned checkpoints, machine-readable result records, or
physical ROS recordings. Generated outputs stay under the local `results/`
directory. The distinctions below concern the experiment evidence available when
the release was prepared; they are not new training results.

## Seven-node illustration

Saved records identify ten seeds each for GEVD, coverage-only, and coverage-first.
The GEVD analysis interval ends at 10,000 budgeted simulator calls; the two
coverage baselines used 20,000 calls. The manuscript figure instead labels its
axis **Episodes**, up to 5,000. Those units are different, and the exact original
plotting input has not been established. The supplied figure is displayed as a
manuscript illustration, not as a newly reproduced learning curve. No confidence
intervals or multi-seed statistics have been inferred from that image.

## Main comparison

The 54 displayed G/S/success entries in Table I were traced to saved selected
routes at the manuscript's three-decimal precision. Their available sources are
single runs, not established ten-run means. Two GEVD-labeled entries use GEVD: `map7/N2`, seed 906300, and `map8/N3`, seed 906305. The other seven
GEVD-labeled task entries use the older primary-only backbone with the auxiliary
Q-prime branch disabled. The five comparison baselines use seed 906300.

The released GEVD configurations for all nine tasks enable new experiments;
they do not relabel older backbone results or establish complete nine-task
reproduction. An interrupted GEVD nine-task batch was cancelled and is not
presented as a completed experiment.

## Component ablations

The five Table II entries match seed 906305 on `map8/N3`, rather than an aggregate
across five seeds. Their selection uses a verified successful route retained
within budget, or a complete saved failure route when no such success exists.
This differs from reporting the final greedy policy. For example, the GEVD
retained route has G = 12.865144 and S = 0.291735; its final current policy has
G = 12.756921 and S = 0.186942. These interpretations must remain separate in any
future evaluation or manuscript revision.

## Physical experiment

The supplied images show two robots and a seven-region environment. The region
image supports a qualitative description of the setup and visible starting
regions 0 and 4. It does not determine metric scale, a calibrated horizon, beta,
or a calibrated machine-readable deployment graph. A figure-level, unscaled
adjacency transcription is included in `configs/real/paper7_external/illustrated_topology.json`. Unavailable deployment fields remain
explicitly unspecified in the configuration metadata.

The physical ROS recordings are held separately on Ubuntu and are not part of
this release. A later 58-node office experiment is a different environment and
must not be used to fill missing seven-region parameters. Structural-score labels
in the supplied images are manuscript annotations; no sensor-level replay has
been performed for this release.

## What validation establishes

The release reorganizes the implementation into importable packages and portable
command-line entries while retaining its mathematical definitions. It does not
claim byte-identical source files across that packaging change.

Scientific regression tests and structural score replay check implementation
behavior and saved-route arithmetic where local evidence is available (`scripts/verify_release.py --local-results`).
Public-only release verification does not require unpublished result bundles.
These checks do not prove that a fresh stochastic training run will reproduce a particular
retained route, a multi-seed mean, or a physical trajectory. CPU/GPU and library
versions can affect optimization trajectories. Record the exact configuration,
seed, software versions, budget counters, and selection rule for every new run.

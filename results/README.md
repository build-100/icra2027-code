# Local experiment results

Simulation results are organized as `simulation/seven_node`,
`simulation/comparison`, and `simulation/ablation`. The `bags` folders are
logical experiment bundles, not ROS bag files. Local `result.json`,
`provenance.json`, evaluation CSVs and `_raw` checkpoints/logs are not tracked
by Git. The same raw run may support more than one table entry.

New runs are written to `generated/` by default. QA runs are software checks,
not manuscript experiments. Physical recordings under `real/` remain external
on the authors' Ubuntu system. Supplied paper illustrations appear in docs;
they are not a replacement for raw measurements.

Run `python scripts/verify_release.py --local-results` in the local deployment
to validate saved route evidence. A clean public clone can verify source hashes,
calibration arithmetic and synthetic scientific tests without private results.

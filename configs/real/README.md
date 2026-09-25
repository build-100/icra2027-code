# Physical experiment metadata

The supplied manuscript and figures depict two robots in seven regions. The
paper images are provided in `docs/assets`; they establish the illustrated
layout, region labels and qualitative routes. They do not supply a calibrated
machine-readable graph, metric edge lengths, or the physical deployment's
T_max and beta. Unknown fields remain null in `paper7_external/status.json`.

Physical ROS recordings remain in the authors' Ubuntu system and are not part
of this release. No robot deployment or sensor-level replay is claimed here.
The later 58-node office environment belongs to a different experiment and
is excluded from this GEVD release.

`illustrated_topology.json` records the nine drawn adjacencies and starting
regions [0, 4] as qualitative figure evidence. Its coordinates, metric edge
lengths and deployment parameters remain unspecified; it is not a runnable
physical-experiment configuration.

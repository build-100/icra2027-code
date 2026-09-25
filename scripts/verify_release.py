"""Verify calibration, optional local result routes, and an explicit file manifest."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gevd.cli import ROOT, load, replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Local result directory to replay recursively.")
    parser.add_argument("--local-results", action="store_true", help="Replay the local results/simulation archive.")
    parser.add_argument("--manifest", type=Path, help="Override RELEASE_MANIFEST.json with another SHA-256 manifest.")
    args = parser.parse_args(argv)
    route_count = calls = hashes = 0
    if args.results or args.local_results:
        selected = args.results or Path("results/simulation")
        results = selected if selected.is_absolute() else ROOT / selected
        if not results.is_dir():
            parser.error(f"Results directory does not exist: {results}")
        for path in sorted(results.rglob("result.json")):
            if "_raw" in path.parts:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if "selected" not in row or "config" not in row:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                actual = replay(load(row["config"]), row)
            route_count += 1
            calls += actual["T"]
    calibration_count = 0
    for path in sorted((ROOT / "configs/simulation/calibration").glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))["task"]
        if task["t_max"] != task["T_ref"] + max(2, math.ceil(0.15 * task["T_ref"])):
            raise ValueError(f"Invalid horizon calibration: {path}")
        if not math.isclose(task["beta"], math.sqrt(task["beta_slope_q25"] * task["beta_slope_q75"]), rel_tol=1e-12):
            raise ValueError(f"Invalid beta calibration: {path}")
        calibration_count += 1
    manifest = args.manifest or (Path("RELEASE_MANIFEST.json") if (ROOT / "RELEASE_MANIFEST.json").exists() else None)
    if manifest:
        path = manifest if manifest.is_absolute() else ROOT / manifest
        for relative, expected in json.loads(path.read_text(encoding="utf-8"))["files"].items():
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f"Manifest path is outside the project: {relative}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"File hash mismatch: {relative}")
            hashes += 1
    print(json.dumps({"passed": True, "saved_routes_replayed": route_count,
        "replay_simulator_calls": calls, "calibrations_checked": calibration_count,
        "file_hashes_checked": hashes, "training_started": False}, indent=2))


if __name__ == "__main__":
    main()

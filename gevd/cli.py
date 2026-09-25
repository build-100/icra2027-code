"""Configure, train, and replay GEVD experiments from the project checkout."""
from pathlib import Path
import argparse
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/generated/.matplotlib"))


def load(config_path, device="cpu", seed=None, output=None):
    """Resolve paths relative to the checkout and apply explicit CLI overrides."""
    import torch
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["vdn"]["device"] = device
    if seed is not None:
        cfg["vdn"]["seed"] = seed
        cfg.setdefault("experiment", {})["seed"] = seed
    cfg["environment"]["graph_path"] = str(ROOT / cfg["environment"]["graph_path"])
    if output is not None:
        cfg["output"]["directory"] = str(ROOT / output)
    torch.set_num_threads(int(cfg["vdn"].get("cpu_threads", 1)))
    return cfg


def replay(cfg, result):
    """Recompute a saved joint route under the common GEVD evaluation objective."""
    from .environments.prior_graph import PriorGraph
    from .environments.multi_robot import GEVDMultiRobotEnv

    env_cfg, reward = cfg["environment"], dict(cfg["reward"])
    # Coverage baselines train with their own potential and share this evaluation.
    for key in ("rho_v", "rho_g", "beta"):
        if key in cfg.get("evaluation_reward", {}):
            reward[key] = cfg["evaluation_reward"][key]
    env = GEVDMultiRobotEnv(
        PriorGraph(env_cfg["graph_path"], env_cfg["map_width"], env_cfg["start_nodes"][0], "cpu"),
        env_cfg["start_nodes"], env_cfg["t_max"], reward["alpha"], reward["beta"],
        reward["rho_v"], reward["rho_g"], reward["pair_factor_weight"],
    )
    selected = result.get("selected", result)
    if not isinstance(selected, dict) or "routes" not in selected:
        raise ValueError("The result must contain a saved joint route.")
    routes = selected["routes"]
    if len(routes) != env_cfg["robot_count"] or len(set(map(len, routes))) != 1:
        raise ValueError("Expected equal-length joint routes for all robots.")
    if not all(routes) or [route[0] for route in routes] != env_cfg["start_nodes"]:
        raise ValueError("The route start nodes differ from the configuration.")
    env.reset()
    total = distance = 0.0
    for labels in zip(*[route[1:] for route in routes]):
        if env.done:
            raise ValueError("The route continues after termination.")
        step = env.step([
            env.prior.action_for_neighbor(a, b)
            for a, b in zip(env.current_labels, labels)
        ])
        total += step.reward
        distance += step.travel_distance
    actual = {
        "G": total, "S": env.structural_score(), "D": distance, "T": env.state.time,
        "coverage": env.coverage_count(), "c": env.component_count(), "success": bool(env.success),
    }
    for key in ("G", "S", "coverage", "c", "success"):
        if key in result and not math.isclose(float(actual[key]), float(result[key]), abs_tol=1e-8):
            raise ValueError(f"{key}: saved={result[key]} replay={actual[key]}")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulation/seven_node/gevd.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/generated/gevd"))
    parser.add_argument("--budget", type=int, help="Maximum counted simulator calls for a fresh run.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Start one fresh training run.")
    mode.add_argument("--replay", type=Path, help="Replay a local saved result without training.")
    args = parser.parse_args(argv)
    cfg = load(args.config, args.device, args.seed, args.output)
    if args.replay:
        path = args.replay if args.replay.is_absolute() else ROOT / args.replay
        print(json.dumps(replay(cfg, json.loads(path.read_text(encoding="utf-8"))), indent=2))
        return
    from .training.gevd import GEVDTrainer

    if not cfg.get("retrospective_utility", {}).get("enabled") or cfg.get("full_gevd_ablation"):
        parser.error("GEVD requires retrospective utility; use python -m ablations.run for variants.")
    if not args.run:
        trainer = GEVDTrainer.from_config(cfg)
        print(json.dumps({
            "status": "preflight_passed", "algorithm": "GEVD", "nodes": trainer.env.num_nodes,
            "edges": trainer.env.prior.prior_graph.number_of_edges(), "N": trainer.env.num_robots,
            "starts": trainer.env.start_labels, "T_max": trainer.env.t_max, "simulator_calls": 0,
            "seed": trainer.seed, "device": str(trainer.device),
        }, indent=2))
        return
    if (ROOT / args.output).exists():
        parser.error(f"Existing output is protected: {ROOT / args.output}")
    from .training.runner import run_budgeted_main

    budget = args.budget if args.budget is not None else cfg["search"]["total_simulator_budget"]
    print(json.dumps(run_budgeted_main(cfg, budget, trainer_class=GEVDTrainer), indent=2))


if __name__ == "__main__":
    main()

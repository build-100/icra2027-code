"""Run individual GEVD ablations with a shared evaluation objective."""
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gevd.cli import ROOT, load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["no_retrospective", "no_gauge", "no_vdn", "raw_structure"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=906305)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path(f"results/generated/{args.variant}_seed{args.seed}")
    cfg = load(f"configs/simulation/ablation/map8_N3/{args.variant}.yaml", args.device, args.seed, output)
    from gevd.training.search import BudgetedSearchTrainer
    from ablations.variants import GEVDAblationTrainer

    trainer_type = BudgetedSearchTrainer if args.variant == "no_retrospective" else GEVDAblationTrainer
    if not args.run:
        trainer = trainer_type.from_config(cfg)
        print(json.dumps({"status": "preflight_passed", "algorithm": trainer.algorithm_name,
            "variant": args.variant, "N": trainer.env.num_robots,
            "nodes": trainer.env.num_nodes, "simulator_calls": 0}, indent=2))
        return
    if (ROOT / output).exists():
        parser.error(f"Existing output is protected: {ROOT / output}")
    from gevd.training.runner import run_budgeted_main

    budget = args.budget if args.budget is not None else cfg["search"]["total_simulator_budget"]
    print(json.dumps(run_budgeted_main(cfg, budget, trainer_class=trainer_type), indent=2))


if __name__ == "__main__":
    main()

"""CLI de ChurnShield-Agent; S01 permanece bloqueado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .s03 import run as run_s03


def main() -> None:
    parser = argparse.ArgumentParser(prog="churnshield")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True)
    run.add_argument("--config", default="entrega_manus/config/agente.yaml")
    run.add_argument("--json", action="store_true")
    sub.add_parser("dev-eval")
    phase3 = sub.add_parser("phase3")
    phase3.add_argument("--repetition-seed", type=int, default=101)
    phase3.add_argument("--output-dir", default="artifacts/phase3")
    phase4 = sub.add_parser("phase4")
    phase4.add_argument("--scenario", choices=["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"])
    phase4.add_argument("--all", action="store_true")
    phase4.add_argument("--repetition-seed", type=int, default=101)
    phase4.add_argument("--all-seeds", action="store_true")
    phase4.add_argument("--output-dir", default="artifacts/phase4")
    args = parser.parse_args()
    if args.command == "run" and args.scenario.upper() == "S03":
        result = run_s03()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("S03_OK")
            print(f"n_active={result['n_active']}")
            print(f"k={result['k']}")
            print(f"n_selected={result['n_selected']}")
            print(f"coverage={result['coverage']:.6f}")
            print("selected_ids=" + ",".join(row["customer_id"] for row in result["decisions"] if row["selected"]))
        return
    if args.command == "dev-eval":
        from .dev_eval import run_dev_eval
        print(json.dumps(run_dev_eval(), indent=2, default=float))
        return
    if args.command == "phase3":
        from .phase3 import run_phase3
        print(json.dumps(run_phase3(repetition_seed=args.repetition_seed, output_dir=args.output_dir), indent=2, default=float))
        return
    if args.command == "phase4":
        from .phase4 import SCENARIOS, run_many, run_scenario
        if args.all:
            scenarios = SCENARIOS
        elif args.scenario:
            scenarios = (args.scenario,)
        else:
            raise SystemExit("phase4 requiere --scenario S02..S10 o --all")
        if args.all_seeds:
            print(json.dumps(run_many(scenarios, output_dir=args.output_dir), indent=2, default=float))
        else:
            print(json.dumps([run_scenario(s, seed=args.repetition_seed, output_dir=args.output_dir) for s in scenarios], indent=2, default=float))
        return
    raise SystemExit("Fase 1 solo implementa el escenario S03")


if __name__ == "__main__":
    main()

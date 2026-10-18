"""CLI de ChurnShield-Agent; S01 permanece bloqueado."""

from __future__ import annotations

import argparse
import json
from .s03 import run as run_s03


def main() -> None:
    parser = argparse.ArgumentParser(prog="churnshield")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True)
    run.add_argument("--config", default="config/agente.yaml")
    run.add_argument("--json", action="store_true")
    sub.add_parser("dev-eval")
    simulation = sub.add_parser(
        "simulate", help="ejecuta la simulación longitudinal de cohortes"
    )
    simulation.add_argument("--seed", type=int, default=101)
    simulation.add_argument("--output-dir", default="artifacts/cohort_simulation")
    simulation.add_argument("--evaluator-output-dir", default=None)
    scenario_parser = sub.add_parser(
        "scenarios", help="ejecuta escenarios de retención"
    )
    from .scenarios.retention import SCENARIOS

    scenario_parser.add_argument("--scenario", choices=SCENARIOS)
    scenario_parser.add_argument("--all", action="store_true")
    scenario_parser.add_argument("--seed", type=int, default=101)
    scenario_parser.add_argument("--all-seeds", action="store_true")
    scenario_parser.add_argument("--output-dir", default="artifacts/scenarios")
    scenario_parser.add_argument("--evaluator-output-dir", default=None)
    evaluator = sub.add_parser("evaluate", help="calcula M2 y M3 desde los logs")
    evaluator.add_argument("--input-dir", default="artifacts/scenarios")
    evaluator.add_argument("--output", default=None)
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
            print(
                "selected_ids="
                + ",".join(
                    row["customer_id"] for row in result["decisions"] if row["selected"]
                )
            )
        return
    if args.command == "dev-eval":
        from .dev_eval import run_dev_eval

        print(json.dumps(run_dev_eval(), indent=2, default=float))
        return
    if args.command == "simulate":
        from .simulation.cohort import run_simulation

        print(
            json.dumps(
                run_simulation(
                    repetition_seed=args.seed,
                    output_dir=args.output_dir,
                    evaluator_output_dir=args.evaluator_output_dir,
                ),
                indent=2,
                default=float,
            )
        )
        return
    if args.command == "scenarios":
        from .simulation.scenario_runner import run_many, run_scenario

        if args.all:
            selected_scenarios = SCENARIOS
        elif args.scenario:
            selected_scenarios = (args.scenario,)
        else:
            raise SystemExit("scenarios requiere --scenario S02..S10 o --all")
        if args.all_seeds:
            print(
                json.dumps(
                    run_many(
                        selected_scenarios,
                        output_dir=args.output_dir,
                        evaluator_output_dir=args.evaluator_output_dir,
                    ),
                    indent=2,
                    default=float,
                )
            )
        else:
            print(
                json.dumps(
                    [
                        run_scenario(
                            s,
                            seed=args.seed,
                            output_dir=args.output_dir,
                            evaluator_output_dir=args.evaluator_output_dir,
                        )
                        for s in selected_scenarios
                    ],
                    indent=2,
                    default=float,
                )
            )
        return
    if args.command == "evaluate":
        from pathlib import Path
        from .evaluation import evaluate_batch

        result = evaluate_batch(args.input_dir)
        rendered = json.dumps(result, indent=2, default=float)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return
    raise SystemExit("Solo está implementado el escenario S03 para el comando run")


if __name__ == "__main__":
    main()

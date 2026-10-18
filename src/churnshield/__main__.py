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
    run.add_argument("--config", default="config/agent.yaml")
    run.add_argument("--json", action="store_true")
    dev_evaluation = sub.add_parser(
        "dev-eval", help="entrena y evalúa usando únicamente desarrollo"
    )
    dev_evaluation.add_argument(
        "--output-dir", default="artifacts/public/dev_eval/stacking"
    )
    simulation = sub.add_parser(
        "simulate", help="ejecuta la simulación longitudinal de cohortes"
    )
    simulation.add_argument("--seed", type=int, default=101)
    simulation.add_argument(
        "--output-dir", default="artifacts/public/cohort_simulation"
    )
    simulation.add_argument("--evaluator-output-dir", default=None)
    scenario_parser = sub.add_parser(
        "scenarios", help="ejecuta escenarios de retención"
    )
    from .scenarios.retention import SCENARIOS

    scenario_parser.add_argument("--scenario", choices=SCENARIOS)
    scenario_parser.add_argument("--all", action="store_true")
    scenario_parser.add_argument("--seed", type=int, default=101)
    scenario_parser.add_argument("--all-seeds", action="store_true")
    scenario_parser.add_argument("--output-dir", default="artifacts/public/scenarios")
    scenario_parser.add_argument("--evaluator-output-dir", default=None)
    evaluator = sub.add_parser("evaluate", help="calcula M2 y M3 desde los logs")
    evaluator.add_argument("--input-dir", default="artifacts/public/scenarios")
    evaluator.add_argument(
        "--output", default="artifacts/public/scenario_evaluation/evaluation.json"
    )
    evaluator.add_argument(
        "--expected-scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=None,
        help="conjunto exacto esperado para rechazar ejecuciones antiguas",
    )
    evaluator.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=None,
        help="semillas exactas esperadas junto con --expected-scenarios",
    )
    report_parser = sub.add_parser(
        "scenario-report",
        help="genera un resumen ejecutivo por escenario en JSON, CSV y Markdown",
    )
    report_parser.add_argument(
        "--input-dir", default="artifacts/public/scenarios"
    )
    report_parser.add_argument(
        "--evaluation",
        default="artifacts/public/scenario_evaluation/evaluation.json",
    )
    report_parser.add_argument(
        "--output-dir", default="artifacts/public/scenario_report"
    )
    report_parser.add_argument(
        "--expected-scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=None,
    )
    report_parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=None
    )
    report_parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="espera las semillas 101..105 para todos los escenarios S02..S10",
    )
    final_evaluator = sub.add_parser(
        "evaluate-final",
        help="ejecuta la evaluación final reservada con autorización explícita",
    )
    final_evaluator.add_argument(
        "--confirm-final",
        action="store_true",
        help="confirma que la evaluación reservada fue autorizada",
    )
    final_evaluator.add_argument(
        "--output-dir", default="artifacts/private/final_evaluation"
    )
    final_evaluator.add_argument(
        "--authorization-file",
        default=None,
        help="archivo JSON externo con dos aprobaciones hash-pinned",
    )
    sub.add_parser(
        "prepare-final-authorization",
        help="imprime plantilla de autorización sin cargar filas reservadas",
    )
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

        print(
            json.dumps(
                run_dev_eval(output_dir=args.output_dir),
                indent=2,
                default=float,
            )
        )
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

        if (args.expected_scenarios is None) != (args.expected_seeds is None):
            raise SystemExit(
                "--expected-scenarios y --expected-seeds deben usarse juntos"
            )
        expected_runs = None
        if args.expected_scenarios is not None:
            expected_runs = {
                (scenario, seed)
                for scenario in args.expected_scenarios
                for seed in args.expected_seeds
            }
        result = evaluate_batch(args.input_dir, expected_runs=expected_runs)
        rendered = json.dumps(result, indent=2, default=float)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return
    if args.command == "scenario-report":
        if args.all_seeds and (
            args.expected_scenarios is not None or args.expected_seeds is not None
        ):
            raise SystemExit(
                "--all-seeds no se puede combinar con --expected-scenarios/--expected-seeds"
            )
        if (args.expected_scenarios is None) != (args.expected_seeds is None):
            raise SystemExit(
                "--expected-scenarios y --expected-seeds deben usarse juntos"
            )
        expected_runs = None
        if args.all_seeds:
            expected_runs = {
                (scenario, seed)
                for scenario in SCENARIOS
                for seed in (101, 102, 103, 104, 105)
            }
        elif args.expected_scenarios is not None:
            expected_runs = {
                (scenario, seed)
                for scenario in args.expected_scenarios
                for seed in args.expected_seeds
            }
        from .scenario_report import generate_scenario_report

        result = generate_scenario_report(
            input_dir=args.input_dir,
            evaluation_path=args.evaluation,
            output_dir=args.output_dir,
            expected_runs=expected_runs,
        )
        print(json.dumps(result, indent=2, default=float))
        return
    if args.command == "evaluate-final":
        from .final_eval import run_final_evaluation

        print(
            json.dumps(
                run_final_evaluation(
                    output_dir=args.output_dir,
                    confirm_final=args.confirm_final,
                    authorization_file=args.authorization_file,
                ),
                indent=2,
                default=float,
            )
        )
        return
    if args.command == "prepare-final-authorization":
        from .protocol import final_authorization_template

        print(json.dumps(final_authorization_template(), indent=2))
        return
    raise SystemExit("Solo está implementado el escenario S03 para el comando run")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

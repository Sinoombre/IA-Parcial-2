"""CLI de la Fase 1."""

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
    raise SystemExit("Fase 1 solo implementa el escenario S03")


if __name__ == "__main__":
    main()

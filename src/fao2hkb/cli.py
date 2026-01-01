from __future__ import annotations

import argparse
import json
import sys

from . import run as run_pipeline

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fao2hkb")
    sub = p.add_subparsers(dest="cmd", required=True)

    prun = sub.add_parser("run", help="Run FAO2HKB pipeline from a YAML config.")
    prun.add_argument("--config", required=True, help="Path to config.yaml")
    prun.add_argument("--json", action="store_true", help="Print output as JSON")
    prun.add_argument(
        "--execution",
        action="store_true",
        help="Enable execution timeline tracking under <run_dir>/work/execution/.",
    )

    args = p.parse_args(argv)
    # Default behavior: if neither --json nor --execution is given,
    # behave as if both were enabled (JSON + execution timeline).
    if args.cmd == 'run' and not args.json and not getattr(args, 'execution', False):
        args.json = True
        args.execution = True


    if args.cmd == "run":
        out = run_pipeline(args.config, execution=bool(getattr(args, "execution", False)))
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("Run complete.")
            for k, v in out.items():
                print(f"- {k}: {v}")

            if getattr(args, "execution", False):
                # convenience hint for humans
                mj = out.get("execution_milestones_jsonl") or out.get("execution_milestones_json")
                if mj:
                    print(f"\nExecution timeline: {mj}")

if __name__ == "__main__":
    main(sys.argv[1:])
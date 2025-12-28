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

    args = p.parse_args(argv)

    if args.cmd == "run":
        out = run_pipeline(args.config)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("Run complete.")
            for k, v in out.items():
                print(f"- {k}: {v}")

if __name__ == "__main__":
    main(sys.argv[1:])

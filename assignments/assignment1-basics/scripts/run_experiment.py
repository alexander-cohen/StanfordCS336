from __future__ import annotations

import argparse
import json

from cs336_basics.experiment import CONFIG_FIELDS, ExperimentConfigError, resolve_experiment, run_experiment


class ConfigOverride(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        overrides = dict(getattr(namespace, self.dest, None) or {})
        key = "runtime.device" if option_string == "--device" else option_string[2:]
        overrides[key] = values
        setattr(namespace, self.dest, overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and validate a CS336 experiment configuration.", allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/smoke.toml", help="Preset path relative to the project root.")
    parser.add_argument(
        "--local-config",
        default="configs/local.toml",
        help="Machine-local path configuration relative to the project root.",
    )
    parser.add_argument("--data-root", help="Override the data root (default: assignment1-basics/data; also configurable via CS336_DATA_ROOT or configs/local.toml).")
    parser.add_argument("--runs-root", help="Override the runs root (default: assignment1-basics/runs; also configurable via CS336_RUNS_ROOT or configs/local.toml).")
    overrides = parser.add_argument_group("TOML overrides", "Use --section.key VALUE or --section.key=VALUE; last occurrence wins.")
    for key in CONFIG_FIELDS:
        flags = [f"--{key}"]
        if key == "runtime.device":
            flags.append("--device")
        overrides.add_argument(*flags, dest="overrides", action=ConfigOverride, metavar="VALUE",
                               help=f"Override {key} (arrays use TOML/JSON syntax).")
    parser.add_argument("--resume-from", help="Checkpoint to continue in a new attempt directory.")
    parser.add_argument("--stop-after", type=int, help="Stop at this absolute step without changing the full budget.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without writing artifacts or training.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        resolved = resolve_experiment(
            args.config,
            local_config_path=args.local_config,
            data_root_override=args.data_root,
            runs_root_override=args.runs_root,
            overrides=args.overrides,
        )
        if args.dry_run:
            print(json.dumps(resolved, indent=2, sort_keys=True))
        else:
            result = run_experiment(resolved, resume_from=args.resume_from, stop_after=args.stop_after)
            print(f"Run artifacts: {result['output_dir']}")
    except ExperimentConfigError as error:
        parser.exit(status=2, message=f"configuration error: {error}\n")



if __name__ == "__main__":
    main()

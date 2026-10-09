import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from torch_submission import client


def setup(host, account, root):
    cfg = {"host": host, "account": account, "root": root, "repo": root + "/repo"}
    client.SETTINGS.write_text(json.dumps(cfg, indent=2) + "\n")
    client.ssh(shlex.join(["mkdir", "-p", cfg["repo"]]))
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude=.venv/",
            "--exclude=.git/",
            "--exclude=notes/",
            "--exclude=data/",
            "--exclude=furu/",
            "--exclude=output/",
            "--exclude=logs/",
            "--exclude=__pycache__/",
            "--exclude=.pytest_cache/",
            "--exclude=.ruff_cache/",
            "--exclude=.env",
            "--exclude=settings.local.json",
            "--exclude=torch-results/",
            str(client.REPO) + "/",
            f"{host}:{shlex.quote(cfg['repo'])}/",
        ],
        check=True,
    )
    print(
        client.ssh(
            shlex.join(["bash", cfg["repo"] + "/torch_submission/setup.sh", root])
        ),
        end="",
    )
    print(json.dumps(cfg, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Submit CS336 configurations to one L40S on NYU Torch"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--host", default="torch")
    p.add_argument("--account", default="torch_pr_1323_general")
    p.add_argument("--root", default="/scratch/ac6074/cs336-a3")
    p = sub.add_parser("prepare-data")
    p.add_argument("--train-tokens", type=int, default=2**32)
    for name in ["submit", "final"]:
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
    for name in ["get", "cancel"]:
        p = sub.add_parser(name)
        p.add_argument("experiment_id", type=int)
    sub.add_parser("budget")
    sub.add_parser("list")
    p = sub.add_parser("fetch")
    p.add_argument("destination", nargs="?", default="torch-results")
    p.add_argument("--experiment-id", type=int)
    p = sub.add_parser("data-status")
    args = parser.parse_args()
    if args.action == "setup":
        setup(args.host, args.account, args.root)
        return
    if args.action == "submit":
        result = client.submit_experiment(json.loads(args.config.read_text()))
    elif args.action == "final":
        result = client.request(
            "final",
            "--account",
            client.settings()["account"],
            payload=json.loads(args.config.read_text()),
        )
    elif args.action == "prepare-data":
        result = client.request(
            "prepare-data",
            "--account",
            client.settings()["account"],
            "--train-tokens",
            args.train_tokens,
        )
    elif args.action == "fetch":
        result = {
            "downloaded_to": client.fetch_results(args.destination, args.experiment_id)
        }
    elif args.action == "data-status":
        result = client.request("data-status")
    else:
        result = client.request(
            args.action,
            *([args.experiment_id] if args.action in {"get", "cancel"} else []),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

"""SSH client usable from either the CLI or a local notebook."""

import json
import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SETTINGS = REPO / "torch_submission" / "settings.local.json"


def settings():
    result = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    result["host"] = os.environ.get("A3_TORCH_HOST", result.get("host", "torch"))
    result["root"] = os.environ.get(
        "A3_TORCH_ROOT", result.get("root", "/scratch/ac6074/cs336-a3")
    )
    result["account"] = os.environ.get(
        "A3_TORCH_ACCOUNT", result.get("account", "torch_pr_1323_general")
    )
    result["repo"] = os.environ.get(
        "A3_TORCH_REPO",
        result.get(
            "repo", "/scratch/ac6074/StanfordCS336/assignments/assignment3-scaling"
        ),
    )
    return result


def ssh(command, payload=None):
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            settings()["host"],
            command,
        ],
        input=payload,
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"ssh exited {proc.returncode}"
        )
    return proc.stdout


def request(action, *args, payload=None):
    cfg = settings()
    argv = [
        f"{cfg['root']}/venv/bin/python",
        "-m",
        "torch_submission.server",
        "--root",
        cfg["root"],
        action,
        *map(str, args),
    ]
    command = f"cd {shlex.quote(cfg['repo'])} && {shlex.join(argv)}"
    return json.loads(
        ssh(command, None if payload is None else json.dumps(payload, allow_nan=False))
    )


def as_json(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def submit_experiment(training_config):
    return request(
        "submit", "--account", settings()["account"], payload=as_json(training_config)
    )


def get_budget():
    return request("budget")


def list_experiments():
    return request("list")


def get_experiment(experiment_id):
    return request("get", experiment_id)


def submit_final(training_config, predicted_final_loss):
    return request(
        "final",
        "--account",
        settings()["account"],
        payload={
            "training_config": as_json(training_config),
            "predicted_final_loss": predicted_final_loss,
        },
    )


def fetch_results(destination, experiment_id=None):
    # Refresh Slurm accounting before copying the ledger and human-readable totals.
    get_budget()
    cfg = settings()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    remote = (
        cfg["root"]
        if experiment_id is None
        else f"{cfg['root']}/runs/{int(experiment_id)}"
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude=venv/",
            "--exclude=/repo/",
            "--exclude=data/",
            "--exclude=hf-cache/",
            "--exclude=jax-cache/",
            "--exclude=ledger.sqlite3*",
            "--exclude=*.tmp",
            f"{cfg['host']}:{shlex.quote(remote)}/",
            str(destination) + "/",
        ],
        check=True,
    )
    return str(destination)

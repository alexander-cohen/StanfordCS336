"""Login-node commands; training runs only in single-GPU Slurm jobs."""

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from torch_submission.store import ACTIVE, Store, atomic_json, now

REPO = Path(__file__).resolve().parent.parent
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"{argv[0]} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def refresh(store):
    # Recover the narrow crash window between successful sbatch and storing its
    # output, matching both job name and the unique snapshot path.
    for record in store.records():
        if record["status"]["status_type"] == "submitting" and not record.get("job_id"):
            name = f"a3-{record['kind']}-{record['experiment_id']}"
            found = command(["squeue", "--me", "-h", "--name", name, "-o", "%i|%o"])
            for line in found.splitlines():
                job_id, _, job_command = line.partition("|")
                if (
                    str(store.run_dir(record["experiment_id"])) in job_command
                    and job_id.isdigit()
                ):

                    def recovered(r):
                        r["job_id"] = job_id
                        r["status"]["status_type"] = "queued"

                    store.update(record["experiment_id"], recovered)
                    break
    records = [
        r
        for r in store.records()
        if r.get("job_id")
        and (r["status"]["status_type"] in ACTIVE or not r.get("accounting_final"))
    ]
    if not records:
        with store.transaction() as db:
            store.publish(db)
        return
    output = command(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(r["job_id"] for r in records),
            "--format=JobIDRaw,State,ElapsedRaw,End,ExitCode",
        ]
    )
    rows = {}
    for line in output.splitlines():
        values = line.split("|")
        if len(values) >= 5:
            rows[values[0]] = values[1:5]
    for record in records:
        row = rows.get(record["job_id"])
        if row is None:
            continue  # Keep reservations until Slurm has authoritative accounting.
        state, elapsed, end, exit_code = row
        state = state.split()[0].rstrip("+")
        if state not in TERMINAL:
            continue
        eid = record["experiment_id"]
        current = store.get(eid)
        if current["status"]["status_type"] in ACTIVE:
            used = 0.0
            if current.get("training_started_unix"):
                # sacct dates use the login node's local timezone.
                end_time = dt.datetime.fromisoformat(end).timestamp()
                used = max(0.0, end_time - current["training_started_unix"])
            reason = {
                "reason": "unexpected",
                "failure": f"Slurm {state}, exit {exit_code}; worker did not finish reporting",
            }
            store.finish(eid, "failed", used, reason)
            if (
                not current.get("training_started_unix")
                and state == "CANCELLED"
                and float(elapsed) == 0
            ):
                store.update(eid, lambda r: r.update(charged_seconds=0.0))
        store.update(
            eid,
            lambda r: r.update(
                job_elapsed_seconds=float(elapsed),
                accounting_final=True,
                slurm_state=state,
                slurm_exit_code=exit_code,
            ),
        )


def validate_data(root, config):
    import numpy as np

    path = root / "data" / "manifest.json"
    if not path.exists():
        raise ValueError(
            "data is not ready; check data preparation logs before submitting"
        )
    manifest = json.loads(path.read_text())
    files = manifest["train_filepaths"]
    if len(files) < 2:
        raise ValueError(
            "the course reader requires at least two training shards for cross-shard next-token labels"
        )
    shard_size = manifest["n_train_tokens"] // len(files)
    if shard_size % config.seq_len:
        raise ValueError("training shard length must be divisible by 512")
    if config.total_train_tokens > shard_size * (len(files) - 1):
        raise ValueError(
            "requested tokens exceed usable prepared data (last shard is reserved for cross-shard labels); prepare more data in a new environment"
        )
    if manifest["n_validation_tokens"] < config.n_val_tokens + 1:
        raise ValueError("not enough validation tokens")
    for filename in files + [manifest["validation_filepath"]]:
        arr = np.load(filename, mmap_mode="r")
        expected = (
            manifest["n_validation_tokens"]
            if filename == manifest["validation_filepath"]
            else shard_size
        )
        if arr.dtype != np.uint16 or arr.ndim != 1 or len(arr) != expected:
            raise ValueError(f"invalid token shard: {filename}")
    if config.architecture_config.vocab_size != 32000:
        raise ValueError("DCLM/Llama-2 data requires vocab_size=32000")
    return manifest


def submit(store, payload, kind, account):
    from cs336_scaling.schemas import FinalSubmissionRequest
    from cs336_scaling.training.training_config import TrainingConfig

    predicted = None
    if kind == "final":
        final = FinalSubmissionRequest.model_validate(payload)
        config, predicted = final.training_config, final.predicted_final_loss
    else:
        config = TrainingConfig.model_validate(payload)
    if not math.isfinite(config.max_runtime_seconds) or (
        predicted is not None and not math.isfinite(predicted)
    ):
        raise ValueError("runtime and prediction must be finite")
    manifest = validate_data(store.root, config)
    refresh(store)
    record = store.reserve(
        config.model_dump(mode="json"), config.unique_id, kind, predicted
    )
    eid = record["experiment_id"]
    run_dir = store.run_dir(eid)
    try:
        source = run_dir / "source"
        source.mkdir()
        for folder in ["cs336_scaling", "torch_submission"]:
            shutil.copytree(
                REPO / folder,
                source / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        atomic_json(run_dir / "training_config.json", config.model_dump(mode="json"))
        atomic_json(run_dir / "data_manifest.json", manifest)
        atomic_json(
            run_dir / "source_hashes.json",
            {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob("*.py"))
            },
        )
        if kind == "final":
            atomic_json(run_dir / "final_submission.json", payload)
        # Setup/compilation do not consume training budget; Slurm accounting
        # records them separately. Slurm owns the hard allocation deadline.
        wall_minutes = math.ceil((config.max_runtime_seconds * 1.05 + 10 + 3600) / 60)
        job_id = command(
            [
                "sbatch",
                "--parsable",
                f"--account={account}",
                "--gres=gpu:1",
                "--constraint=l40s",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--no-requeue",
                "--comment=",
                f"--time={wall_minutes}",
                f"--job-name=a3-{kind}-{eid}",
                f"--output={run_dir}/slurm.out",
                f"--error={run_dir}/slurm.err",
                str(source / "torch_submission" / "slurm.sh"),
                str(run_dir),
                str(store.root),
                str(eid),
                str(store.root / "venv" / "bin" / "python"),
            ]
        ).split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch job ID: {job_id}")

        def submitted(r):
            r["job_id"] = job_id
            if r["status"]["status_type"] == "submitting":
                r["status"]["status_type"] = "queued"

        store.update(eid, submitted)
        if kind == "final":
            atomic_json(
                store.root / "final_submission.json",
                {
                    **payload,
                    "experiment_id": eid,
                    "job_id": job_id,
                    "submitted_at": now(),
                },
            )
        return {
            "experiment_id": eid,
            "job_id": job_id,
            "budget_summary": store.budget(),
        }
    except BaseException as exc:
        failure = str(exc)
        if not store.get(eid).get("job_id"):
            store.update(
                eid,
                lambda r: r["status"].update(
                    status_type="submission_failed", failure=failure
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ["submit", "final"]:
        p = sub.add_parser(name)
        p.add_argument("--account", required=True)
    sub.add_parser("budget")
    sub.add_parser("list")
    sub.add_parser("data-status")
    p = sub.add_parser("get")
    p.add_argument("experiment_id", type=int)
    p = sub.add_parser("cancel")
    p.add_argument("experiment_id", type=int)
    p = sub.add_parser("prepare-data")
    p.add_argument("--account", required=True)
    p.add_argument("--train-tokens", type=int, default=2**32)
    args = parser.parse_args()
    store = Store(args.root)
    if args.action == "data-status":
        job_file = store.root / "data_job.json"
        job = json.loads(job_file.read_text()) if job_file.exists() else {}
        state = (
            command(["sacct", "-X", "-n", "-P", "-j", job["job_id"], "--format=State"])
            if job
            else "not submitted"
        )

        def tail(filename):
            path = store.root / filename
            return (
                path.read_text(errors="replace").splitlines()[-12:]
                if path.exists()
                else []
            )

        result = {
            **job,
            "ready": (store.root / "data" / "manifest.json").exists(),
            "slurm_state": state,
            "stdout_tail": tail("prepare-data.out"),
            "stderr_tail": tail("prepare-data.err"),
        }
    elif args.action in {"submit", "final"}:
        result = submit(
            store,
            json.load(sys.stdin),
            args.action if args.action == "final" else "experiment",
            args.account,
        )
    elif args.action == "prepare-data":
        if (store.root / "data" / "manifest.json").exists():
            raise ValueError(
                "prepared data already exists; use a new root to change datasets"
            )
        script = REPO / "torch_submission" / "prepare_data.sh"
        job = command(
            [
                "sbatch",
                "--parsable",
                f"--account={args.account}",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=12:00:00",
                "--no-requeue",
                "--comment=",
                "--job-name=a3-data",
                f"--output={store.root}/prepare-data.out",
                f"--error={store.root}/prepare-data.err",
                str(script),
                str(REPO),
                str(store.root),
                str(args.train_tokens),
            ]
        )
        atomic_json(
            store.root / "data_job.json",
            {"job_id": job.split(";")[0], "train_tokens": args.train_tokens},
        )
        result = {
            "job_id": job.split(";")[0],
            "logs": str(store.root / "prepare-data.out"),
        }
    else:
        refresh(store)
        if args.action == "budget":
            result = store.budget()
        elif args.action == "list":
            result = store.records()
        elif args.action == "get":
            result = store.get(args.experiment_id)
        else:
            record = store.get(args.experiment_id)
            if record["status"]["status_type"] not in ACTIVE:
                raise ValueError("job is already finished")
            command(["scancel", record["job_id"]])
            result = {"cancel_requested": record["job_id"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

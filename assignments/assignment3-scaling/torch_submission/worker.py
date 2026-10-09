import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from torch_submission.store import Store, atomic_json, now
from torch_submission.telemetry import RunTelemetry


def training_seconds(record):
    started = record.get("training_started_unix")
    return max(0.0, time.time() - started) if started else 0.0


def child(root, eid):
    import jax
    import wandb
    from cs336_scaling.tokenized_data import DCLMDataResult
    from cs336_scaling.training.driver import run_training_driver
    from cs336_scaling.training.training_config import TrainingConfig

    store = Store(root)
    record = store.get(eid)
    run_dir = store.run_dir(eid)
    config = TrainingConfig.model_validate(record["training_config"])
    devices = jax.devices()
    if (
        len(devices) != 1
        or devices[0].platform != "gpu"
        or "L40S" not in devices[0].device_kind
    ):
        raise RuntimeError(f"expected exactly one L40S GPU, got {devices}")
    telemetry = RunTelemetry(devices[0], config.total_train_tokens)
    atomic_json(
        run_dir / "environment.json",
        {
            "jax": jax.__version__,
            "python": sys.version,
            "devices": [str(d) for d in devices],
            "gpu": devices[0].device_kind,
        },
    )
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    data = DCLMDataResult(
        validation_filepath=Path(manifest["validation_filepath"]),
        train_filepaths=[Path(p) for p in manifest["train_filepaths"]],
        n_train_tokens=manifest["n_train_tokens"],
        n_validation_tokens=manifest["n_validation_tokens"],
        n_source_chunks=manifest["n_source_chunks"],
    )

    def report_event(experiment_id, event, **kwargs):
        payload = event.model_dump(mode="json")
        with (run_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps({"at": now(), **payload}) + "\n")
        if event.log_type == "validation_loss":
            if not all(math.isfinite(v) for v in event.val_losses):
                raise RuntimeError("non-finite validation loss")
            snapshot = telemetry.snapshot(read_memory=False)

            def progress_record(r):
                r["status"].update(val_losses=list(event.val_losses))
                # A delayed validation callback must not overwrite final metrics.
                if r["status"]["status_type"] == "running":
                    r["metrics"] = snapshot

            store.update(eid, progress_record)
        elif event.log_type == "run_heartbeat":
            store.update(
                eid,
                lambda r: r["status"].update(
                    elapsed_training_seconds=event.elapsed_seconds
                ),
            )

    def report_finish(experiment_id, payload, **kwargs):
        metrics = telemetry.snapshot(
            payload.used_runtime_seconds, completed=payload.result_type == "completed"
        )
        if payload.result_type == "completed":
            if not all(math.isfinite(v) for v in payload.val_losses):
                raise RuntimeError("non-finite validation loss")
            store.finish(
                eid,
                "completed",
                payload.used_runtime_seconds,
                val_losses=payload.val_losses,
                metrics=metrics,
            )
        else:
            store.finish(
                eid,
                "failed",
                payload.used_runtime_seconds,
                payload.reason.model_dump(mode="json"),
                metrics=metrics,
            )

    def configure(path):
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.FileHandler(path), logging.StreamHandler()],
            force=True,
        )

    def record_model(count):
        telemetry.model_created(count)
        store.update(eid, lambda r: r.update(metrics=telemetry.snapshot()))

    runtime = SimpleNamespace(
        run_id=os.environ.get("SLURM_JOB_ID", "local"),
        output_path=run_dir,
        cache_path=Path(root) / "jax-cache",
        data=SimpleNamespace(try_load=lambda: data),
        local_data_dir=Path(os.environ["TMPDIR"]) / "data",
        metrics=wandb,
        configure_logging=configure,
        report_event=report_event,
        report_finish=report_finish,
        cancel=lambda: os._exit(124),
        training_started=lambda: store.update(
            eid, lambda r: r.update(training_started_unix=time.time())
        ),
        model_created=record_model,
        training_progress=telemetry.progress,
    )

    result = run_training_driver(config, os.environ.get("USER", "local"), eid, runtime)
    atomic_json(
        run_dir / "training_result.json",
        {**vars(result), "metrics": store.get(eid)["metrics"]},
    )


def supervise(root, eid):
    store = Store(root)
    started = time.monotonic()

    def started_record(r):
        r["job_started_unix"] = time.time()
        r["status"].update(
            status_type="running",
            dispatched_at=now(),
            run_id=os.environ.get("SLURM_JOB_ID"),
        )

    store.update(eid, started_record)
    proc = subprocess.Popen(
        [sys.executable, "-m", "torch_submission.worker", root, str(eid), "--child"],
        start_new_session=True,
    )

    def interrupted(signum, frame):
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        code = proc.wait()
    finally:
        record = store.get(eid)
        if record["status"]["status_type"] == "running":
            store.finish(
                eid,
                "failed",
                training_seconds(record),
                {
                    "reason": "unexpected",
                    "failure": f"worker interrupted or exited with code {proc.returncode}",
                },
            )
        store.update(
            eid, lambda r: r.update(job_elapsed_seconds=time.monotonic() - started)
        )
    return code


if __name__ == "__main__":
    state_dir, experiment_id = sys.argv[1], int(sys.argv[2])
    if "--child" in sys.argv:
        child(state_dir, experiment_id)
    else:
        sys.exit(supervise(state_dir, experiment_id))

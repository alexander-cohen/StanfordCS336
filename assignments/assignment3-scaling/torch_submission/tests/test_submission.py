import concurrent.futures
import json
import subprocess

import numpy as np
import pytest

from torch_submission import server
from torch_submission.store import Store


def config():
    return {
        "architecture_config": {
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 32000,
        },
        "optimizer_config": {
            "lr_scheduler": {
                "peak_value": 3e-4,
                "final_lr_frac": 0.1,
                "warmup_frac": 0.05,
                "init_value": 0,
            }
        },
        "train_batch_size": 1,
        "val_batch_size": 1,
        "n_evals": 1,
        "total_train_tokens": 512,
        "max_runtime_seconds": 60,
        "model_seed": 0,
    }


def data(root):
    directory = root / "data"
    directory.mkdir()
    files = []
    for i in range(2):
        path = directory / f"train-{i}.npy"
        np.save(path, np.zeros(1024, dtype=np.uint16))
        files.append(str(path))
    val = directory / "val.npy"
    np.save(val, np.zeros(2**18 + 512, dtype=np.uint16))
    manifest = {
        "train_filepaths": files,
        "validation_filepath": str(val),
        "n_train_tokens": 2048,
        "n_validation_tokens": 2**18 + 512,
        "n_source_chunks": 1,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_concurrent_submissions_cannot_overreserve(tmp_path):
    store = Store(tmp_path)
    with store.transaction() as db:
        db.execute("UPDATE settings SET value='90' WHERE key='budget_seconds'")

    def reserve(i):
        try:
            return store.reserve(config(), str(i), "experiment")
        except ValueError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert sum(r is not None for r in results) == 1
    assert store.budget()["reserved_seconds"] == 60
    assert json.loads((tmp_path / "budget.json").read_text())["remaining_seconds"] == 30


def test_concurrent_final_submissions_allow_two_jobs(tmp_path):
    store = Store(tmp_path)

    def reserve(i):
        try:
            return store.reserve(config(), str(i), "final", 3.0 + i)
        except ValueError as exc:
            assert str(exc) == "at most two final jobs may be active"
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    accepted = [r for r in results if r is not None]
    assert len(accepted) == 2
    assert len({r["experiment_id"] for r in accepted}) == 2
    assert store.budget()["reserved_seconds"] == 0
    assert store.budget()["remaining_seconds"] == 43200


def test_final_slots_released_after_completion(tmp_path):
    store = Store(tmp_path)
    first = store.reserve(config(), "first", "final", 4.0)
    second = store.reserve(config(), "second", "final", 3.5)
    store.update(
        first["experiment_id"], lambda r: r["status"].update(status_type="running")
    )
    store.update(
        second["experiment_id"], lambda r: r["status"].update(status_type="queued")
    )
    with pytest.raises(ValueError, match="at most two final jobs"):
        store.reserve(config(), "third", "final", 3.0)
    store.finish(first["experiment_id"], "completed", 50, val_losses=[3.9])
    third = store.reserve(config(), "third", "final", 3.0)
    assert third["experiment_id"] == 3
    assert store.get(first["experiment_id"])["status"]["val_losses"] == [3.9]
    assert store.get(second["experiment_id"])["status"]["status_type"] == "queued"
    assert store.budget()["remaining_seconds"] == 43200


def test_final_runtime_limit_still_applies(tmp_path):
    store = Store(tmp_path)
    too_long = config()
    too_long["max_runtime_seconds"] = 172801
    with pytest.raises(ValueError, match="at most 48 hours"):
        store.reserve(too_long, "too-long", "final", 3.0)
    assert store.records() == []


def test_budget_refund_failure_clamp_and_final_separation(tmp_path):
    store = Store(tmp_path)
    a = store.reserve(config(), "a", "experiment")
    store.finish(a["experiment_id"], "completed", 12, val_losses=[4.0])
    b = store.reserve(config(), "b", "experiment")
    store.finish(
        b["experiment_id"],
        "failed",
        90,
        {"reason": "timeout", "partial_val_losses": []},
    )
    c = store.reserve(config(), "final", "final", 3.0)
    store.finish(c["experiment_id"], "completed", 50, val_losses=[3.0])
    assert store.budget()["spent_seconds"] == 72
    assert store.budget()["reserved_seconds"] == 0
    assert store.budget()["remaining_seconds"] == 43200 - 72
    assert store.budget()["total_training_seconds"] == 102
    with pytest.raises(ValueError, match="already been submitted"):
        store.reserve(config(), "a", "experiment")


def test_submission_snapshot_resources_and_result(tmp_path, monkeypatch):
    store = Store(tmp_path)
    data(tmp_path)
    calls = []

    def slurm(argv):
        calls.append(argv)
        return "12345" if argv[0] == "sbatch" else ""

    monkeypatch.setattr(server, "command", slurm)
    result = server.submit(store, config(), "experiment", "account")
    assert result["job_id"] == "12345"
    assert "--gres=gpu:1" in calls[0] and "--constraint=l40s" in calls[0]
    assert "--no-requeue" in calls[0]
    source = store.run_dir(1) / "source"
    assert (source / "cs336_scaling/training/loop.py").read_bytes() == (
        server.REPO / "cs336_scaling/training/loop.py"
    ).read_bytes()
    assert (store.run_dir(1) / "source_hashes.json").exists()
    assert store.get(1)["status"]["status_type"] == "queued"


def test_sbatch_failure_releases_reservation(tmp_path, monkeypatch):
    store = Store(tmp_path)
    data(tmp_path)

    def fail(argv):
        raise subprocess.CalledProcessError(1, argv, stderr="invalid account")

    monkeypatch.setattr(server, "command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        server.submit(store, config(), "experiment", "bad")
    assert store.budget()["remaining_seconds"] == 43200
    assert store.get(1)["status"]["status_type"] == "submission_failed"


def test_explicit_infrastructure_refund_allows_retry(tmp_path):
    store = Store(tmp_path)
    record = store.reserve(config(), "a", "experiment")
    store.finish(record["experiment_id"], "failed", 0, {"reason": "unexpected"})
    store.update(
        record["experiment_id"],
        lambda r: r.update(charged_seconds=0.0, retry_allowed=True),
    )
    assert store.budget()["spent_seconds"] == 0
    retry = store.reserve(config(), "a", "experiment")
    assert retry["experiment_id"] == 2
    with pytest.raises(ValueError, match="already been submitted"):
        store.reserve(config(), "a", "experiment")


def test_sacct_reconciles_hard_kill_and_queue_cancel(tmp_path, monkeypatch):
    store = Store(tmp_path)
    a = store.reserve(config(), "a", "experiment")
    store.update(
        a["experiment_id"], lambda r: r.update(job_id="100", training_started_unix=1000)
    )
    b = store.reserve(config(), "b", "experiment")
    store.update(b["experiment_id"], lambda r: r.update(job_id="101"))
    monkeypatch.setattr(
        server,
        "command",
        lambda _: (
            "100|OUT_OF_MEMORY|120|1970-01-01T00:18:20+00:00|1:0|\n101|CANCELLED by 0|0|1970-01-01T00:18:20+00:00|0:15|"
        ),
    )
    server.refresh(store)
    assert store.budget()["reserved_seconds"] == 0
    assert store.budget()["spent_seconds"] == 60
    assert store.budget()["total_job_seconds"] == 120
    assert store.get(1)["status"]["used_runtime_seconds"] == 100
    assert store.get(2)["charged_seconds"] == 0
    server.refresh(store)
    assert store.budget()["spent_seconds"] == 60


def test_invalid_configuration_and_data_do_not_reserve(tmp_path, monkeypatch):
    store = Store(tmp_path)
    data(tmp_path)
    c = config()
    c["total_train_tokens"] = 513
    with pytest.raises(ValueError, match="divisible"):
        server.submit(store, c, "experiment", "account")
    c = config()
    c["total_train_tokens"] = 2048
    with pytest.raises(ValueError, match="exceed usable"):
        server.submit(store, c, "experiment", "account")
    assert store.records() == []


def test_final_prediction_saved_and_excluded_from_experiment_budget(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    data(tmp_path)
    monkeypatch.setattr(server, "command", lambda _: "12345")
    payload = {"training_config": config(), "predicted_final_loss": 4.25}
    server.submit(store, payload, "final", "account")
    assert store.budget()["reserved_seconds"] == 0
    assert store.get(1)["predicted_final_loss"] == 4.25
    assert (
        json.loads((tmp_path / "final_submission.json").read_text())[
            "predicted_final_loss"
        ]
        == 4.25
    )
    second_payload = {"training_config": config(), "predicted_final_loss": 3.75}
    server.submit(store, second_payload, "final", "account")
    assert store.get(1)["status"]["status_type"] == "queued"
    assert store.get(2)["status"]["status_type"] == "queued"
    assert store.budget()["reserved_seconds"] == 0
    for eid, prediction in [(1, 4.25), (2, 3.75)]:
        assert store.get(eid)["predicted_final_loss"] == prediction
        saved = json.loads((store.run_dir(eid) / "final_submission.json").read_text())
        assert saved["predicted_final_loss"] == prediction
        assert (store.run_dir(eid) / "source_hashes.json").exists()
    assert (
        json.loads((tmp_path / "final_submission.json").read_text())["experiment_id"]
        == 2
    )

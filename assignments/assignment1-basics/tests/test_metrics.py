import json
import os
from types import SimpleNamespace

import pytest
import torch

from cs336_basics.metrics import metric_logger, format_metrics
from cs336_basics.experiment import run_experiment
from tests.test_run_experiment import setup_experiment


def test_tracking_mirrors_metrics_and_closes_on_error(tmp_path, monkeypatch):
    """Test online wiring without uploading anything or needing credentials."""
    import sys

    resolved, _, _, _ = setup_experiment(tmp_path)
    resolved["tracking"]["wandb_mode"] = "online"
    calls, logged, exits = [], [], []

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, kind, value, traceback):
            exits.append(kind)

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, record, *, step):
            logged.append((record, step))

    def init(**kwargs):
        calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs))
    output = tmp_path / "execution"
    output.mkdir()
    record = {"step": 3, "train_loss": 1.5, "run_id": "logical-run"}
    with pytest.raises(RuntimeError, match="training failed"):
        with metric_logger(output, resolved, verbose=False) as log:
            log(record)
            raise RuntimeError("training failed")
    assert calls[0]["id"] == output.name
    assert calls[0]["group"] == "logical-run"
    assert "resolved_paths" not in calls[0]["config"]
    assert logged == [({"step": 3, "train_loss": 1.5}, 3)]
    assert exits == [RuntimeError]
    assert json.loads((output / "metrics.jsonl").read_text()) == record


@pytest.mark.skipif(os.environ.get("RUN_WANDB_INTEGRATION") != "1", reason="Opt-in local W&B SDK test")
@pytest.mark.timeout(60)
def test_real_offline_tracking_preserves_training(tmp_path):
    """Exercise the installed SDK locally; never contact the W&B service."""
    resolved, _, _, _ = setup_experiment(tmp_path)
    baseline = run_experiment(resolved, verbose=False)
    resolved["tracking"]["wandb_mode"] = "offline"
    tracked = run_experiment(resolved, verbose=False)
    for key, value in baseline["model"].state_dict().items():
        torch.testing.assert_close(value, tracked["model"].state_dict()[key], rtol=0, atol=0)
    assert list((tracked["output_dir"] / "wandb").glob("offline-run-*/*.wandb"))
    assert (tracked["output_dir"] / "summary.json").exists()


def test_disabled_tracking_only_writes_local_metrics(tmp_path, monkeypatch, capsys):
    import sys

    resolved, _, _, _ = setup_experiment(tmp_path)
    # An accidental import fails rather than initializing a third-party service.
    monkeypatch.setitem(sys.modules, "wandb", None)
    output = tmp_path / "execution"
    output.mkdir()
    record = {"step": 1, "train_loss": 2.0, "run_id": "logical-run"}
    with metric_logger(output, resolved, verbose=False) as log:
        log(record)
    assert json.loads((output / "metrics.jsonl").read_text()) == record
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_console_format_preserves_stored_precision(tmp_path, capsys):
    resolved, _, _, _ = setup_experiment(tmp_path)
    output = tmp_path / "console"
    output.mkdir()
    record = {
        "step": 2, "train_loss": 3.123456789, "validation_loss": 3.23456789,
        "learning_rate": 0.001, "elapsed_seconds": 3661.9,
        "train_tokens_per_second": 12345.6, "estimated_train_flops_per_second": 12_345_000_000,
    }
    with metric_logger(output, resolved) as log:
        log(record)
    terminal = capsys.readouterr().err
    assert "2/4" in terminal and "3.1235" in terminal
    assert "12,346" in terminal and "01:01:01" in terminal
    assert "est. FLOP/s" in terminal and "1.23e+10" in terminal
    assert json.loads((output / "metrics.jsonl").read_text()) == record
    # Missing validation or warmup throughput must not look like measured zero.
    assert "—" in format_metrics({"step": 1, "train_loss": 4.0})


def test_cpu_memory_does_not_query_accelerators(monkeypatch):
    from cs336_basics import metrics

    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=3 * 1024**3)))
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU logging must not query GPU memory")
    monkeypatch.setattr(torch.cuda, "mem_get_info", unexpected)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", unexpected)
    record = metrics.memory_metrics("cpu")
    assert record == {"memory_process_rss_bytes": 3 * 1024**3}
    assert "RAM RSS 3.00 GiB" in format_metrics({"step": 1, "train_loss": 1, **record})


def test_cuda_memory_units_and_device_scope(monkeypatch):
    from cs336_basics import metrics

    gib = 1024**3
    values = {
        "mem_get_info": (2 * gib, 24 * gib),
        "memory_allocated": 8 * gib,
        "memory_reserved": 12 * gib,
        "max_memory_allocated": 10 * gib,
        "max_memory_reserved": 14 * gib,
        "memory_stats": {"num_alloc_retries": 3, "num_ooms": 1},
    }
    def getter(value):
        def query(device):
            assert device == torch.device("cuda:1")
            return value
        return query
    for name, value in values.items():
        monkeypatch.setattr(torch.cuda, name, getter(value))
    record = metrics.memory_metrics("cuda:1")
    assert record["memory_cuda_allocated_bytes"] == 8 * gib
    assert record["memory_cuda_peak_allocated_bytes"] == 10 * gib
    assert record["memory_cuda_peak_reserved_bytes"] == 14 * gib
    assert record["memory_cuda_num_alloc_retries"] == 3
    assert record["memory_cuda_num_ooms"] == 1
    assert record["memory_process_rss_bytes"] > 0
    terminal = format_metrics({"step": 1, "train_loss": 1, **record})
    assert "GPU alloc 8.00 peak 10.00 reserved 12.00 free 2.00 / total 24.00 GiB" in terminal
    assert "alloc retries 3" in terminal


def test_mps_memory_uses_recommended_limit_label(monkeypatch):
    from cs336_basics import metrics

    for name, value in (("current_allocated_memory", 1024**3),
                        ("driver_allocated_memory", 2 * 1024**3),
                        ("recommended_max_memory", 8 * 1024**3)):
        monkeypatch.setattr(torch.mps, name, lambda value=value: value)
    record = metrics.memory_metrics("mps")
    assert record["memory_mps_allocated_bytes"] == 1024**3
    assert "memory_cuda_free_bytes" not in record
    terminal = format_metrics({"step": 1, "train_loss": 1, **record})
    assert "MPS alloc 1.00 driver 2.00 recommended max 8.00 GiB" in terminal

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
        "train_tokens_per_second": 12345.6, "estimated_train_tflops": 0.012345,
    }
    with metric_logger(output, resolved) as log:
        log(record)
    terminal = capsys.readouterr().err
    assert "2/4" in terminal and "3.1235" in terminal
    assert "12,346" in terminal and "01:01:01" in terminal
    assert "0.0123" in terminal
    assert json.loads((output / "metrics.jsonl").read_text()) == record
    # Missing validation or warmup throughput must not look like measured zero.
    assert "—" in format_metrics({"step": 1, "train_loss": 4.0})

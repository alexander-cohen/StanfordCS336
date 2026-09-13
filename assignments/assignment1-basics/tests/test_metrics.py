import json
import os
from types import SimpleNamespace

import pytest
import torch

from cs336_basics.metrics import metric_logger
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

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from cs336_basics.experiment import ExperimentConfigError, resolve_experiment, run_experiment
from tests.test_experiment_config import make_project


def setup_experiment(tmp_path):
    data, runs, preset = make_project(tmp_path)
    # Use actual NPY files here: resolver unit tests only need file existence,
    # but an end-to-end training test must exercise real array loading.
    np.save(data / "tiny/train.npy", np.arange(100, dtype=np.int32) % 100)
    np.save(data / "tiny/valid.npy", np.arange(100, dtype=np.int32)[::-1] % 100)
    resolved = resolve_experiment(
        preset, root=tmp_path, environment={},
        data_root_override=str(data), runs_root_override=str(runs))
    return resolved, preset, data, runs


def test_run_and_resume_with_validation(tmp_path):
    resolved, _, _, _ = setup_experiment(tmp_path)
    original = copy.deepcopy(resolved)
    full = run_experiment(resolved, verbose=False)
    first = run_experiment(resolved, stop_after=2, verbose=False)
    resumed = run_experiment(
        resolved, resume_from=first["output_dir"] / "2.pt", verbose=False)
    assert resolved == original
    saved_config = json.loads((full["output_dir"] / "config.resolved.json").read_text())
    assert saved_config == resolved
    assert resumed["run_id"] == first["run_id"]
    assert len({r["output_dir"] for r in (full, first, resumed)}) == 3
    for key, value in full["model"].state_dict().items():
        torch.testing.assert_close(value, resumed["model"].state_dict()[key], rtol=0, atol=0)
    assert full["last_metric"]["validation_loss"] == resumed["last_metric"]["validation_loss"]
    records = [json.loads(line) for line in (full["output_dir"] / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records if "validation_loss" in row] == [2, 4]
    assert json.loads((first["output_dir"] / "summary.json").read_text())["status"] == "paused"
    assert json.loads((resumed["output_dir"] / "summary.json").read_text())["status"] == "complete"


def test_invalid_tokens_fail_before_creating_run(tmp_path):
    resolved, _, data, runs = setup_experiment(tmp_path)
    np.save(data / "tiny/train.npy", np.full(100, 100, dtype=np.int32))
    with pytest.raises(ExperimentConfigError, match="outside"):
        run_experiment(resolved)
    assert not runs.exists()


def test_resume_after_relocation_and_comment_change(tmp_path):
    import shutil

    resolved, preset, data, _ = setup_experiment(tmp_path)
    full = run_experiment(resolved, verbose=False)
    first = run_experiment(resolved, stop_after=2, verbose=False)
    moved = tmp_path / "moved-data"
    shutil.copytree(data, moved)
    preset.write_text(preset.read_text() + "\n# A comment does not change training\n")
    updated = resolve_experiment(
        preset, root=tmp_path, environment={}, data_root_override=str(moved),
        runs_root_override=str(tmp_path / "moved-runs"))
    updated["logging"]["every_steps"] = 3
    resumed = run_experiment(updated, resume_from=first["output_dir"] / "2.pt", verbose=False)
    for key, value in full["model"].state_dict().items():
        torch.testing.assert_close(value, resumed["model"].state_dict()[key], rtol=0, atol=0)
    updated["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="differs"):
        run_experiment(updated, resume_from=first["output_dir"] / "2.pt", verbose=False)


def test_retention_and_console_validation(tmp_path, capsys):
    resolved, _, _, _ = setup_experiment(tmp_path)
    resolved["checkpoint"].update(every_steps=1, keep_last=2)
    result = run_experiment(resolved)
    output = result["output_dir"]
    assert sorted(path.name for path in output.glob("*.pt")) == ["3.pt", "4.pt"]
    assert "validation_loss=" in capsys.readouterr().err
    assert not (output / "events.jsonl").exists()
    assert (output / "summary.json").exists()


def test_summary_failure_never_reports_complete(tmp_path, monkeypatch):
    from cs336_basics import experiment

    resolved, _, _, runs = setup_experiment(tmp_path)
    original = experiment.atomic_json
    def fail(payload, path):
        if path.name == "summary.json":
            raise OSError("summary failed")
        original(payload, path)
    monkeypatch.setattr(experiment, "atomic_json", fail)
    with pytest.raises(OSError, match="summary failed"):
        run_experiment(resolved, verbose=False)
    output, = runs.iterdir()
    assert not (output / "summary.json").exists()
    assert (output / "4.pt").exists()


def test_cli_dry_run_and_training(tmp_path):
    _, preset, data, runs = setup_experiment(tmp_path)
    # CLI tokenizers are relative to the real project; use its existing files.
    text = preset.read_text().replace("tokenizers/vocab.pkl", "tokenizers/ts_train_10000_vocab.pkl")
    preset.write_text(text.replace("tokenizers/merges.pkl", "tokenizers/ts_train_10000_merges.pkl"))
    command = [sys.executable, "-m", "scripts.run_experiment", "--config", str(preset),
               "--data-root", str(data), "--runs-root", str(runs)]
    dry = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)["training"]["steps"] == 4
    assert not runs.exists()
    subprocess.run(command, capture_output=True, text=True, check=True)
    attempts = list(runs.iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "4.pt").exists()

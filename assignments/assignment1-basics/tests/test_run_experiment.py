import copy
import json
import pickle
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from cs336_basics.experiment import ExperimentConfigError, resolve_experiment, run_experiment
from tests.test_experiment_config import make_project


def setup_experiment(tmp_path):
    data, runs, preset = make_project(tmp_path)
    # A small real tokenizer whose IDs match this test model. All prompt bytes
    # are printable ASCII, so this vocabulary is sufficient for sample generation.
    vocab = {i: bytes([i + 32]) for i in range(99)}
    vocab[99] = b"<|endoftext|>"
    (tmp_path / "tokenizers/vocab.pkl").write_bytes(pickle.dumps(vocab))
    (tmp_path / "tokenizers/merges.pkl").write_bytes(pickle.dumps([]))
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
    # Git evidence is collected at launch, not during preset resolution. This
    # temporary fixture is outside Git, so unavailable metadata must be explicit.
    code = {key: saved_config["provenance"].pop(key) for key in ("git_commit", "git_dirty", "git_error")}
    assert code["git_commit"] is None and code["git_dirty"] is None
    assert code["git_error"]
    summary = json.loads((full["output_dir"] / "summary.json").read_text())
    assert all(summary[key] == value for key, value in code.items())
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


@pytest.mark.parametrize("missing_when", ["resolve", "launch"])
def test_missing_validation_warns_and_trains(tmp_path, missing_when):
    resolved, preset, data, runs = setup_experiment(tmp_path)
    (data / "tiny/valid.npy").unlink()
    with pytest.warns(UserWarning, match="continuing without validation"):
        if missing_when == "resolve":
            resolved = resolve_experiment(
                preset, root=tmp_path, environment={},
                data_root_override=str(data), runs_root_override=str(runs))
        result = run_experiment(resolved, verbose=False)
    assert result["last_metric"]["step"] == 4
    rows = [json.loads(line) for line in (result["output_dir"] / "metrics.jsonl").read_text().splitlines()]
    assert all("validation_loss" not in row for row in rows)
    saved = json.loads((result["output_dir"] / "config.resolved.json").read_text())
    assert saved["resolved_paths"]["validation_data"] is None
    assert saved["input_metadata"]["validation_data_size_bytes"] is None
    assert (result["output_dir"] / "4.pt").is_file()


def test_same_validation_file_warns_and_evaluates(tmp_path):
    _, preset, data, runs = setup_experiment(tmp_path)
    preset.write_text(preset.read_text().replace('validation_file = "tiny/valid.npy"',
                                               'validation_file = "tiny/train.npy"'))
    with pytest.warns(UserWarning, match="same file"):
        resolved = resolve_experiment(
            preset, root=tmp_path, environment={},
            data_root_override=str(data), runs_root_override=str(runs))
    result = run_experiment(resolved, verbose=False)
    assert result["last_metric"]["step"] == 4
    assert "validation_loss" in result["last_metric"]


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
    captured = capsys.readouterr()
    assert "| val " in captured.err
    assert "Sample (step " in captured.err
    assert not (output / "events.jsonl").exists()
    assert (output / "summary.json").exists()


def test_tokenizer_vocabulary_must_match_model(tmp_path):
    resolved, _, _, runs = setup_experiment(tmp_path)
    resolved["model"]["vocab_size"] += 1
    with pytest.raises(ExperimentConfigError, match="Tokenizer vocabulary"):
        run_experiment(resolved, verbose=False)
    assert not runs.exists()


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
    text = text.replace("vocab_size = 100\n", "vocab_size = 10000\n")
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


def test_ctrl_c_saves_safe_step_and_resumes_exactly(tmp_path, monkeypatch):
    from cs336_basics import training

    resolved, _, _, _ = setup_experiment(tmp_path)
    for section in ("evaluation", "logging", "checkpoint"):
        resolved[section]["every_steps"] = 3
    full = run_experiment(resolved, verbose=False)
    original_step = training.AdamW.step
    count = 0
    previous_handler = signal.getsignal(signal.SIGINT)

    def step(self, *args, **kwargs):
        nonlocal count
        count += 1
        # Request interruption before update 2 finishes: the handler must return
        # so the optimizer completes before any checkpoint can be written.
        if count == 2:
            signal.raise_signal(signal.SIGINT)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(training.AdamW, "step", step)
    stopped = run_experiment(resolved, verbose=False)
    assert signal.getsignal(signal.SIGINT) == previous_handler
    output = stopped["output_dir"]
    assert stopped["last_metric"]["step"] == 2
    assert "validation_loss" in stopped["last_metric"]
    assert (output / "2.pt").exists()
    assert not (output / "3.pt").exists()
    assert json.loads((output / "summary.json").read_text())["status"] == "paused"
    assert json.loads((output / "metrics.jsonl").read_text())["step"] == 2
    resumed = run_experiment(resolved, resume_from=output / "2.pt", verbose=False)
    for key, value in full["model"].state_dict().items():
        torch.testing.assert_close(value, resumed["model"].state_dict()[key], rtol=0, atol=0)


def test_second_ctrl_c_interrupts_and_restores_handler():
    from cs336_basics.experiment import _graceful_stop

    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        with _graceful_stop() as should_stop:
            assert not should_stop()
            signal.raise_signal(signal.SIGINT)
            assert should_stop()
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) == previous_handler

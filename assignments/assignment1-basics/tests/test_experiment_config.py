from pathlib import Path

import pytest

from cs336_basics.experiment import ExperimentConfigError, resolve_experiment


PRESET = """
schema_version = 1

[experiment]
name = "test"
seed = 7
tags = ["smoke"]

[data]
name = "tiny"
version = "v1"
train_file = "tiny/train.npy"
validation_file = "tiny/valid.npy"

[tokenizer]
vocab_file = "tokenizers/vocab.pkl"
merges_file = "tokenizers/merges.pkl"
end_of_text = "<|endoftext|>"

[model]
vocab_size = 100
context_length = 8
num_heads = 2
num_layers = 1
d_model = 16
d_ff = 32
rope_theta = 10000.0

[training]
batch_size = 2
steps = 4
grad_clip = 1.0
lr_mode = "basic"
learning_rate = 0.001
weight_decay = 0.0
betas = [0.9, 0.999]
epsilon = 1e-8

[evaluation]
every_steps = 2
batches = 1

[logging]
every_steps = 1

[checkpoint]
every_steps = 2
keep_last = 2

[runtime]
device = "cpu"

[tracking]
wandb_mode = "disabled"
wandb_project = "test"
"""


def make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "external-data"
    runs_root = tmp_path / "external-runs"
    (data_root / "tiny").mkdir(parents=True)
    (data_root / "tiny" / "train.npy").write_bytes(b"train")
    (data_root / "tiny" / "valid.npy").write_bytes(b"valid")
    (tmp_path / "tokenizers").mkdir()
    (tmp_path / "tokenizers" / "vocab.pkl").write_bytes(b"vocab")
    (tmp_path / "tokenizers" / "merges.pkl").write_bytes(b"merges")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "smoke.toml").write_text(PRESET)
    return data_root, runs_root, tmp_path / "configs" / "smoke.toml"


@pytest.mark.parametrize("budget,steps,requested,actual", [
    ("steps = 4", 4, None, 64),
    ("total_tokens = 64", 4, 64, 64),
    ("total_tokens = 65", 5, 65, 80),
    ("total_tokens = 1", 1, 1, 16),
    ("total_tokens = 9007199254740993", 562949953421313, 9007199254740993, 9007199254741008),
])
def test_budget_resolution(tmp_path, budget, steps, requested, actual):
    data_root, runs_root, preset = make_project(tmp_path)
    preset.write_text(PRESET.replace("steps = 4", budget))
    resolved = resolve_experiment(preset, data_root_override=str(data_root),
                                  runs_root_override=str(runs_root), environment={}, root=tmp_path)
    assert resolved["training"]["steps"] == steps
    assert resolved["resolved_budget"] == {
        "source": "steps" if requested is None else "total_tokens",
        "requested_total_tokens": requested,
        "tokens_per_update": 16,
        "steps": steps,
        "actual_total_tokens": actual,
    }
    assert not runs_root.exists()


@pytest.mark.parametrize("budget", [
    "", "steps = 4\ntotal_tokens = 64",
    "steps = 0", "steps = -1", "steps = true", "steps = 1.5",
    "total_tokens = 0", "total_tokens = -1", "total_tokens = true", "total_tokens = 1.5",
])
def test_invalid_budget(tmp_path, budget):
    data_root, runs_root, preset = make_project(tmp_path)
    preset.write_text(PRESET.replace("steps = 4", budget))
    with pytest.raises(ExperimentConfigError, match="exactly one|positive integer"):
        resolve_experiment(preset, data_root_override=str(data_root),
                           runs_root_override=str(runs_root), environment={}, root=tmp_path)


def test_local_config_resolves_without_creating_runs_directory(tmp_path: Path) -> None:
    data_root, runs_root, preset = make_project(tmp_path)
    local_config = tmp_path / "configs" / "local.toml"
    local_config.write_text(f'[data]\nroot = "{data_root}"\n\n[runs]\nroot = "{runs_root}"\n')

    resolved = resolve_experiment(
        preset,
        local_config_path=local_config,
        environment={},
        root=tmp_path,
    )

    assert resolved["resolved_paths"]["data_root"] == str(data_root)
    assert resolved["resolved_paths"]["runs_root"] == str(runs_root)
    assert resolved["provenance"]["data_root_source"] == "local_config"
    assert resolved["provenance"]["runs_root_source"] == "local_config"
    assert not runs_root.exists()


@pytest.mark.parametrize("mode", ["basic", "cosine_scheduled"])
def test_resolved_training_schedule(tmp_path, mode):
    from cs336_basics.training import learning_rate_at

    data_root, runs_root, preset = make_project(tmp_path)
    text = PRESET.replace("steps = 4", "total_tokens = 65")
    if mode == "cosine_scheduled":
        text = text.replace('lr_mode = "basic"', 'lr_mode = "cosine_scheduled"')
        text = text.replace("learning_rate = 0.001",
                            "warmup_steps = 1\nanneal_steps = 5\nlr_max = 0.001\nlr_min = 0.0001")
    preset.write_text(text)
    resolved = resolve_experiment(preset, data_root_override=str(data_root),
                                  runs_root_override=str(runs_root), environment={}, root=tmp_path)
    training = resolved["training"]
    assert training["steps"] == 5
    assert learning_rate_at(training, 1) == pytest.approx(0.001)
    assert learning_rate_at(training, 5) == pytest.approx(0.001 if mode == "basic" else 0.0001)

    assert resolved["training"]["betas"] == [0.9, 0.999]


def test_environment_overrides_local_config(tmp_path: Path) -> None:
    data_root, runs_root, preset = make_project(tmp_path)
    wrong_root = tmp_path / "wrong"
    local_config = tmp_path / "configs" / "local.toml"
    local_config.write_text(f'[data]\nroot = "{wrong_root}"\n\n[runs]\nroot = "{wrong_root}"\n')

    resolved = resolve_experiment(
        preset,
        local_config_path=local_config,
        environment={
            "CS336_DATA_ROOT": str(data_root),
            "CS336_RUNS_ROOT": str(runs_root),
        },
        root=tmp_path,
    )

    assert resolved["provenance"]["data_root_source"] == "environment:CS336_DATA_ROOT"
    assert resolved["provenance"]["runs_root_source"] == "environment:CS336_RUNS_ROOT"


def test_command_line_root_overrides_environment(tmp_path: Path) -> None:
    data_root, runs_root, preset = make_project(tmp_path)

    resolved = resolve_experiment(
        preset,
        data_root_override=str(data_root),
        runs_root_override=str(runs_root),
        environment={
            "CS336_DATA_ROOT": "/wrong/data",
            "CS336_RUNS_ROOT": "/wrong/runs",
        },
        root=tmp_path,
    )

    assert resolved["provenance"]["data_root_source"] == "command_line"
    assert resolved["provenance"]["runs_root_source"] == "command_line"


def test_dataset_paths_cannot_escape_data_root(tmp_path: Path) -> None:
    data_root, runs_root, preset = make_project(tmp_path)
    text = preset.read_text().replace('train_file = "tiny/train.npy"', 'train_file = "../outside.npy"')
    preset.write_text(text)

    with pytest.raises(ExperimentConfigError, match="escapes"):
        resolve_experiment(
            preset,
            data_root_override=str(data_root),
            runs_root_override=str(runs_root),
            environment={},
            root=tmp_path,
        )


def test_training_and_validation_files_must_be_distinct(tmp_path: Path) -> None:
    data_root, runs_root, preset = make_project(tmp_path)
    text = preset.read_text().replace('validation_file = "tiny/valid.npy"', 'validation_file = "tiny/train.npy"')
    preset.write_text(text)

    with pytest.raises(ExperimentConfigError, match="must be distinct"):
        resolve_experiment(
            preset,
            data_root_override=str(data_root),
            runs_root_override=str(runs_root),
            environment={},
            root=tmp_path,
        )

from __future__ import annotations

import copy
import hashlib
import os
import tomllib
import random
import re
import uuid
from datetime import UTC, datetime
import numpy as np
import torch
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cs336_basics.training import training_loop


SUPPORTED_SCHEMA_VERSION = 1
ROOT_ENVIRONMENT_VARIABLES = {
    "data": "CS336_DATA_ROOT",
    "runs": "CS336_RUNS_ROOT",
}


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedRoot:
    path: Path
    source: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            value = tomllib.load(file)
    except FileNotFoundError as error:
        raise ExperimentConfigError(f"Configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ExperimentConfigError(f"Invalid TOML in {path}: {error}") from error

    if not isinstance(value, dict):
        raise ExperimentConfigError(f"Expected a TOML table at the top level of {path}")
    return value


def _path_relative_to(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_table(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"Expected a [{name}] table")
    return value


def _require_string(table: Mapping[str, Any], key: str, table_name: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"Expected {table_name}.{key} to be a non-empty string")
    return value


def _require_positive_int(table: Mapping[str, Any], key: str, table_name: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExperimentConfigError(f"Expected {table_name}.{key} to be a positive integer")
    return value


def _require_positive_number(table: Mapping[str, Any], key: str, table_name: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ExperimentConfigError(f"Expected {table_name}.{key} to be positive")
    return float(value)


def _validate_preset(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ExperimentConfigError(
            f"Expected schema_version = {SUPPORTED_SCHEMA_VERSION}, got {config.get('schema_version')!r}"
        )

    experiment = _require_table(config, "experiment")
    data = _require_table(config, "data")
    tokenizer = _require_table(config, "tokenizer")
    model = _require_table(config, "model")
    training = _require_table(config, "training")
    evaluation = _require_table(config, "evaluation")
    logging = _require_table(config, "logging")
    checkpoint = _require_table(config, "checkpoint")
    runtime = _require_table(config, "runtime")
    tracking = _require_table(config, "tracking")

    _require_string(experiment, "name", "experiment")
    seed = experiment.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ExperimentConfigError("Expected experiment.seed to be a non-negative integer")
    tags = experiment.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
        raise ExperimentConfigError("Expected experiment.tags to be a list of non-empty strings")

    _require_string(data, "name", "data")
    _require_string(data, "version", "data")
    _require_string(data, "train_file", "data")
    _require_string(data, "validation_file", "data")
    _require_string(tokenizer, "vocab_file", "tokenizer")
    _require_string(tokenizer, "merges_file", "tokenizer")
    _require_string(tokenizer, "end_of_text", "tokenizer")

    for key in ("vocab_size", "context_length", "num_heads", "num_layers", "d_model", "d_ff"):
        _require_positive_int(model, key, "model")
    _require_positive_number(model, "rope_theta", "model")
    if model["d_model"] % model["num_heads"] != 0:
        raise ExperimentConfigError("model.d_model must be divisible by model.num_heads")
    if (model["d_model"] // model["num_heads"]) % 2 != 0:
        raise ExperimentConfigError("The attention head dimension must be even for RoPE")

    _require_positive_int(training, "batch_size", "training")
    if ("steps" in training) == ("total_tokens" in training):
        raise ExperimentConfigError("Specify exactly one of training.steps or training.total_tokens")
    budget_key = "steps" if "steps" in training else "total_tokens"
    _require_positive_int(training, budget_key, "training")
    _require_positive_number(training, "grad_clip", "training")
    _require_positive_number(training, "epsilon", "training")
    weight_decay = training.get("weight_decay")
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or weight_decay < 0:
        raise ExperimentConfigError("Expected training.weight_decay to be non-negative")
    betas = training.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or not all(
            isinstance(beta, (int, float)) and not isinstance(beta, bool) and 0 <= beta < 1
            for beta in betas
        )
    ):
        raise ExperimentConfigError("Expected training.betas to contain two numbers in [0, 1)")

    lr_mode = training.get("lr_mode")
    if lr_mode == "basic":
        _require_positive_number(training, "learning_rate", "training")
    elif lr_mode == "cosine_scheduled":
        for key in ("lr_max", "lr_min"):
            _require_positive_number(training, key, "training")
        for key in ("warmup_steps", "anneal_steps"):
            _require_positive_int(training, key, "training")
    else:
        raise ExperimentConfigError("training.lr_mode must be 'basic' or 'cosine_scheduled'")

    _require_positive_int(evaluation, "every_steps", "evaluation")
    _require_positive_int(evaluation, "batches", "evaluation")
    _require_positive_int(logging, "every_steps", "logging")
    _require_positive_int(checkpoint, "every_steps", "checkpoint")
    _require_positive_int(checkpoint, "keep_last", "checkpoint")

    if runtime.get("device") not in {"auto", "cpu", "mps", "cuda"}:
        raise ExperimentConfigError("runtime.device must be one of: auto, cpu, mps, cuda")
    if tracking.get("wandb_mode") not in {"disabled", "offline", "online"}:
        raise ExperimentConfigError("tracking.wandb_mode must be one of: disabled, offline, online")
    _require_string(tracking, "wandb_project", "tracking")


def _load_optional_local_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return _load_toml(path)


def _resolve_root(
    *,
    name: str,
    command_line_value: str | None,
    environment: Mapping[str, str],
    local_config: Mapping[str, Any],
) -> ResolvedRoot:
    environment_name = ROOT_ENVIRONMENT_VARIABLES[name]

    if command_line_value is not None:
        raw_value = command_line_value
        source = "command_line"
    elif environment.get(environment_name):
        raw_value = environment[environment_name]
        source = f"environment:{environment_name}"
    else:
        local_table = local_config.get(name)
        if not isinstance(local_table, dict) or not local_table.get("root"):
            raise ExperimentConfigError(
                f"Missing {name} root. Pass --{name}-root, set {environment_name}, "
                f"or define [{name}].root in configs/local.toml"
            )
        raw_value = local_table["root"]
        source = "local_config"

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ExperimentConfigError(f"The resolved {name} root must be a non-empty string")

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        raise ExperimentConfigError(f"The resolved {name} root must be absolute: {path}")
    path = path.resolve()

    if name == "data":
        if not path.is_dir():
            raise ExperimentConfigError(f"Data root is not an existing directory: {path}")
    elif path.exists() and not path.is_dir():
        raise ExperimentConfigError(f"Runs root exists but is not a directory: {path}")

    return ResolvedRoot(path=path, source=source)


def _resolve_relative_file(base: Path, raw_value: str, label: str) -> Path:
    relative_path = Path(raw_value)
    if relative_path.is_absolute():
        raise ExperimentConfigError(f"{label} must be relative, got: {raw_value}")

    base = base.resolve()
    path = (base / relative_path).resolve()
    if not path.is_relative_to(base):
        raise ExperimentConfigError(f"{label} escapes its configured root: {raw_value}")
    if not path.is_file():
        raise ExperimentConfigError(f"{label} does not exist: {path}")
    return path


def resolve_experiment(
    config_path: str | os.PathLike[str],
    *,
    local_config_path: str | os.PathLike[str] = "configs/local.toml",
    data_root_override: str | None = None,
    runs_root_override: str | None = None,
    device_override: str | None = None,
    environment: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = project_root() if root is None else root.resolve()
    config_path = _path_relative_to(root, config_path)
    local_config_path = _path_relative_to(root, local_config_path)
    environment = os.environ if environment is None else environment

    config = _load_toml(config_path)
    _validate_preset(config)
    local_config = _load_optional_local_config(local_config_path)

    data_root = _resolve_root(
        name="data",
        command_line_value=data_root_override,
        environment=environment,
        local_config=local_config,
    )
    runs_root = _resolve_root(
        name="runs",
        command_line_value=runs_root_override,
        environment=environment,
        local_config=local_config,
    )

    resolved = copy.deepcopy(config)
    training = resolved["training"]
    tokens_per_update = training["batch_size"] * resolved["model"]["context_length"]
    requested_tokens = training.get("total_tokens")
    if requested_tokens is not None:
        training["steps"] = (requested_tokens + tokens_per_update - 1) // tokens_per_update
    resolved["resolved_budget"] = {
        "source": "total_tokens" if requested_tokens is not None else "steps",
        "requested_total_tokens": requested_tokens,
        "tokens_per_update": tokens_per_update,
        "steps": training["steps"],
        "actual_total_tokens": training["steps"] * tokens_per_update,
    }
    runtime = resolved["runtime"]
    if device_override is not None:
        if device_override not in {"auto", "cpu", "mps", "cuda"}:
            raise ExperimentConfigError("--device must be one of: auto, cpu, mps, cuda")
        runtime["device"] = device_override

    data = resolved["data"]
    tokenizer = resolved["tokenizer"]
    train_path = _resolve_relative_file(data_root.path, data["train_file"], "data.train_file")
    validation_path = _resolve_relative_file(
        data_root.path,
        data["validation_file"],
        "data.validation_file",
    )
    if train_path == validation_path:
        raise ExperimentConfigError("Training and validation files must be distinct")

    tokenizer_vocab_path = _resolve_relative_file(root, tokenizer["vocab_file"], "tokenizer.vocab_file")
    tokenizer_merges_path = _resolve_relative_file(root, tokenizer["merges_file"], "tokenizer.merges_file")

    resolved["resolved_paths"] = {
        "project_root": str(root),
        "data_root": str(data_root.path),
        "runs_root": str(runs_root.path),
        "train_data": str(train_path),
        "validation_data": str(validation_path),
        "tokenizer_vocab": str(tokenizer_vocab_path),
        "tokenizer_merges": str(tokenizer_merges_path),
    }
    resolved["provenance"] = {
        "preset": str(config_path),
        "preset_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "local_config": str(local_config_path) if local_config_path.exists() else None,
        "data_root_source": data_root.source,
        "runs_root_source": runs_root.source,
        "device_source": "command_line" if device_override is not None else "preset",
    }
    resolved["input_metadata"] = {
        "train_data_size_bytes": train_path.stat().st_size,
        "validation_data_size_bytes": validation_path.stat().st_size,
        "tokenizer_vocab_size_bytes": tokenizer_vocab_path.stat().st_size,
        "tokenizer_merges_size_bytes": tokenizer_merges_path.stat().st_size,
    }
    return resolved

def run_experiment(resolved, *, resume_from=None, stop_after=None, verbose=True):
    """Execute a resolved preset and return its model, metrics, and output path."""
    # Copy so selecting "auto" does not mutate the caller's configuration.
    resolved = copy.deepcopy(resolved)
    device = resolved["runtime"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ExperimentConfigError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ExperimentConfigError("MPS was requested but is unavailable")
    resolved["runtime"]["device"] = device
    if resolved["tracking"]["wandb_mode"] != "disabled":
        raise ExperimentConfigError("W&B is not connected yet; use wandb_mode = 'disabled'")

    model_config = resolved["model"]
    training_config = resolved["training"]
    paths = resolved["resolved_paths"]

    # Memory mapping keeps large arrays on disk. Scan bounded chunks to reject
    # invalid IDs before allocating a model or creating an output directory.
    arrays = {}
    for label in ("train_data", "validation_data"):
        array = np.load(paths[label], mmap_mode="r", allow_pickle=False)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ExperimentConfigError(f"{label} must be a one-dimensional integer NPY array")
        if len(array) <= model_config["context_length"]:
            raise ExperimentConfigError(f"{label} must be longer than context_length")
        for offset in range(0, len(array), 1_000_000):
            chunk = array[offset:offset + 1_000_000]
            if chunk.min() < 0 or chunk.max() >= model_config["vocab_size"]:
                raise ExperimentConfigError(f"{label} contains token IDs outside the vocabulary")
        arrays[label] = array

    seed = resolved["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)

    # The loop creates this directory exclusively. Each invocation, including a
    # resume, gets a fresh attempt path; the checkpoint preserves logical run ID.
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", resolved["experiment"]["name"]).strip("-") or "experiment"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(paths["runs_root"]) / f"{timestamp}_{name}_{uuid.uuid4().hex[:12]}"
    result = training_loop(
        arrays['train_data'], model_config, training_config, checkpoint_out=output,
        device=device, checkpoint_every=resolved["checkpoint"]["every_steps"],
        keep_last=resolved["checkpoint"]["keep_last"],
        log_every=resolved["logging"]["every_steps"], resolved_config=resolved,
        validation_data=arrays['validation_data'], eval_every=resolved["evaluation"]["every_steps"],
        eval_batches=resolved["evaluation"]["batches"],
        resume_from=resume_from, stop_after=stop_after, verbose=verbose,
    )
    result["output_dir"] = output
    return result

import logging
from pathlib import Path
from types import SimpleNamespace

import modal
import wandb

from cs336_scaling.internal_client import (
    post_internal_finish,
    post_internal_worker_event,
)
from cs336_scaling.log.setup import configure_logging
from cs336_scaling.modal_utils import MODAL_SECRETS, VOLUME_MOUNTS, app, build_image
from cs336_scaling.tokenized_data import DCLMData
from cs336_scaling.training.driver import TrainingResult, run_training_driver
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def default_training_config() -> TrainingConfig:
    return TrainingConfig(
        architecture_config=BasicTransformerConfig(
            attention_bias=False,
            head_dim=64,
            hidden_size=448,
            intermediate_size=1280,
            num_attention_heads=7,
            num_hidden_layers=9,
            num_key_value_heads=7,
            rms_norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=False,
            dtype="bfloat16",
            vocab_size=32000,
        ),
        optimizer_config=AdamWConfig(),
        train_batch_size=128,
        val_batch_size=32,
        total_train_tokens=16 * 2**16,
        max_runtime_seconds=30,
        model_seed=0,
    )


@app.function(
    image=build_image(include_tailscale=True),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
    secrets=MODAL_SECRETS,
    gpu="B200",
    ephemeral_disk=1099511,
)
def run_training(
    training_config: TrainingConfig, sunet_id: str, experiment_id: int
) -> TrainingResult:
    run_id = modal.current_function_call_id()
    if run_id is None:
        raise ValueError(
            "modal function call id is unavailable outside a Modal function call"
        )
    runtime = SimpleNamespace(
        run_id=run_id,
        output_path=Path(f"output/{sunet_id}/{experiment_id}"),
        cache_path=Path("/tmp/jax_cache"),
        data=DCLMData(),
        local_data_dir=Path("/tmp/data"),
        metrics=wandb,
        report_event=post_internal_worker_event,
        report_finish=post_internal_finish,
        cancel=lambda: modal.FunctionCall.from_id(run_id).cancel(
            terminate_containers=True
        ),
        configure_logging=lambda path: configure_logging(
            log_file=path, log_format="json"
        ),
        training_started=lambda: None,
    )
    return run_training_driver(training_config, sunet_id, experiment_id, runtime)


@app.local_entrypoint()
def modal_main():
    training_config = default_training_config()
    print(
        "final result",
        run_training.remote(
            training_config,
            sunet_id="local",
            experiment_id=1,
        ),
    )


if __name__ == "__main__":
    run_training.local(default_training_config(), sunet_id="local", experiment_id=2)

"""One metrics destination: local JSONL, optional console, optional W&B."""

from contextlib import ExitStack, contextmanager
import sys

import torch

from cs336_basics.run_io import capture_rng, restore_rng, write_jsonl


@contextmanager
def metric_logger(output, config, *, verbose=True):
    """Yield a callback; the launcher owns opening and closing all log outputs."""
    with ExitStack() as stack:
        stream = stack.enter_context((output / "metrics.jsonl").open("x"))
        tracker = None

        def log(record):
            nonlocal tracker
            # Flush local evidence first, even if the optional tracking service fails.
            write_jsonl(stream, record)
            if verbose:
                print(" ".join(f"{key}={value}" for key, value in record.items() if key != "run_id"), file=sys.stderr)
            if config["tracking"]["wandb_mode"] == "disabled":
                return

            # Tracking must not change the random batches or a resumed trajectory.
            rng = capture_rng(torch.device(config["runtime"]["device"]))
            try:
                if tracker is None:
                    import wandb

                    # A directory identifies an execution; run_id groups continuations.
                    # Start on the first record, when the loop knows the resumed ID.
                    portable = {key: config[key] for key in (
                        "model", "training", "evaluation", "logging", "checkpoint", "runtime"
                    )}
                    tracker = stack.enter_context(wandb.init(
                        project=config["tracking"]["wandb_project"],
                        mode=config["tracking"]["wandb_mode"],
                        id=output.name, group=record["run_id"],
                        name=config["experiment"]["name"],
                        tags=config["experiment"].get("tags", []),
                        config={**portable, "seed": config["experiment"]["seed"]},
                        dir=str(output),
                        # No automatic source, Git metadata, or console uploads.
                        settings=wandb.Settings(disable_code=True, disable_git=True, console="off"),
                    ))
                    tracker.define_metric("step")
                    tracker.define_metric("*", step_metric="step")
                tracker.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])
            finally:
                restore_rng(rng)

        yield log

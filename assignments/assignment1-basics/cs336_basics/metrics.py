"""One metrics destination: local JSONL, optional console, optional W&B."""

from contextlib import ExitStack, contextmanager
import sys

import torch

from cs336_basics.run_io import capture_rng, restore_rng, write_jsonl


def format_metrics(record, total_steps=None):
    """Round only the console view; stored measurements retain full precision."""
    step = f"{record['step']:,}"
    if total_steps is not None:
        step += f"/{total_steps:,}"
    validation = f"{record['validation_loss']:.4f}" if "validation_loss" in record else "—"
    rate = f"{record['train_tokens_per_second']:,.0f}" if "train_tokens_per_second" in record else "—"
    tflops = f"{record['estimated_train_tflops']:.3g}" if "estimated_train_tflops" in record else "—"
    seconds = int(record.get("elapsed_seconds", 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"step {step:>11} | loss {record['train_loss']:7.4f} | val {validation:>7} | "
        f"lr {record.get('learning_rate', 0):.2e} | tok/s {rate:>9} | "
        f"est. TFLOP/s {tflops:>7} | {hours:02}:{minutes:02}:{seconds:02}"
    )


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
                print(format_metrics(record, config["training"]["steps"]), file=sys.stderr, flush=True)
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
                        config={
                            **portable, "seed": config["experiment"]["seed"],
                            **{key: config.get("provenance", {}).get(key)
                               for key in ("git_commit", "git_dirty", "git_error")},
                        },
                        dir=str(output),
                        # Only our explicit Git fields are sent; no source/diff uploads.
                        settings=wandb.Settings(disable_code=True, disable_git=True, console="off"),
                    ))
                    tracker.define_metric("step")
                    tracker.define_metric("*", step_metric="step")
                tracker.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])
            finally:
                restore_rng(rng)

        yield log

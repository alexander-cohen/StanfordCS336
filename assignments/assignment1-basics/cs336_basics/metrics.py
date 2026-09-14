"""One metrics destination: local JSONL, optional console, optional W&B."""

from contextlib import ExitStack, contextmanager
import sys

import torch
import psutil

from cs336_basics.run_io import capture_rng, restore_rng, write_jsonl


def memory_metrics(device):
    """Sample capacity usage without synchronizing or resetting allocator statistics.

    CUDA peaks/counters cover this process since startup (or an external reset).
    Device free memory includes other processes. RSS is this process, not a Slurm job total.
    """
    device = torch.device(device)
    record = {"memory_process_rss_bytes": psutil.Process().memory_info().rss}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        record.update({
            "memory_cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "memory_cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            "memory_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "memory_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "memory_cuda_free_bytes": free,
            "memory_cuda_total_bytes": total,
        })
        stats = torch.cuda.memory_stats(device)
        for name in ("num_alloc_retries", "num_ooms"):
            if name in stats:
                record[f"memory_cuda_{name}"] = stats[name]
    elif device.type == "mps":
        record.update({
            "memory_mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "memory_mps_driver_bytes": torch.mps.driver_allocated_memory(),
            "memory_mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
        })
    return record


def format_metrics(record, total_steps=None):
    """Round only the console view; stored measurements retain full precision."""
    step = f"{record['step']:,}"
    if total_steps is not None:
        step += f"/{total_steps:,}"
    validation = f"{record['validation_loss']:.4f}" if "validation_loss" in record else "—"
    rate = f"{record['train_tokens_per_second']:,.0f}" if "train_tokens_per_second" in record else "—"
    flops_per_second = f"{record['estimated_train_flops_per_second']:.3g}" if "estimated_train_flops_per_second" in record else "—"
    seconds = int(record.get("elapsed_seconds", 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    memory = ""
    gib = 1024**3
    if "memory_process_rss_bytes" in record:
        memory += f" | RAM RSS {record['memory_process_rss_bytes'] / gib:.2f} GiB"
    if "memory_cuda_allocated_bytes" in record:
        memory += (
            f" | GPU alloc {record['memory_cuda_allocated_bytes'] / gib:.2f}"
            f" peak {record['memory_cuda_peak_allocated_bytes'] / gib:.2f}"
            f" reserved {record['memory_cuda_reserved_bytes'] / gib:.2f}"
            f" free {record['memory_cuda_free_bytes'] / gib:.2f}"
            f" / total {record['memory_cuda_total_bytes'] / gib:.2f} GiB"
        )
        if "memory_cuda_num_alloc_retries" in record:
            memory += f" | alloc retries {record['memory_cuda_num_alloc_retries']}"
    elif "memory_mps_allocated_bytes" in record:
        memory += (
            f" | MPS alloc {record['memory_mps_allocated_bytes'] / gib:.2f}"
            f" driver {record['memory_mps_driver_bytes'] / gib:.2f}"
            f" recommended max {record['memory_mps_recommended_max_bytes'] / gib:.2f} GiB"
        )
    return (
        f"step {step:>11} | loss {record['train_loss']:7.4f} | val {validation:>7} | "
        f"lr {record.get('learning_rate', 0):.2e} | tok/s {rate:>9} | "
        f"est. FLOP/s {flops_per_second:>7} | {hours:02}:{minutes:02}:{seconds:02}{memory}"
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
            if config["tracking"]["wandb_mode"] == "disabled":
                if verbose:
                    print(format_metrics(record, config["training"]["steps"]), file=sys.stderr, flush=True)
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
                            "run_folder": output.name,
                            "logical_run_id": record["run_id"],
                            **{key: config.get("provenance", {}).get(key)
                               for key in ("git_commit", "git_dirty", "git_error")},
                        },
                        dir=str(output),
                        # Only our explicit Git fields are sent; no source/diff uploads.
                        settings=wandb.Settings(disable_code=True, disable_git=True, console="auto"),
                    ))
                    tracker.define_metric("step")
                    tracker.define_metric("*", step_metric="step")
                    if verbose:
                        print(
                            f"W&B run ID: {output.name}\n"
                            f"Logical run ID: {record['run_id']}\n"
                            f"Run folder: {output.resolve()}",
                            file=sys.stderr, flush=True,
                        )
                # Initialize console capture before printing the first metric.
                if verbose:
                    print(format_metrics(record, config["training"]["steps"]), file=sys.stderr, flush=True)
                tracker.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])
            finally:
                restore_rng(rng)

        yield log

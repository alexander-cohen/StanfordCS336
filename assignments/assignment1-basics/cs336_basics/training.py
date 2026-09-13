import math
from typing import Dict
# from matplotlib.font_manager import weight_dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
from cs336_basics import nn_utils
from cs336_basics.nn_utils import softmax
from collections.abc import Callable, Iterable
from typing import Literal, Optional
# import matplotlib.pyplot as plt
import numpy as np
import cs336_basics.transformer as transformer
import cs336_basics.decode as decode
from tests.conftest import vocab_size
import hashlib
import uuid
from cs336_basics.run_io import atomic_save, capture_rng, restore_rng, resume_settings
from cs336_basics.metrics import format_metrics, memory_metrics
import sys
import textwrap
import pprint
import random
import time
from pathlib import Path
import os

# Bump this when the checkpoint structure changes incompatibly, not for new runs.
CHECKPOINT_FORMAT_VERSION = 1


def old_cross_entropy(pred_logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]):

    softmaxd = softmax(pred_logits, dim=-1)
    neglogprob = -torch.log(softmaxd[targets])
    return torch.mean(neglogprob)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                return loss


def learning_rate_at(training, step):
    """Read the validated schedule settings directly from the training section."""
    if training["lr_mode"] == "basic":
        return training["learning_rate"]
    if training["lr_mode"] != "cosine_scheduled":
        raise ValueError("Unknown learning-rate mode")
    return lr_cosine_schedule(
        step, training["lr_max"], training["lr_min"],
        training["warmup_steps"], training["anneal_steps"],
    )


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps = 1e-8, weight_decay = 0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr" : lr, 
                    "beta1" : betas[0], 
                    "beta2" : betas[1], 
                    "ep" : eps, 
                    "weight_decay" : weight_decay}

        super().__init__(params, defaults)

        for p in self.state.keys():
            self.state[p]['m'] = torch.zeros(p.shape)
            self.state[p]['v'] = torch.zeros(p.shape)

    def set_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            ep = group['ep']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.

                t = state.get("t", 1) # Get iteration number from the state, or 0.
                state["t"] = t + 1 # Increment iteration number.

                cur_rate = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # apply weight decay
                p.data = (1 - lr * weight_decay) * p.data

                # update  moments
                if 'm' not in state:
                    state['m'] = torch.zeros(p.shape).to(device = p.device)
                if 'v' not in state:
                    state['v'] = torch.zeros(p.shape).to(device = p.device)


                state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * p.grad**2
                p.data = p.data - cur_rate * state['m'] / (torch.sqrt(state['v']) + ep)
                
def lr_cosine_schedule(t, lr_max, lr_min, T_warmup, T_final_anneal):
    if t < T_warmup:
        return (t / T_warmup) * lr_max
    elif t <= T_final_anneal:
        t_an = t - T_warmup
        full_time = T_final_anneal - T_warmup
        return lr_min + (1/2) * (1 + math.cos( math.pi * t_an / full_time )) * (lr_max - lr_min)
    else:
        return lr_min


# takes in numpy array of token IDs, batch_size, context_length, and a pytorch device string
# outputs an iterable of batches, each of which is a tuple of input tokens and next-token targets
def get_batch(x, batch_size, context_length, device=torch.device('cpu')):
    xsize = x.shape[0]

    inds = torch.tensor(random.choices(range(xsize - context_length), k=batch_size))

    idx = inds.view(-1, 1)  # Reshape to column vector: [[2], [7], [4]]

    # 3. Create the 2D index matrix using broadcasting
    # This adds j offsets (0, 1, 2) to each element in idx
    j_offsets = torch.arange(context_length).view(1, -1)  # Row vector: [[0, 1, 2]]
    in_indices = idx + j_offsets 
    out_indices = idx + j_offsets +1 

    # 4. Gather the elements from x_tensor
    in_tens = torch.tensor(x[in_indices], device=device, dtype=torch.long)
    out_tens =  torch.tensor(x[out_indices], device=device, dtype=torch.long)

    return in_tens, out_tens

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
    other_dat=None,
):
    if other_dat is None:
        other_dat = {}

    if {"model", "optimizer", "iteration"} & other_dat.keys():
        raise ValueError("Checkpoint metadata cannot override model, optimizer, or iteration")
    out_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        **other_dat,
    }

    if isinstance(out, (str, os.PathLike)):
        out = Path(out)
        atomic_save(out_dict, out)
    else:
        torch.save(out_dict, out)

def load_checkpoint(src, # pathlike
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer):
        saved = torch.load(src, map_location="cpu", weights_only=True)
        iter = saved['iteration']
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        return iter

def grad_norm(parameters):
    grad_norm = 0
    for p in parameters:
        grad_norm += torch.linalg.norm(p.grad)**2
    grad_norm = math.sqrt(grad_norm)
    return grad_norm


def training_loop(
    x, model_params: dict, training_params: dict,
    checkpoint_out=None, verbose=True, tok=None, prompt="",
    *, device="cpu", checkpoint_every=100, log_every=1,
    resume_from=None, stop_after=None, resolved_config=None,
    validation_data=None, eval_every=100, eval_batches=2, keep_last=None,
    log_metrics: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
):
    """Train, evaluate, save snapshots, and emit dictionaries to log_metrics.

    checkpoint_out is a NEW attempt directory. On resume, use a new directory
    so metrics after an older checkpoint never conflict with an existing history.
    stop_after is an absolute completed-step boundary for interruption tests;
    it does not change the planned budget or learning-rate schedule.
    The launcher owns logging; direct calls without a callback print metrics only.
    """
    if log_every <= 0 or checkpoint_every <= 0 or training_params["steps"] <= 0:
        raise ValueError("Logging cadence, checkpoint cadence, and training steps must be positive")
    if keep_last is not None and (type(keep_last) is not int or keep_last <= 0):
        raise ValueError("keep_last must be a positive integer or None")
    if len(x) <= model_params["context_length"]:
        raise ValueError("Training data must be longer than context_length")
    if validation_data is not None and (
        len(validation_data) <= model_params["context_length"] or eval_every <= 0 or eval_batches <= 0
    ):
        raise ValueError("Validation requires sufficient data and positive cadence/batch count")
    end_step = training_params["steps"] if stop_after is None else stop_after
    if not 0 < end_step <= training_params["steps"]:
        raise ValueError("stop_after must lie within the planned training budget")

    device = torch.device(device)
    # Store plain configuration values, not pickled dataclass instances.
    # Save the authoritative resolved dictionary directly. Standalone tests can
    # supply just the model/training sections and an explicit device.
    configuration = resolved_config if resolved_config is not None else {
        "model": model_params, "training": training_params,
        "runtime": {"device": str(device)},
    }
    # Include the actual token bytes: equal shapes alone do not identify a dataset.
    data_hash = hashlib.sha256()
    for offset in range(0, len(x), 1_000_000):
        data_hash.update(np.ascontiguousarray(x[offset:offset + 1_000_000]).tobytes())
    data_identity = {"sha256": data_hash.hexdigest(), "shape": list(x.shape), "dtype": str(x.dtype)}
    if validation_data is not None:
        validation_hash = hashlib.sha256()
        for offset in range(0, len(validation_data), 1_000_000):
            validation_hash.update(np.ascontiguousarray(validation_data[offset:offset + 1_000_000]).tobytes())
        data_identity["validation"] = {
            "sha256": validation_hash.hexdigest(), "shape": list(validation_data.shape),
            "dtype": str(validation_data.dtype),
        }

    # Keep the existing Transformer constructor (theta) compatible with adapters.
    model_args = dict(model_params)
    model_args["theta"] = model_args.pop("rope_theta", 10000.0)
    model = transformer.Transformer(**model_args).to(device)
    # Rough forward+backward estimate: 6N FLOPs per training token, with N the
    # total parameter count. Not a hardware measurement; omits attention-matrix
    # costs and excludes validation/sampling. Compute N only once per launch.
    flops_per_token = 6 * sum(parameter.numel() for parameter in model.parameters())
    opt = AdamW(
        model.parameters(), lr=learning_rate_at(training_params, 0),
        betas=tuple(training_params.get("betas", (0.9, 0.999))),
        eps=training_params.get("epsilon", 1e-8),
        weight_decay=training_params.get("weight_decay", 0.0),
    )
    start_step, tokens_seen, elapsed_before = 0, 0, 0.0
    # Identifies the logical experiment across launches: new for fresh training,
    # but inherited from the checkpoint when continuing an existing run.
    run_id = uuid.uuid4().hex
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version: {saved.get('format_version')!r}; "
                f"expected {CHECKPOINT_FORMAT_VERSION} (None means the version is missing)"
            )
        # Compare settings directly: moving data or changing logging is harmless,
        # but changing the model, optimizer, or dataset is not an exact resume.
        if resume_settings(saved["configuration"]) != resume_settings(configuration) or saved["data_identity"] != data_identity:
            raise ValueError("Resume configuration or training data differs from the checkpoint")
        start_step = saved["iteration"]
        if not start_step < end_step:
            raise ValueError("Resume checkpoint must precede the requested stopping step")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        tokens_seen, elapsed_before = saved["tokens_seen"], saved["elapsed_seconds"]
        run_id = saved["run_id"]  # Link this continuation to the original run.
        # Model construction consumes randomness. Restore AFTER constructing/loading.
        restore_rng(saved["rng"])

    # The launcher owns the output directory when it supplies a metrics callback.
    # Direct calls still create a fresh checkpoint directory, without log machinery.
    output = Path(checkpoint_out) if checkpoint_out is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=log_metrics is not None)
    started = time.perf_counter()
    interval_seconds = 0.0
    interval_tokens = 0
    last_metric = None
    saved_paths = []  # Retention applies only to checkpoints from this execution.
    model.train()
    for step in range(start_step + 1, end_step + 1):
        # Drain earlier accelerator work before timing this training update.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        step_started = time.perf_counter()
        inputs, targets = get_batch(x, training_params["batch_size"], model_params["context_length"], device)
        opt.zero_grad()
        loss = nn_utils.cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        before_clip = grad_norm(model.parameters())
        if not math.isfinite(before_clip):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        nn_utils.gradient_clipping(model.parameters(), training_params["grad_clip"])
        after_clip = grad_norm(model.parameters())
        lr = learning_rate_at(training_params, step)
        opt.set_lr(lr)
        opt.step()
        tokens_seen += training_params["batch_size"] * model_params["context_length"]
        # Accelerator work is asynchronous; synchronize for meaningful timings.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        step_seconds = time.perf_counter() - step_started
        # Skip the first update of each launch (including resumes) as warmup.
        # Sum training time only: batching, forward/backward, and optimizer work.
        # Validation, sampling, checkpointing, and logging are outside this timer.
        if step > start_step + 1:
            interval_seconds += step_seconds
            interval_tokens += training_params["batch_size"] * model_params["context_length"]
        elapsed = elapsed_before + time.perf_counter() - started
        last_metric = {
            "run_id": run_id, "step": step,
            "tokens_seen": tokens_seen, "train_loss": loss.item(), "learning_rate": lr,
            # tokens_seen is restored on resume, so cumulative compute continues.
            "estimated_train_flops": flops_per_token * tokens_seen,
            "grad_norm": before_clip, "clipped_grad_norm": after_clip,
            "elapsed_seconds": elapsed,
        }
        # Poll only after a complete optimizer update, never save partial updates.
        # A request arriving during evaluation/logging is handled at the next
        # update boundary. Freeze this decision so evaluation and saving agree.
        stopping = should_stop is not None and should_stop()
        final_step = step == end_step or stopping
        evaluate = validation_data is not None and (step % eval_every == 0 or final_step)
        if evaluate:
            # Use repeatable validation batches, without advancing the
            # Python/PyTorch generators that determine future training.
            rng = capture_rng(device)
            model.eval()
            try:
                random.seed(0)
                validation_loss = 0.0
                with torch.no_grad():
                    for _ in range(eval_batches):
                        val_inputs, val_targets = get_batch(
                            validation_data, training_params["batch_size"], model_params["context_length"], device)
                        validation_loss += nn_utils.cross_entropy(model(val_inputs), val_targets).item()
                last_metric["validation_loss"] = validation_loss / eval_batches
                if not math.isfinite(last_metric["validation_loss"]):
                    raise FloatingPointError(f"Non-finite validation loss at step {step}")
            finally:
                model.train()
                restore_rng(rng)
            # Include evaluation time in this record and saved elapsed time.
            elapsed = elapsed_before + time.perf_counter() - started
            last_metric["elapsed_seconds"] = elapsed
        if step % log_every == 0 or final_step or evaluate:
            last_metric.update(memory_metrics(device))
            if interval_seconds > 0:
                # Interval averages, not cumulative rates. Same approximate 6N
                # FLOPs/token as estimated_train_flops, expressed as FLOP/s.
                last_metric["train_tokens_per_second"] = interval_tokens / interval_seconds
                last_metric["estimated_train_flops_per_second"] = flops_per_token * interval_tokens / interval_seconds
            if log_metrics is not None:
                log_metrics(last_metric)
            elif verbose:
                print(format_metrics(last_metric, training_params["steps"]), file=sys.stderr, flush=True)
            interval_seconds = 0.0
            interval_tokens = 0

        if step % checkpoint_every == 0 or final_step:
            if output is not None:
                # History lives in JSONL; only continuation state goes here.
                destination = output / f"{step}.pt"
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
                save_checkpoint(model, opt, step, destination, other_dat={
                    "format_version": CHECKPOINT_FORMAT_VERSION, "run_id": run_id,
                    "tokens_seen": tokens_seen, "elapsed_seconds": elapsed,
                    "configuration": configuration,
                    "data_identity": data_identity, "rng": capture_rng(device),
                })
                saved_paths.append(destination)
                # Delete old checkpoints only AFTER the replacement saves.
                # The parent checkpoint of a resumed attempt is untouched.
                if keep_last is not None:
                    while len(saved_paths) > keep_last:
                        expired = saved_paths[0]
                        expired.unlink()
                        saved_paths.pop(0)
            if tok is not None and not stopping:
                # Sampling must not change subsequent training randomness.
                rng = capture_rng(device)
                model.eval()
                try:
                    with torch.no_grad():
                        sample = tok.decode(decode.decode(
                            model, tok, prompt or "My good friend Agustin said:", 20, temperature=0.3))
                    if verbose:
                        print(f"\n  Sample (step {step:,}):\n" + textwrap.fill(
                            sample, width=100, initial_indent="    ", subsequent_indent="    "
                        ) + "\n", file=sys.stderr, flush=True)
                finally:
                    model.train()
                    restore_rng(rng)
        if stopping:
            break
    return {"model": model, "optimizer": opt, "last_metric": last_metric, "run_id": run_id}




def main():
    runs = {}
    lrs = np.logspace(-3, 3, num=20)
    for lr in lrs:
        print(f"\n\nlr: {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            losses.append(loss.item())
            print(loss.cpu().item())
            loss.backward()
            opt.step()
        runs[lr] = losses

    fig, ax = plt.subplots()
    for lr, losses in runs.items():
        ax.plot(losses, label=f"{lr}")
    ax.set_ylim(-0.2, 20)
    ax.legend()
    plt.savefig("mse training runs")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(lrs, [runs[j][-1] for j in runs])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("learning rate vs final loss")
    plt.clf()




if __name__ == "__main__":
    pass

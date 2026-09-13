import math
from typing import Dict
from matplotlib.font_manager import weight_dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
from cs336_basics import nn_utils
from cs336_basics.nn_utils import softmax
from collections.abc import Callable, Iterable
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import cs336_basics.transformer as transformer
import cs336_basics.decode as decode
from tests.conftest import vocab_size
from contextlib import ExitStack
import hashlib
import json
import logging
import uuid
from cs336_basics.run_io import atomic_save, atomic_json, capture_rng, restore_rng, write_event, resume_settings
import pprint
import random
import time
from pathlib import Path
import os


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

    idx = torch.tensor(inds).view(-1, 1)  # Reshape to column vector: [[2], [7], [4]]

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


## old version written by me was replaced by codex version for better logging
def training_loop(
    x, model_params: dict, training_params: dict,
    checkpoint_out=None, verbose=True, tok=None, prompt="",
    *, device="cpu", checkpoint_every=100, log_every=1,
    resume_from=None, stop_after=None, resolved_config=None,
    validation_data=None, eval_every=100, eval_batches=2, keep_last=None,
):
    """Train with scalar logs and resumable snapshots.

    checkpoint_out is a NEW attempt directory. On resume, use a new directory
    so metrics after an older checkpoint never conflict with an existing history.
    stop_after is an absolute completed-step boundary for interruption tests;
    it does not change the planned budget or learning-rate schedule.
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
    config_json = json.dumps(configuration, sort_keys=True, default=str)
    compatible_json = json.dumps(resume_settings(configuration), sort_keys=True, default=str)
    config_hash = hashlib.sha256(compatible_json.encode()).hexdigest()
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
    opt = AdamW(
        model.parameters(), lr=learning_rate_at(training_params, 0),
        betas=tuple(training_params.get("betas", (0.9, 0.999))),
        eps=training_params.get("epsilon", 1e-8),
        weight_decay=training_params.get("weight_decay", 0.0),
    )
    start_step, tokens_seen, elapsed_before = 0, 0, 0.0
    run_id = uuid.uuid4().hex
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("This checkpoint lacks the state needed for training resume")
        # Compare explicit settings, including for older format-1 checkpoints
        # whose hash also included machine paths and the raw preset file hash.
        if resume_settings(saved["configuration"]) != resume_settings(configuration) or saved["data_identity"] != data_identity:
            raise ValueError("Resume configuration or training data differs from the checkpoint")
        start_step = saved["iteration"]
        if not start_step < end_step:
            raise ValueError("Resume checkpoint must precede the requested stopping step")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        tokens_seen, elapsed_before = saved["tokens_seen"], saved["elapsed_seconds"]
        run_id = saved["run_id"]
        # Model construction consumes randomness. Restore AFTER constructing/loading.
        restore_rng(saved["rng"])

    attempt_id = uuid.uuid4().hex
    logger = logging.getLogger(f"cs336.training.{attempt_id}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    with ExitStack() as stack:
        if verbose:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            stack.callback(handler.close)
            stack.callback(logger.removeHandler, handler)
        metrics = events = None
        output = None
        if checkpoint_out is not None:
            output = Path(checkpoint_out)
            # Never silently overwrite an existing attempt or its evidence.
            output.mkdir(parents=True, exist_ok=False)
            (output / "config.resolved.json").write_text(config_json + "\n")
            metrics = stack.enter_context((output / "metrics.jsonl").open("x"))
            events = stack.enter_context((output / "events.jsonl").open("x"))

        def event(kind, **fields):
            if events is not None:
                write_event(events, {"event": kind, "run_id": run_id, "attempt_id": attempt_id, **fields})

        event("resume" if resume_from is not None else "start", step=start_step,
              checkpoint=str(resume_from) if resume_from is not None else None)
        started = time.perf_counter()
        last_metric = None
        saved_paths = []  # Only checkpoints created by THIS invocation can expire.
        try:
            model.train()
            for step in range(start_step + 1, end_step + 1):
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
                elapsed = elapsed_before + time.perf_counter() - started
                last_metric = {
                    "run_id": run_id, "attempt_id": attempt_id, "step": step,
                    "tokens_seen": tokens_seen, "train_loss": loss.item(), "learning_rate": lr,
                    "grad_norm": before_clip, "clipped_grad_norm": after_clip,
                    "elapsed_seconds": elapsed,
                }
                evaluate = validation_data is not None and (step % eval_every == 0 or step == end_step)
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
                if step % log_every == 0 or step == end_step or evaluate:
                    if metrics is not None:
                        write_event(metrics, last_metric)
                    logger.info("step=%d tokens=%d loss=%.6f lr=%g", step, tokens_seen, loss.item(), lr)
                    if evaluate:
                        logger.info("step=%d validation_loss=%.6f", step, last_metric["validation_loss"])

                if step % checkpoint_every == 0 or step == end_step:
                    if output is not None:
                        # History lives in JSONL; only continuation state goes here.
                        destination = output / f"{step}.pt"
                        save_checkpoint(model, opt, step, destination, other_dat={
                            "format_version": 1, "run_id": run_id, "attempt_id": attempt_id,
                            "tokens_seen": tokens_seen, "elapsed_seconds": elapsed,
                            "config_hash": config_hash, "configuration": configuration,
                            "data_identity": data_identity, "rng": capture_rng(device),
                        })
                        event("checkpoint", step=step, path=str(destination))
                        logger.info("Saved checkpoint %s", destination)
                        saved_paths.append(destination)
                        # Delete old checkpoints only AFTER the replacement saves.
                        # The parent checkpoint of a resumed attempt is untouched.
                        if keep_last is not None:
                            while len(saved_paths) > keep_last:
                                expired = saved_paths[0]
                                expired.unlink()
                                saved_paths.pop(0)
                                event("checkpoint_deleted", step=step, path=str(expired))
                    if tok is not None:
                        # Sampling must not change subsequent training randomness.
                        rng = capture_rng(device)
                        model.eval()
                        try:
                            with torch.no_grad():
                                sample = tok.decode(decode.decode(
                                    model, tok, prompt or "My good friend Agustin said:", 20, temperature=0.3))
                            event("sample", step=step, text=sample)
                            logger.info("Sample: %s", sample)
                        finally:
                            model.train()
                            restore_rng(rng)
            status = "complete" if end_step == training_params["steps"] else "paused"
            if output is not None:
                atomic_json({
                    "run_id": run_id, "attempt_id": attempt_id, "output_dir": str(output),
                    "status": status, "last_metric": last_metric,
                }, output / "summary.json")
            # Summary publication is part of success, so a failure above is logged
            # as an error rather than followed by a misleading completion event.
            event(status, step=end_step)
        except BaseException as error:
            event("error", error_type=type(error).__name__, message=str(error))
            raise
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

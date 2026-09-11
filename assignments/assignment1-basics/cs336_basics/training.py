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
from cs336_basics.transformer import ModelConfig
import cs336_basics.decode as decode
from tests.conftest import vocab_size
from dataclasses import dataclass, asdict
from contextlib import ExitStack
import hashlib
import json
import logging
import uuid
from cs336_basics.run_io import atomic_save, capture_rng, restore_rng, write_event
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


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    training_steps: int
    save_dist: int
    grad_clip: float
    device: torch.device
    lr_mode: Literal["basic", "cosine_scheduled"]
    eps: float = 1e-8
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    lr: float | None = None
    time_warmup: int | None = None
    time_final_anneal: int | None = None
    lr_max: float | None = None
    lr_min: float | None = None

    def __post_init__(self):
        if self.lr_mode == "basic":
            if self.lr is None:
                raise ValueError("lr is required when lr_mode='basic'")
            return

        schedule_values = (self.time_warmup, self.time_final_anneal, self.lr_max, self.lr_min)
        if any(value is None for value in schedule_values):
            raise ValueError(
                "time_warmup, time_final_anneal, lr_max, and lr_min are required "
                "when lr_mode='cosine_scheduled'"
            )

    def learning_rate_at(self, step: int) -> float:
        if self.lr_mode == "basic":
            if self.lr is None:
                raise RuntimeError("basic learning-rate configuration was not validated")
            return self.lr

        if (
            self.time_warmup is None
            or self.time_final_anneal is None
            or self.lr_max is None
            or self.lr_min is None
        ):
            raise RuntimeError("cosine learning-rate configuration was not validated")
        return lr_cosine_schedule(
            step,
            self.lr_max,
            self.lr_min,
            self.time_warmup,
            self.time_final_anneal,
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

    @classmethod
    def from_training_params(cls, params, training_params: TrainingConfig):
        return cls(params, 
                   betas=training_params.betas,
                    eps=training_params.eps,
                    lr=training_params.learning_rate_at(0),
                    weight_decay=training_params.weight_decay)

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
    x, model_params: ModelConfig, training_params: TrainingConfig,
    checkpoint_out=None, verbose=True, tok=None, prompt="",
    *, log_every=1, resume_from=None, stop_after=None, resolved_config=None,
):
    """Train with scalar logs and resumable snapshots.

    checkpoint_out is a NEW attempt directory. On resume, use a new directory
    so metrics after an older checkpoint never conflict with an existing history.
    stop_after is an absolute completed-step boundary for interruption tests;
    it does not change the planned budget or learning-rate schedule.
    """
    if log_every <= 0 or training_params.save_dist <= 0 or training_params.training_steps <= 0:
        raise ValueError("Logging cadence, checkpoint cadence, and training steps must be positive")
    if len(x) <= model_params.context_length:
        raise ValueError("Training data must be longer than context_length")
    end_step = training_params.training_steps if stop_after is None else stop_after
    if not 0 < end_step <= training_params.training_steps:
        raise ValueError("stop_after must lie within the planned training budget")

    device = torch.device(training_params.device)
    # Store plain configuration values, not pickled dataclass instances.
    configuration = {
        "model": asdict(model_params),
        "training": {**asdict(training_params), "device": str(device)},
        "experiment": resolved_config,
    }
    config_json = json.dumps(configuration, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    # Include the actual token bytes: equal shapes alone do not identify a dataset.
    data_hash = hashlib.sha256()
    for offset in range(0, len(x), 1_000_000):
        data_hash.update(np.ascontiguousarray(x[offset:offset + 1_000_000]).tobytes())
    data_identity = {"sha256": data_hash.hexdigest(), "shape": list(x.shape), "dtype": str(x.dtype)}

    model = transformer.Transformer.from_model_params(model_params).to(device)
    opt = AdamW.from_training_params(model.parameters(), training_params)
    start_step, tokens_seen, elapsed_before = 0, 0, 0.0
    run_id = uuid.uuid4().hex
    if resume_from is not None:
        saved = torch.load(resume_from, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("This checkpoint lacks the state needed for training resume")
        if saved["config_hash"] != config_hash or saved["data_identity"] != data_identity:
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
        try:
            model.train()
            for step in range(start_step + 1, end_step + 1):
                inputs, targets = get_batch(x, training_params.batch_size, model_params.context_length, device)
                opt.zero_grad()
                loss = nn_utils.cross_entropy(model(inputs), targets)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss.backward()
                before_clip = grad_norm(model.parameters())
                if not math.isfinite(before_clip):
                    raise FloatingPointError(f"Non-finite gradient norm at step {step}")
                nn_utils.gradient_clipping(model.parameters(), training_params.grad_clip)
                after_clip = grad_norm(model.parameters())
                lr = training_params.learning_rate_at(step)
                opt.set_lr(lr)
                opt.step()
                tokens_seen += training_params.batch_size * model_params.context_length
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
                if step % log_every == 0 or step == end_step:
                    if metrics is not None:
                        write_event(metrics, last_metric)
                    logger.info("step=%d tokens=%d loss=%.6f lr=%g", step, tokens_seen, loss.item(), lr)

                if step % training_params.save_dist == 0 or step == end_step:
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
            event("complete" if end_step == training_params.training_steps else "paused", step=end_step)
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

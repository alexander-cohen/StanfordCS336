"""Small disk and RNG helpers for training recovery."""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def atomic_save(payload, destination):
    # Write beside the destination so replacement stays on the same filesystem.
    # If serialization fails, readers still see the previous complete checkpoint.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_rng(device):
    # Python controls our batch sampling; PyTorch controls initialization/sampling.
    # NumPy is included so future data operations can also resume consistently.
    # Convert NumPy's array to primitives compatible with weights_only loading.
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def atomic_json(payload, destination):
    """Publish complete JSON before reporting a run as finished."""
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".summary-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resume_settings(configuration):
    # Location/provenance, logging and checkpoint cadence do not alter updates.
    # Keep device restrictions: exact continuation across backends is unverified.
    return {key: configuration[key] for key in ("model", "training", "runtime")}


def restore_rng(state):
    # Checkpoints are loaded on CPU; accelerator RNG APIs receive CPU byte tensors.
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def write_event(stream, event):
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()

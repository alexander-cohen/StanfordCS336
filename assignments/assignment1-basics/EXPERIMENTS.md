# Experiment workflow

This project uses a small, local-first experiment system. The authoritative record is an immutable run directory on
disk. Weights & Biases is an optional view of the same configuration and metrics, not the only copy of them.

## Core rules

1. Never put large datasets, checkpoints, or generated run directories in Git.
2. Never overwrite a completed run directory.
3. Save the fully resolved configuration before the first optimizer update.
4. Record code identity and dirty state for every run.
5. Append metrics as they are produced so an interrupted run retains its history.
6. Keep training and validation inputs explicit and distinct.
7. A checkpoint is valid only if resuming it has been tested against an uninterrupted run.
8. Use the same run ID in the local directory, research note, and W&B run.

## Machine-specific roots

Two environment variables locate large, machine-specific files:

- `CS336_DATA_ROOT`: raw and tokenized datasets.
- `CS336_RUNS_ROOT`: run directories, checkpoints, logs, and generated summaries.

Set them in the shell that launches training. For example:

```sh
export CS336_DATA_ROOT="/path/to/data"
export CS336_RUNS_ROOT="/path/to/runs/cs336"
```

These absolute paths must not appear in committed experiment presets. A one-off command-line override may supersede
an environment variable. For local convenience, the resolver also reads the ignored `configs/local.toml`; copy
`configs/local.example.toml` to create it on a new machine. The resulting absolute paths and their sources must be
captured in the resolved run configuration. The program should fail clearly when a required root or input file is
missing rather than silently writing into the repository.

A local `data` symlink is optional for browsing. Training must not require it, and `/data` is ignored by Git.

## Configuration contract

Committed presets live under `configs/`. A preset contains the scientific inputs and portable relative paths:

- experiment name, seed, and tags;
- dataset identity and train/validation filenames relative to `CS336_DATA_ROOT`;
- tokenizer artifacts relative to the repository root;
- model architecture;
- optimizer and training schedule;
- evaluation, logging, and checkpoint cadence;
- runtime device and optional W&B mode.

Configuration precedence is:

1. committed preset;
2. explicit command-line overrides;
3. machine roots from command-line arguments;
4. machine roots from `CS336_DATA_ROOT` and `CS336_RUNS_ROOT`;
5. machine roots from the ignored `configs/local.toml`.

Before training, these sources are validated and materialized as one `config.resolved.json`. No implicit configuration
should remain after that point.

## Run directory contract

Presets require exactly one positive integer: `training.steps` or `training.total_tokens`.
Both or neither is an error. For current single-device batches, tokens per update equals batch size times
context length. Token budgets round up to whole updates. The output includes `training.steps` and
`resolved_budget` with source, requested tokens (null for step budgets), tokens per update, steps, and actual tokens.
For example, 65 requested tokens at 16 tokens per update gives 5 steps and 80 actual tokens.
Counts include repeated training positions and exclude validation tokens. Schedules remain in steps.
Resolved output is not an input preset: it preserves requested tokens alongside the derived steps.
Distributed training and gradient accumulation will require extending this calculation when supported.

Each launch creates a unique run ID such as `20260911T160800Z_tinystories-smoke_a1b2c3`. The corresponding directory
under `CS336_RUNS_ROOT` is the unit of experimental evidence:

```text
20260911T160800Z_tinystories-smoke_a1b2c3/
├── checkpoints/
├── command.txt
├── config.input.toml
├── config.resolved.json
├── environment.json
├── events.jsonl
├── git.diff
├── git_status.txt
├── metrics.jsonl
├── source_manifest.json
└── summary.json
```

- `metrics.jsonl` contains machine-readable numerical observations keyed by step and tokens seen.
- `events.jsonl` contains lifecycle events such as start, checkpoint, resume, exception, and completion.
- `summary.json` is written at completion and contains final/best metrics and artifact references.
- `source_manifest.json` hashes the source files that define the run.
- `git.diff` and `git_status.txt` preserve uncommitted code identity without pretending the tree was clean.
- Checkpoints are written atomically and contain only recovery state plus compact metadata, not the full metric history.

The exact checkpoint contract will include model state, optimizer state, completed step, token count, resolved-config
hash, and Python, NumPy, and PyTorch random-number-generator states.

## Research notes

Human reasoning lives in `notes/experiments/`. Start a note before launching a meaningful comparison and link it to the
same run ID or W&B group. Record the question, prediction, changed variable, controls, result, interpretation, and next
decision. Failed or invalid runs stay in the record with an explanation.

## Local verification sequence

The training loop now accepts keyword-only `log_every`, `resume_from`, `stop_after`, and
`resolved_config` arguments. `checkpoint_out` must name a new attempt directory; it contains
configuration, JSONL metrics/events, and periodic/final numbered checkpoints. A resumed attempt uses a
new directory, retains the original run ID, and records the parent checkpoint in its resume event.
Use `stop_after` to interrupt at an absolute step while keeping the full training budget unchanged.
Resumption checks configuration and a hash of the actual training tokens, then restores optimizer
and RNG state. Hashing reads the dataset once per invocation in bounded chunks.
Each checkpoint excludes accumulated metric histories. JSONL records are flushed after each write;
checkpoint files are flushed, fsynced, and replaced atomically. Retention/deletion is not yet enabled.
CPU interruption equivalence is tested; accelerator equivalence has not yet been verified.
The CLI still only performs configuration dry runs; connecting it to training is a subsequent step.
The fuller run-directory contract below/above describes the target workflow; source/environment manifests,
validation evaluation, and W&B integration are not yet wired into this loop.

Resolve the smoke preset without creating a run:

```sh
uv run python -m scripts.train --config configs/smoke.toml --dry-run
```

Infrastructure is promoted in this order:

1. Resolve and validate a preset without starting training.
2. Run a few CPU steps and inspect every generated artifact.
3. Compare an uninterrupted run with an interrupted-and-resumed run from identical initial state.
4. Run a short MPS experiment and confirm local JSONL metrics agree with the console summary.
5. Enable W&B in offline mode, then online mode, and confirm it mirrors the same run ID and metrics.

Debug runs prove mechanics, not scientific conclusions. They are tagged `smoke` and kept separate from evidence runs.

## Cluster transition

Cluster execution comes only after the local sequence passes. Git transports source, presets, and notes; `rsync`
transports large datasets and selected run artifacts; Slurm executes jobs. Cluster job scripts set machine roots and
resource requests without changing the scientific preset. Current Torch storage paths and available partitions must be
verified again before those scripts are added.

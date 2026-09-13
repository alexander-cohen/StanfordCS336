# Experiment workflow

This project uses a small, local-first experiment system. The authoritative record is an immutable run directory on
disk. Weights & Biases is an optional view of the same configuration and metrics, not the only copy of them.

## Core rules

1. Never put large datasets, checkpoints, or generated run directories in Git.
2. Never overwrite a completed run directory.
3. Save the fully resolved configuration before the first optimizer update.
4. Git commit/dirty status are recorded automatically; commit relevant code before important runs.
5. Append metrics as they are produced so an interrupted run retains its history.
6. Keep training and validation inputs explicit and distinct.
7. A checkpoint is valid only if resuming it has been tested against an uninterrupted run.
8. Link research notes to the output directory; a logical run ID groups resumed executions.

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

Each launch creates a unique directory such as `20260911T160800Z_tinystories-smoke_a1b2c3`. The directory
under `CS336_RUNS_ROOT` is the unit of experimental evidence:

```text
20260911T160800Z_tinystories-smoke_a1b2c3/
├── 2.pt
├── 4.pt
├── config.resolved.json
├── metrics.jsonl
└── summary.json
```

- `metrics.jsonl` contains machine-readable numerical observations keyed by step and tokens seen.
- `summary.json` records final metrics and complete/paused status. It is absent if execution fails.
- Checkpoints are written atomically and contain only recovery state plus compact metadata, not the full metric history.

Checkpoints include model/optimizer state, completed step, token count, settings, data hashes, and RNG states.
There is no separate event log, attempt ID, or redundant configuration hash.

## Where the code lives

At launch, `git_commit`, `git_dirty`, and `git_error` are saved in resolved-config provenance,
the local summary, and W&B config. Git failures/timeouts produce unknown values and a warning,
not a false clean status. Dirty runs also save local `code.diff` (staged and unstaged tracked
changes, including binary patches) and `untracked_files.json` (names only). Neither is uploaded
to W&B. Untracked contents are not preserved, so this is not a complete source snapshot.
Ignored data/run files do not make Git dirty. These checks run once, with five-second timeouts.

- `experiment.py`: resolve configuration, load data, seed generators, open logging, and save the summary.
- `training.py`: train, evaluate, save checkpoints, and emit metric dictionaries via `log_metrics`.
- `metrics.py`: write each dictionary to JSONL and optionally the terminal and W&B; close outputs on exit.
- `run_io.py`: atomic writes and RNG recovery helpers.

Terminal output is controlled by the Python `verbose` argument (default true), not TOML.
Direct `training_loop` calls without a callback print metrics but do not create JSONL/config/summary files;
use `run_experiment` for the complete workflow.

## W&B charts

Set `[tracking].wandb_mode` to `offline` for local W&B records or `online` for live charts.
It defaults to `disabled`; JSONL logging is always enabled by the launcher.
For online use, first run `uv run wandb login`, then launch the experiment normally.
Do not put an API key in TOML or Git. Offline records can later be uploaded with
`uv run wandb sync /path/to/output/wandb/offline-run-...`.

Numeric metrics are plotted against optimizer step. Each execution has a distinct W&B run identified by its
directory name; resumed executions share a group via the logical run ID, so restarting an older checkpoint
does not overwrite newer history. W&B opens at the first logged step and closes on success or failure.
Portable settings and numeric metrics are sent; this integration does not upload datasets/checkpoints or
automatically capture source or Git metadata. Console output is captured from the first logged step onward.
W&B may collect system metadata.
Tracking errors propagate, but already-flushed JSONL records and checkpoints remain on disk.
Forwarding, grouping, and cleanup are tested with a fake service. Real SDK verification is opt-in:
`RUN_WANDB_INTEGRATION=1 uv run pytest tests/test_metrics.py -q`.
The local SDK import stalled during verification; real offline execution and online uploads remain unverified.
`estimated_train_flops` logs cumulative approximate training compute: 6 times total model parameters times
training tokens seen, including before a resume. Select it as the x-axis for `validation_loss` in W&B.
This excludes validation/sampling and omits attention-matrix costs; it is not a hardware measurement.
`train_tokens_per_second` and `estimated_train_flops_per_second` average training throughput since the last logged
record. Timing includes batching and optimizer work but excludes evaluation, sampling, checkpointing,
and logging. Accelerators synchronize at the timing boundaries. The first update of each launch is
excluded as warmup; a record with no timed updates omits these rates. FLOP/s uses the same approximate
6N FLOPs/token estimate. Short smoke runs are not reliable hardware benchmarks.

## Research notes

Human reasoning lives in `notes/experiments/`. Start a note before launching a meaningful comparison and link it to the
same run ID or W&B group. Record the question, prediction, changed variable, controls, result, interpretation, and next
decision. Failed or invalid runs stay in the record with an explanation.

## Local verification sequence

Configuration remains a dictionary from resolution through execution and JSON recording.
`run_experiment` passes `resolved["model"]` and `resolved["training"]` directly to the loop;
there are no model/training configuration dataclasses. `learning_rate_at(training, step)` selects the schedule.
The loop saves the resolved dictionary directly, without nesting another copy of it. Standalone loop tests
may supply the two dictionaries directly; in that case a minimal configuration record is assembled.
Pass `device`, `checkpoint_every`, and `log_every` directly to `training_loop` from the runtime, checkpoint,
and logging sections. The loop defaults to CPU and a checkpoint cadence of 100; `run_experiment` resolves
`device = "auto"` to a concrete device before calling it.
Checkpoints with obsolete configuration field names cannot resume without an explicit migration.
Existing format-1 checkpoints with the current settings layout still load; obsolete extra fields are ignored.

The training loop now accepts keyword-only `log_every`, `resume_from`, `stop_after`, and
`resolved_config` arguments. The launcher creates a fresh directory and passes a metrics callback;
standalone calls create a fresh checkpoint directory. A resumed execution uses a
new directory and retains the original run ID.
Use `stop_after` to interrupt at an absolute step while keeping the full training budget unchanged.
Resumption checks model/training/device settings and hashes of the actual training and validation tokens, then restores optimizer
and RNG state. Hashing reads the dataset once per invocation in bounded chunks.
Each checkpoint excludes accumulated metric histories. JSONL records are flushed after each write;
checkpoint files are flushed, fsynced, and replaced atomically. `keep_last` retains the newest checkpoints in the
current execution only, removing older ones after a successful save. Parent executions
are untouched. Deleted checkpoints are not recoverable from this workflow; retain copies separately if needed.
CPU interruption equivalence is tested; accelerator equivalence has not yet been verified.
The CLI supports execution and dry runs. Launch with `uv run python -m scripts.run_experiment --config configs/smoke.toml`.
Use `--stop-after 2` to pause at step 2 and `--resume-from /absolute/path/to/2.pt` to continue in a fresh attempt.
During training, the first Ctrl+C requests a graceful pause: finish the next optimizer update, evaluate,
log metrics, save a checkpoint, skip text sampling, close W&B, and write a paused summary. The original
training budget stays unchanged for resume. A request during evaluation/logging is checked after the
next update. A second Ctrl+C raises KeyboardInterrupt immediately (normal cleanup still runs), so it
does not guarantee a new checkpoint or summary. The previous signal handler is restored afterward.
This behavior applies to main-thread `run_experiment` calls; setup before training uses normal Ctrl+C.
`run_experiment` seeds generators, validates and memory-maps integer NPY token arrays, and writes a final summary.
Validation uses repeatable batches without advancing training RNG state. Final-step metrics include validation.
Data checking and hashing read the datasets in bounded chunks before training, even for short smoke runs.
The launcher loads the trusted tokenizer pickle files and checks vocabulary IDs against the model.
Training still consumes pretokenized IDs; the tokenizer produces 20-token samples at checkpoint steps,
printed when verbose is enabled. Sampling preserves training RNG state. Vocabulary size checks do not
prove the datasets used the same tokenizer; that compatibility must be established during preprocessing.
Summary JSON is published atomically; summary failures leave saved checkpoints available without a success summary.
Absolute paths, TOML comments, logging cadence, and provenance are excluded from resume compatibility comparisons.

Resolve the smoke preset without creating a run:

```sh
uv run python -m scripts.run_experiment --config configs/smoke.toml --dry-run
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
resource requests without changing the scientific preset. Check your Torch allocation and storage paths before
submitting; see the submission instructions below.

## Submit on NYU Torch

From the cluster checkout's `assignments/assignment1-basics` directory, prepare the Linux environment
and authenticate W&B once:

```sh
export UV_CACHE_DIR="$SCRATCH/uv-cache"
uv sync --locked
uv run wandb login
my_slurm_accounts
```

Copy the tokenized train and validation arrays into `$SCRATCH/data/tinystories/`. Tokenizer artifacts
are tracked in Git. The batch script defaults to `$SCRATCH/data` and `$SCRATCH/runs/cs336`;
export `CS336_DATA_ROOT` and `CS336_RUNS_ROOT` before submission to override these paths.
Use an account returned by `my_slurm_accounts` in place of `YOUR_SLURM_ACCOUNT` below.

Start with a short GPU check, then submit the full cosine preset:

```sh
sbatch --account=YOUR_SLURM_ACCOUNT scripts/train.slurm --config configs/smoke.toml --stop-after 4
sbatch --account=YOUR_SLURM_ACCOUNT scripts/train.slurm
squeue -u "$USER"
```

The script requests one GPU, four CPUs, 32 GB RAM, and four hours. Override resources with
`sbatch` options before the script path. Launcher options follow the script path. CUDA is required.
Slurm combines stdout and stderr in `slurm-JOBID.out` in the submission directory; online W&B
runs also capture metrics and console output. Submit from the assignment directory.
The cluster environment must already be installed: the job uses `uv run --no-sync`.
The time limit may interrupt training; retain checkpoints and use `--resume-from` to continue.

## Memory monitoring

Every logged step includes `memory_process_rss_bytes`. CUDA runs also record
`memory_cuda_allocated_bytes`, `memory_cuda_reserved_bytes`,
`memory_cuda_peak_allocated_bytes`, `memory_cuda_peak_reserved_bytes`,
`memory_cuda_free_bytes`, and `memory_cuda_total_bytes`. Allocator retry and OOM counters
(`memory_cuda_num_alloc_retries`, `memory_cuda_num_ooms`) are included when the allocator exposes them.
Numeric values go to JSONL and W&B; console memory values use GiB. Sampling is outside the timed
training interval. CUDA peaks/counters are process-wide since startup or an external stats reset;
they include validation and sampling and are not reset by this logger.

Allocated memory measures live PyTorch tensors; reserved memory also includes reusable allocator cache.
Free/total memory is device-wide, including other processes and non-PyTorch allocations. A large reserved
value alone does not imply an impending OOM. Low free memory together with a high tensor peak and
increasing allocation retries suggests capacity pressure. These metrics do not diagnose memory-bandwidth
bottlenecks; that requires profiling. Fatal OOMs can occur before another metric is written.

On MPS, `memory_mps_allocated_bytes`, `memory_mps_driver_bytes`, and
`memory_mps_recommended_max_bytes` report tensor use, driver allocations, and the recommended working-set
limit. That recommendation is not free VRAM. Process RSS is not total Slurm job memory and excludes
other processes and some job-accounted memory such as filesystem cache; use Slurm accounting for job totals.

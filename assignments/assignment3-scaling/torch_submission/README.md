# Assignment 3 on NYU Torch

Submit the course's JSON training configurations over SSH. Each run gets one
L40S, its own source snapshot, and saved validation losses. No API server,
Stanford account, Modal deployment, or W&B login is required.

Use `./scripts/torch-a3` as a short form of
`.venv/bin/python -m torch_submission` in the commands below.

## Setup

From this assignment directory, with the NYU VPN and `ssh torch` working:

```sh
.venv/bin/python -m torch_submission setup
.venv/bin/python -m torch_submission prepare-data
.venv/bin/python -m torch_submission data-status
```

Setup defaults to `/scratch/ac6074/cs336-a3` and account
`torch_pr_1323_general`. Override these with `setup --root ... --account ...`.
Your local settings are saved in ignored `torch_submission/settings.local.json`.
Rerun `setup` after changing the harness to upload changes. Jobs already
submitted keep their saved source.

Data preparation is a CPU Slurm job. It invokes the supplied DCLM tokenizer and
shuffler on a deterministic subset of Hugging Face's
`mlfoundations/dclm-baseline-1.0-parquet`, using
`NousResearch/Llama-2-7b-hf` and the course's 32K vocabulary. By default it
prepares approximately 4.3 billion training tokens (eight 1 GiB shards).
The last shard supplies labels across shard boundaries, leaving approximately
3.76 billion usable training tokens. Set `prepare-data --train-tokens INTEGER`
before the initial preparation to choose a different size. Data cannot be
changed within an existing environment; use a new `--root` for a different
dataset. Preparation must finish before submitting experiments.

## Experiments

Save your own `TrainingConfig` JSON, using the exact fields in the handout or
`examples/client_example.ipynb`:

```sh
.venv/bin/python -m torch_submission submit experiment.json
.venv/bin/python -m torch_submission list
.venv/bin/python -m torch_submission get 1
.venv/bin/python -m torch_submission budget
.venv/bin/python -m torch_submission fetch
.venv/bin/python -m torch_submission cancel 1
```

`submit` returns an experiment ID and Slurm job ID. `get` returns the
configuration and `status.val_losses`; the last loss is the final validation
loss for a completed run. Failed/timeout runs retain partial losses. Fetch
downloads logs, configurations, events, results, and budget summaries to
`torch-results/`. Use `fetch --experiment-id 1 results/run1` for one run.

For notebooks, import `torch_submission.client` and call
`submit_experiment(config)`, `get_experiment(id)`, `list_experiments()`,
`get_budget()`, or `fetch_results(path)`. These return ordinary dictionaries;
the existing Stanford HTTP client is unchanged.

Results also include a `metrics` object:

- `effective_flops_per_second` and `effective_tflops_per_second`: estimated
  throughput computed as `6 * model_params * completed_train_tokens / training_seconds`.
  Successful runs use all requested tokens and the actual, unclipped training
  timer. That timer includes validation and excludes initial compilation and
  data staging. This is approximate model FLOP throughput rather than hardware
  instruction counting. Failed or interrupted runs report a null rate because
  tokens processed within an unfinished evaluation chunk are unknown.
- `peak_gpu_memory_bytes` and `peak_gpu_memory_gib`: JAX allocator's native
  `peak_bytes_in_use` counter over the worker's lifetime, including model
  initialization, training, and validation. It measures active allocations;
  `gpu_memory_pool_bytes` separately reports the preallocated pool. Allocations
  outside JAX's allocator are excluded. Unsupported statistics return null.

The same metrics are saved in `result.json` and successful runs'
`training_result.json`, including final jobs. Counter reads happen at existing
evaluation boundaries and finish reporting, with no polling thread, profiler,
or extra GPU synchronization. A hard-killed worker retains its last saved
partial measurements; completed historical runs cannot recover an unrecorded
memory peak. See [JAX's memory guidance](https://codelabs.developers.google.com/jax-gpu-03-profiling-debugging#8)
for allocator versus reserved memory.

## Time accounting

The authoritative file on Torch is
`/scratch/ac6074/cs336-a3/ledger.sqlite3`. A readable summary is saved beside it
as `budget.json`; each run also has `runs/ID/result.json`.

The experiment budget is 43,200 seconds of training time on an L40S. Queued and
running experiments reserve `max_runtime_seconds`. Finished/failed experiments
are charged reported training time, clipped between one second and their
requested maximum, following the supplied Stanford budget rules. A job canceled
before starting uses zero seconds. Duplicate experiment configurations are
rejected, and concurrent submissions cannot overreserve the budget.
Explicitly refunded runner infrastructure failures can be resubmitted with the
same configuration; their original failure and full job wall time remain recorded.

`budget.json` distinguishes `spent_seconds`, `reserved_seconds`, and
`remaining_seconds`. `total_training_seconds` records actual training duration,
including timeout grace; `total_job_seconds` records full GPU job wall time,
including startup, data staging, and JAX compilation. Queue wait is excluded.
`final_job_seconds` tracks final runs separately. Running totals update on
worker heartbeats; completed job wall time is reconciled with Slurm `sacct`
whenever you retrieve status or budget, including after a node failure or hard
kill. If Slurm accounting is delayed, the reservation remains in place.

Compilation is excluded from the training budget, as in the supplied trainer.
Slurm grants one extra hour for setup plus the trainer's timeout grace.
Automatic requeue and preemption are disabled. The course's trainer does not
resume checkpoints; an interrupted run is recorded as failed.

## Final job

Create your own JSON matching the handout's final-submission request:

```json
{
  "training_config": {},
  "predicted_final_loss": 3.0
}
```

Replace the empty object with your complete configuration and the example
prediction with your own prediction, then run:

```sh
.venv/bin/python -m torch_submission final final_submission.json
```

This saves your prediction and launches the actual final job. It accepts up to
172,800 seconds (48 hours) of training, runs on one L40S, and is accounted
separately from the experiment budget. Status and results use the same commands.
Up to two final jobs may be active at once; Slurm may run them simultaneously
when GPUs are available. Each keeps its own configuration, prediction, logs,
and results under `runs/ID/`. The latest submission is also saved
as `final_submission.json` at the state root.

These are personal L40S budgets and a locally prepared DCLM subset. They do not
reproduce the handout's B200 compute budget, Stanford's exact shuffled corpus,
or its larger leaderboard validation set. The local trainer retains the
course's 262,144-token validation size and training/configuration checks.

## Files on Torch

```text
/scratch/ac6074/cs336-a3/
  repo/                  uploaded assignment and harness
  venv/                  CUDA 12 / JAX 0.9.1 environment
  ledger.sqlite3         authoritative experiment and time records
  budget.json            readable time totals
  data/manifest.json     fixed dataset paths and provenance
  prepare-data.out       CPU data-preparation log
  final_submission.json  latest final configuration and prediction
  runs/ID/
    training_config.json
    data_manifest.json
    source/              source frozen at submission time
    source_hashes.json
    slurm.out
    slurm.err
    log.log
    events.jsonl
    environment.json
    result.json
    training_result.json # available for successful runs
```

Fetch results periodically to keep a local copy; Torch scratch is temporary
storage. The accounting database should remain on Torch rather than being
edited from a downloaded copy.

GPU requests follow [NYU's Slurm instructions](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/):
`--gres=gpu:1 --constraint=l40s`, with an explicit account and no manually
selected partition. CUDA 12 requirements follow
[JAX's installation guidance](https://docs.jax.dev/en/latest/installation.html).

Validation: thirteen data-staging/accounting/submission/telemetry checks pass. Synthetic experiment and
final-mode jobs completed on single L40S GPUs, with results and Slurm time
accounting verified. These smoke jobs use a separate ledger and do not charge
the assignment experiment budget. Full Hugging Face data preparation runs
separately; `data-status` reports whether the dataset is ready.

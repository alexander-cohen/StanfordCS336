"""Invoke the course's existing DCLM tokenizer/shuffler on a bounded HF subset."""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

from torch_submission.store import atomic_json, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--train-tokens", type=int, default=2**32)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "data" / "manifest.json"
    if manifest_path.exists():
        print(manifest_path.read_text())
        return
    if args.train_tokens <= 0:
        parser.error("train-tokens must be positive")
    from furu.config import config as furu_config
    from cs336_scaling import tokenized_data as course

    furu_config.directories.data = root / "data" / "furu"
    temp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    course._LOCAL_RAW_TOKEN_DIR = temp_dir / "raw_tokens"
    course._LOCAL_FINAL_SHARD_DIR = temp_dir / "final_shards"
    course.DCLMData.n_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    data = course.DCLMData(
        total_tokens=args.train_tokens,
        validation_tokens=2**18 + 512,
        shuffle_run_target_bytes=256 * 1024**2,
    )
    required = (
        math.ceil(data.total_tokens / data.tokens_per_chunk) * data.tokens_per_chunk
        + data.validation_tokens
    )
    seen = 0
    source_urls = []
    for url in data.shuffled_dclm_urls.load_or_create():
        result = course.TokenizedDCLMChunk(url=url).load_or_create()
        seen += result.n_tokens
        source_urls.append(url)
        print(f"cached {seen:,}/{required:,} tokens", flush=True)
        if seen >= required:
            break
    if seen < required:
        raise RuntimeError("not enough source tokens")
    result = data.load_or_create()
    manifest = json.loads(json.dumps(dataclasses.asdict(result), default=str))
    manifest.update(
        created_at=now(),
        source="mlfoundations/dclm-baseline-1.0-parquet",
        tokenizer="NousResearch/Llama-2-7b-hf",
        source_urls=source_urls,
        seed=67,
        format="uint16 .npy",
        subset=True,
    )
    atomic_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()

import json
from types import SimpleNamespace

from torch_submission.telemetry import RunTelemetry
from torch_submission.store import Store


def test_completed_run_reports_rate_and_allocator_peak():
    counters = {"peak_bytes_in_use": 3 * 2**30, "pool_bytes": 40 * 2**30}
    telemetry = RunTelemetry(SimpleNamespace(memory_stats=lambda: counters), 1000)
    telemetry.model_created(2000)
    telemetry.progress(1000)
    metrics = telemetry.snapshot(2.0, completed=True)
    assert metrics["estimated_train_flops"] == 12_000_000
    assert metrics["effective_flops_per_second"] == 6_000_000
    assert metrics["effective_tflops_per_second"] == 6e-6
    assert metrics["peak_gpu_memory_gib"] == 3
    # Allocator pool reservation must not be mistaken for active memory usage.
    assert metrics["gpu_memory_pool_bytes"] == 40 * 2**30


def test_failed_run_uses_completed_tokens_and_does_not_claim_full_run_throughput():
    telemetry = RunTelemetry(SimpleNamespace(memory_stats=lambda: None), 1000)
    telemetry.model_created(2000)
    telemetry.progress(250)
    metrics = telemetry.snapshot(2.0)
    assert metrics["completed_train_tokens"] == 250
    assert metrics["estimated_train_flops"] == 3_000_000
    assert metrics["effective_flops_per_second"] is None
    assert metrics["peak_gpu_memory_bytes"] is None
    assert not metrics["metrics_complete"]


def test_memory_query_failure_retains_peak_and_does_not_fail_experiment():
    counters = iter(
        [{"peak_bytes_in_use": 1234}, RuntimeError("allocator unavailable")]
    )

    def query():
        result = next(counters)
        if isinstance(result, Exception):
            raise result
        return result

    telemetry = RunTelemetry(SimpleNamespace(memory_stats=query), 1000)
    telemetry.progress(250)
    metrics = telemetry.snapshot(2.0)
    assert metrics["peak_gpu_memory_bytes"] == 1234
    assert "allocator unavailable" in metrics["memory_stats_error"]


def test_metrics_are_persisted_in_returned_record_and_result_file(tmp_path):
    store = Store(tmp_path)
    record = store.reserve({"max_runtime_seconds": 60}, "metrics", "experiment")
    metrics = {"effective_flops_per_second": 6000000.0, "peak_gpu_memory_bytes": 1234}
    store.finish(
        record["experiment_id"], "completed", 2.0, metrics=metrics, val_losses=[4.0]
    )
    assert store.get(1)["metrics"] == metrics
    assert (
        json.loads((store.run_dir(1) / "result.json").read_text())["metrics"] == metrics
    )

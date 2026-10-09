"""Read JAX's existing allocator counters; no sampling thread or device sync."""

import threading


class RunTelemetry:
    def __init__(self, device, total_train_tokens):
        self.device = device
        self.total_train_tokens = total_train_tokens
        self.model_params = None
        self.completed_train_tokens = 0
        self.peak_gpu_memory_bytes = None
        self.gpu_memory_pool_bytes = None
        self.memory_stats_error = None
        self.lock = threading.Lock()

    def _read_memory(self):
        try:
            stats = self.device.memory_stats() or {}
            peak = stats.get("peak_bytes_in_use")
            if peak is not None:
                self.peak_gpu_memory_bytes = max(
                    self.peak_gpu_memory_bytes or 0, int(peak)
                )
            pool = stats.get("pool_bytes")
            if pool is not None:
                self.gpu_memory_pool_bytes = int(pool)
        except Exception as exc:
            # A telemetry/backend failure must not turn a valid training run into
            # a failed experiment. Retain the last allocator counter we read.
            self.memory_stats_error = f"{type(exc).__name__}: {exc}"

    def model_created(self, model_params):
        with self.lock:
            self.model_params = int(model_params)

    def progress(self, train_tokens):
        with self.lock:
            self.completed_train_tokens = max(
                self.completed_train_tokens, int(train_tokens)
            )
            self._read_memory()

    def snapshot(self, training_seconds=None, completed=False, read_memory=True):
        with self.lock:
            if read_memory:
                self._read_memory()
            tokens = (
                self.total_train_tokens if completed else self.completed_train_tokens
            )
            flops = (
                6 * self.model_params * tokens
                if self.model_params is not None
                else None
            )
            rate = (
                flops / training_seconds
                if completed
                and flops is not None
                and training_seconds
                and training_seconds > 0
                else None
            )
            peak = self.peak_gpu_memory_bytes
            result = {
                "model_params": self.model_params,
                "completed_train_tokens": tokens,
                "training_seconds": training_seconds,
                "estimated_train_flops": flops,
                "effective_flops_per_second": rate,
                "effective_tflops_per_second": rate / 1e12
                if rate is not None
                else None,
                "flops_time_scope": "run training timer; includes validation; excludes initial compilation",
                "peak_gpu_memory_bytes": peak,
                "peak_gpu_memory_gib": peak / 2**30 if peak is not None else None,
                "gpu_memory_pool_bytes": self.gpu_memory_pool_bytes,
                "gpu_memory_source": "jax.Device.memory_stats().peak_bytes_in_use",
                "gpu_memory_scope": "JAX allocator peak since worker startup, including initialization and validation",
                "metrics_complete": completed,
            }
            if self.memory_stats_error:
                result["memory_stats_error"] = self.memory_stats_error
            return result

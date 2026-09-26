# FlashAttention Full Benchmark

Full default sweep from `scripts/benchmark_flash_attention.py`, run through `scripts/basic_srun.sh`.

- Device: `cuda` (GPU model not captured in the JSONL output)
- Batch size: 1; causal masking enabled; Triton block size: 64
- `triton.testing.do_bench`: 25 ms warmup, 100 ms repetition
- Latencies are milliseconds. Each cell lists forward / backward-only / forward-plus-backward.
- Reference: `cs336_basics.model.scaled_dot_product_attention`; Triton: `triton_flash_attn`.

| Dtype | Head dim | Sequence length | Reference F / B / FB (ms) | Triton F / B / FB (ms) |
|---|---:|---:|---:|---:|
| bfloat16 | 16 | 128 | 0.029 / 0.051 / 0.075 | 0.010 / 0.050 / 0.058 |
| bfloat16 | 16 | 256 | 0.031 / 0.057 / 0.080 | 0.014 / 0.051 / 0.062 |
| bfloat16 | 16 | 512 | 0.037 / 0.069 / 0.093 | 0.021 / 0.069 / 0.087 |
| bfloat16 | 16 | 1024 | 0.052 / 0.105 / 0.135 | 0.034 / 0.060 / 0.092 |
| bfloat16 | 16 | 2048 | 0.094 / 0.188 / 0.237 | 0.060 / 0.132 / 0.185 |
| bfloat16 | 16 | 4096 | 0.382 / 0.908 / 1.206 | 0.114 / 0.867 / 0.985 |
| bfloat16 | 16 | 8192 | 2.852 / 6.545 / 9.250 | 0.221 / 3.886 / 4.069 |
| bfloat16 | 16 | 16384 | 12.322 / 26.731 / 38.912 | 0.705 / 14.845 / 15.471 |
| bfloat16 | 16 | 32768 | 48.836 / 105.769 / 154.263 | 2.824 / 59.758 / 62.084 |
| bfloat16 | 16 | 65536 | oom / n/a / n/a | 11.091 / oom / oom |
| bfloat16 | 32 | 128 | 0.029 / 0.050 / 0.075 | 0.013 / 0.053 / 0.063 |
| bfloat16 | 32 | 256 | 0.031 / 0.055 / 0.080 | 0.017 / 0.050 / 0.065 |
| bfloat16 | 32 | 512 | 0.037 / 0.068 / 0.093 | 0.027 / 0.054 / 0.079 |
| bfloat16 | 32 | 1024 | 0.054 / 0.098 / 0.133 | 0.046 / 0.089 / 0.132 |
| bfloat16 | 32 | 2048 | 0.097 / 0.195 / 0.234 | 0.083 / 0.244 / 0.321 |
| bfloat16 | 32 | 4096 | 0.383 / 0.915 / 1.215 | 0.163 / 0.948 / 1.098 |
| bfloat16 | 32 | 8192 | 2.852 / 6.556 / 9.269 | 0.321 / 4.117 / 4.419 |
| bfloat16 | 32 | 16384 | 11.803 / 26.317 / 37.971 | 1.020 / 16.293 / 17.295 |
| bfloat16 | 32 | 32768 | 46.692 / 103.598 / 150.028 | 4.101 / 65.032 / 69.667 |
| bfloat16 | 32 | 65536 | oom / n/a / n/a | 16.190 / oom / oom |
| bfloat16 | 64 | 128 | 0.030 / 0.051 / 0.075 | 0.072 / 0.054 / 0.125 |
| bfloat16 | 64 | 256 | 0.032 / 0.057 / 0.081 | 0.135 / 0.047 / 0.181 |
| bfloat16 | 64 | 512 | 0.039 / 0.071 / 0.096 | 0.260 / 0.055 / 0.316 |
| bfloat16 | 64 | 1024 | 0.057 / 0.099 / 0.134 | 0.515 / 0.089 / 0.605 |
| bfloat16 | 64 | 2048 | 0.101 / 0.207 / 0.253 | 1.041 / 0.223 / 1.262 |
| bfloat16 | 64 | 4096 | 0.349 / 0.885 / 1.128 | 2.149 / 1.004 / 3.155 |
| bfloat16 | 64 | 8192 | 2.776 / 6.539 / 9.282 | 14.958 / 3.790 / 18.711 |
| bfloat16 | 64 | 16384 | 11.835 / 26.310 / 38.039 | 62.131 / 15.270 / 77.032 |
| bfloat16 | 64 | 32768 | 46.478 / 103.786 / 149.978 | 261.508 / 58.315 / 319.699 |
| bfloat16 | 64 | 65536 | oom / n/a / n/a | 1141.798 / oom / oom |
| bfloat16 | 128 | 128 | 0.031 / 0.053 / 0.077 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 256 | 0.033 / 0.057 / 0.082 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 512 | 0.041 / 0.067 / 0.095 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 1024 | 0.058 / 0.100 / 0.135 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 2048 | 0.117 / 0.209 / 0.264 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 4096 | 0.406 / 0.910 / 1.195 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 8192 | 2.893 / 6.561 / 9.335 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 16384 | 11.884 / 26.285 / 38.087 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 32768 | 48.342 / 104.852 / 152.899 | resource_limit / n/a / n/a |
| bfloat16 | 128 | 65536 | oom / n/a / n/a | resource_limit / n/a / n/a |
| float32 | 16 | 128 | 0.035 / 0.061 / 0.092 | 0.012 / 0.045 / 0.054 |
| float32 | 16 | 256 | 0.036 / 0.065 / 0.093 | 0.016 / 0.046 / 0.060 |
| float32 | 16 | 512 | 0.053 / 0.087 / 0.123 | 0.024 / 0.069 / 0.091 |
| float32 | 16 | 1024 | 0.065 / 0.126 / 0.157 | 0.041 / 0.079 / 0.118 |
| float32 | 16 | 2048 | 0.135 / 0.328 / 0.371 | 0.075 / 0.228 / 0.300 |
| float32 | 16 | 4096 | 0.924 / 2.482 / 3.312 | 0.143 / 0.936 / 1.072 |
| float32 | 16 | 8192 | 5.793 / 13.061 / 18.698 | 0.279 / 4.168 / 4.427 |
| float32 | 16 | 16384 | 22.455 / 51.574 / 74.212 | 1.004 / 16.185 / 17.143 |
| float32 | 16 | 32768 | 88.966 / 203.887 / 293.007 | 4.050 / 65.244 / 69.433 |
| float32 | 16 | 65536 | oom / n/a / n/a | 16.103 / oom / oom |
| float32 | 32 | 128 | 0.036 / 0.065 / 0.096 | 0.015 / 0.048 / 0.061 |
| float32 | 32 | 256 | 0.035 / 0.062 / 0.091 | 0.023 / 0.042 / 0.064 |
| float32 | 32 | 512 | 0.039 / 0.073 / 0.102 | 0.037 / 0.045 / 0.080 |
| float32 | 32 | 1024 | 0.064 / 0.128 / 0.162 | 0.066 / 0.073 / 0.138 |
| float32 | 32 | 2048 | 0.162 / 0.354 / 0.408 | 0.124 / 0.194 / 0.313 |
| float32 | 32 | 4096 | 0.934 / 2.489 / 3.333 | 0.239 / 0.885 / 1.119 |
| float32 | 32 | 8192 | 5.652 / 12.985 / 18.521 | 0.474 / 3.787 / 4.241 |
| float32 | 32 | 16384 | 22.371 / 51.465 / 73.628 | 1.874 / 14.882 / 16.701 |
| float32 | 32 | 32768 | 88.803 / 204.293 / 293.416 | 7.439 / 59.177 / 66.428 |
| float32 | 32 | 65536 | oom / n/a / n/a | 29.643 / oom / oom |
| float32 | 64 | 128 | 0.036 / 0.064 / 0.095 | 0.052 / 0.047 / 0.105 |
| float32 | 64 | 256 | 0.034 / 0.062 / 0.089 | 0.093 / 0.040 / 0.138 |
| float32 | 64 | 512 | 0.042 / 0.076 / 0.107 | 0.174 / 0.050 / 0.227 |
| float32 | 64 | 1024 | 0.073 / 0.132 / 0.177 | 0.335 / 0.086 / 0.422 |
| float32 | 64 | 2048 | 0.180 / 0.381 / 0.456 | 0.664 / 0.219 / 0.895 |
| float32 | 64 | 4096 | 0.990 / 2.605 / 3.476 | 1.344 / 1.000 / 2.327 |
| float32 | 64 | 8192 | 5.698 / 13.120 / 18.684 | 2.963 / 3.821 / 6.873 |
| float32 | 64 | 16384 | 22.526 / 51.466 / 74.010 | 11.638 / 15.352 / 26.968 |
| float32 | 64 | 32768 | 88.395 / 203.735 / 292.409 | 46.056 / 58.213 / 104.969 |
| float32 | 64 | 65536 | oom / n/a / n/a | 173.845 / oom / oom |
| float32 | 128 | 128 | 0.033 / 0.059 / 0.085 | resource_limit / n/a / n/a |
| float32 | 128 | 256 | 0.037 / 0.064 / 0.098 | resource_limit / n/a / n/a |
| float32 | 128 | 512 | 0.049 / 0.086 / 0.119 | resource_limit / n/a / n/a |
| float32 | 128 | 1024 | 0.086 / 0.153 / 0.209 | resource_limit / n/a / n/a |
| float32 | 128 | 2048 | 0.232 / 0.429 / 0.554 | resource_limit / n/a / n/a |
| float32 | 128 | 4096 | 1.121 / 2.855 / 3.924 | resource_limit / n/a / n/a |
| float32 | 128 | 8192 | 5.885 / 13.386 / 19.172 | resource_limit / n/a / n/a |
| float32 | 128 | 16384 | 23.044 / 52.272 / 75.279 | resource_limit / n/a / n/a |
| float32 | 128 | 32768 | 92.024 / 207.814 / 299.215 | resource_limit / n/a / n/a |
| float32 | 128 | 65536 | oom / n/a / n/a | resource_limit / n/a / n/a |

Unavailable measurements:

- `oom` means CUDA allocation failed. At sequence length 65,536, reference forward and backward-related Triton passes could not allocate the dense intermediates; entries retain their per-pass status.
- `resource_limit` means Triton could not launch with the configured tile due to per-program shared-memory use. All Triton cases with head dimension 128 have this status on the allocated GPU; this is not a global-memory OOM.
- Recorded status counts across implementation passes: `{'oom': 20, 'resource_limit': 20}`.

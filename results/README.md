# Measured results

Run `python -m benchmarks.compare_runtimes` after training and ONNX export.
It writes `benchmark.csv` and `benchmark.json` here; neither is committed.

The CSV contains measured ROC-AUC, probability differences versus PyTorch,
mean/p50/p95 batch latency in milliseconds, samples/second, and runtime status.
Missing runtimes have blank measurements and explanatory notes. The JSON records
the checkpoint hash, OS/architecture, package versions, batch size, warmups,
iterations, and thread policy. Keep the CSV and JSON together when sharing a run.

Inference timing excludes model load and file I/O. It includes input setup,
embedded standardization, model execution, and output materialization. Throughput
is batch size divided by mean elapsed seconds. C++ results come from a fresh
subprocess invocation, while its timer excludes process startup and CSV loading.

Never fill missing cells with estimates or compare measurements across different
machines as if they shared an experimental setup. Read the root README for local
execution status and limitations.

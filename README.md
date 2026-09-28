# Cross-Platform Predictive ML

An end-to-end predictive modeling and deployment pipeline that trains a
PyTorch customer conversion model over large-scale transactional data and
deploys the same model through ONNX Runtime/C++, Core ML, and LiteRT.

## Overview

**Can the same PyTorch predictive model be deployed across ONNX Runtime,
Core ML, and LiteRT while preserving prediction fidelity and supporting
efficient production inference?**

This repository makes that question testable: train one checkpoint, export its
preprocessing and probability calculation, compare predictions on identical
held-out rows, and measure warm CPU inference. Optional runtimes have real
implementations and explicit availability checks. Missing measurements stay blank.

## Architecture

```text
Synthetic Transactions
        │
        ▼
   DuckDB / SQL
        │
        ▼
  Feature Matrix
        │
        ▼
     PyTorch
        │
   ┌────┼─────────┐
   ▼    ▼         ▼
 ONNX  Core ML   LiteRT
   │
   ▼
ONNX Runtime
   │
   ▼
C++ Inference
```

The deployment contract is a contiguous float32 matrix `[batch, 12]` of **raw**
features, in saved feature order, returning `[batch, 1]` purchase probabilities.
All exports embed the training scaler and sigmoid. Do not standardize inputs a
second time. C++ checks the CSV header against the ONNX feature metadata.

## Why this project exists

A useful deployment prototype connects data semantics to serving behavior.
SQL cutoffs prevent future information from leaking into features; a single
inference wrapper prevents preprocessing drift; numerical checks catch export
regressions; and a native executable tests the contract outside Python.
The small MLP keeps attention on these boundaries.

## Dataset

The generator creates deterministic synthetic purchases over 360 days starting
2024-01-01, with IDs, timestamps, amounts, categories, discounts, refunds, channels,
and customer tenure. Each customer has persistent activity, spending, category,
and channel preferences. Heavy-tailed activity makes repeat purchases learnable.
The generator writes 100,000-row Parquet chunks instead of retaining every
transaction in memory. The default is 100,000 transactions and 5,000 customers;
5 million transactions produce 250,000 customers.

The target, `will_purchase_next_30d`, means at least one recorded purchase in
`[cutoff, cutoff + 30 days)`, whether or not it is later refunded. It is computed
from actual future transactions, never independently assigned. Refunds carry
their own event timestamps so history only includes refunds already observed.

Training uses days 180, 210, and 240; validation uses day 270; test uses day 300.
Every label window ends before or exactly at the next split's cutoff. The same
customers can occur at later cutoffs: this measures forecasting for an existing
customer population, not generalization to unseen customers. IDs are excluded
from model inputs. All customers signed up before observation began; absent
history is retained with zero aggregates and recency of 180 days.

This is a stationary synthetic task, not evidence of business impact or realistic
production drift. The raw generator is bounded by chunk size plus customer
state; preparation and training still materialize the much smaller customer
feature matrices in memory.

## Feature engineering

[`sql/create_features.sql`](sql/create_features.sql) contains the joins, temporal
filters, aggregation, and labels. DuckDB scans Parquet directly and can spill to
disk under a configurable memory limit. Pandas is used only to construct generator
chunks, not to calculate features.

The 180-day lookback produces 12 features: recency, transaction count, 30/90-day
counts, total/average/maximum order value, observed refund rate, discount rate,
category diversity, online transaction ratio, and tenure at cutoff. Spend is
gross purchase value; refunds are represented separately. Tenure uses the
customer signup date, not a future transaction's recorded tenure.

`prepare_data` writes `features.duckdb`, train/validation/test NPZ matrices,
`data_metadata.json`, and `test_features.csv` under `artifacts/`.
[`sql/analytics.sql`](sql/analytics.sql) provides cohort/conversion summaries
against the saved database.

## Model

`ConversionMLP`: `12 → 64 → ReLU → 32 → ReLU → 1` logits, optimized with Adam and
`BCEWithLogitsLoss`. Scaling statistics are fitted on training rows only. CPU
training uses fixed seeds and one thread. The checkpoint with the lowest
validation loss is saved; the test set does not select epochs or thresholds.

`model.pt` contains weights, ordered feature names, scaler statistics, split
metadata, seed, hyperparameters, and the PyTorch version. A readable copy is in
`model_metadata.json`. Evaluation reports ROC-AUC, average precision (the chosen
PR-AUC summary), accuracy, precision, recall, F1, Brier score, and ten-bin ECE.
Classification uses a fixed 0.5 threshold.

## Deployment targets

- **ONNX:** real PyTorch export, opset 17, dynamic batches, ONNX checker, and
  ONNX Runtime CPU validation. The explicit trace exporter keeps dependencies
  small; recent PyTorch versions emit a deprecation warning for this path.
- **Core ML:** direct `torch.jit.trace` → `coremltools.convert` → ML Program,
  saved as `model.mlpackage`, with float32 compute, CPU execution, macOS 13+
  deployment target, and batches 1–4096. Apple's supported direct conversion
  paths are documented in its [PyTorch workflow](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html).
- **LiteRT:** `litert_torch.convert` → `.tflite`, with a fixed batch size (256
  by default). Inference uses `ai_edge_litert.compiled_model.CompiledModel`,
  without importing legacy TensorFlow Lite APIs. Google's
  [converter documentation](https://github.com/google-ai-edge/litert-torch#installation)
  lists Linux as the conversion environment; the
  [Python runtime documentation](https://developers.google.com/edge/litert/next/python)
  describes the current buffer-based API. Dependencies are optional because the
  converter pulls in a substantial toolchain.

Each export gets a checkpoint SHA-256 sidecar; comparisons reject stale exports
after retraining. A missing optional package or unsupported export platform
prints an actionable message and exits 2. Conversion failures exit 1; parity
failures are errors. Saving an artifact alone is not reported as successful
prediction validation.

## C++ inference

The C++17 executable uses ONNX Runtime's C++ API, RAII session/tensor lifetimes,
batched float32 input, and probability CSV output. It validates feature order,
numeric values, and tensor dimensions. The simple CSV reader intentionally
accepts only the unquoted numeric format written by preparation.

Install CMake and a C++17 compiler. Download an official matching OS/architecture
[ONNX Runtime C/C++ SDK](https://github.com/microsoft/onnxruntime/releases) outside
this repository. Python's `onnxruntime` wheel is not the C++ development SDK.
Point CMake at the extracted directory containing `include/` and `lib/`:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk
cmake --build build --config Release -j 2

./build/predict \
  --model artifacts/model.onnx \
  --input artifacts/test_features.csv \
  --output artifacts/cpp_predictions.csv \
  --metrics artifacts/cpp_benchmark.json \
  --batch-size 256 --iterations 100 --warmup 10
```

CMake configures the build RPATH on macOS/Linux. On Windows, use
`build/Release/predict.exe`; CMake copies the runtime DLL alongside the executable.
If SDK discovery needs customization, pass `ONNXRUNTIME_INCLUDE_DIR` and
`ONNXRUNTIME_LIBRARY` explicitly. No SDK is downloaded by the build.

## Cross-runtime validation

```bash
python -m benchmarks.compare_runtimes --cpp-executable build/predict
```

This runs each available runtime over the same held-out test rows and writes
`results/benchmark.csv` plus environment/methodology metadata in
`results/benchmark.json`. Columns include runtime, status, ROC-AUC, maximum/mean
absolute error, mean/p50/p95 latency, throughput, and batch size. C++ predictions
are generated and compared in that same invocation, not imported from a stale
benchmark file. Missing optional runtimes get blank metrics and an explanation.
Execution failures and parity violations produce a nonzero exit status.

The probability tolerance is **max absolute error < 1e-5** in float32. Tests use
deterministic untrained weights and multiple batch sizes; the export and runtime
comparison commands separately check the actual trained checkpoint. Tests also
check cutoff boundary conditions and refund availability.

## Benchmark methodology

Timing uses `perf_counter_ns` in Python and `std::chrono::steady_clock` in C++.
Defaults: 10 warmups, 100 measured iterations, batch size 256. Each iteration
repeats the same first test batch; throughput is `batch_size / mean_seconds`.
Percentiles use linear interpolation. Model loading, CSV/NPZ I/O, and conversion
are excluded. Tensor setup, embedded preprocessing, forward execution, and output
materialization are included; Python adapter overhead is included in its results.

PyTorch and ONNX Runtime (Python/C++) use one CPU thread. Core ML and LiteRT use
their CPU runtime defaults, so thread budgets are not necessarily identical.
Prediction evaluation pads the final short batch and trims outputs; timing always
uses a full batch. LiteRT must be exported at the requested benchmark batch size.
The native CLI itself also supports a final short batch.

These are warm in-process measurements, not service latency or sustained
concurrent throughput. Repeat on deployment hardware with controlled load before
making production claims. No unmeasured performance values are supplied.

## Quick start

Run from the repository root with Python 3.11 or newer (3.11 recommended for
optional converters):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.generate_data --rows 100000
python -m src.prepare_data
python -m src.train
python -m src.evaluate
python -m export.export_onnx
python -m benchmarks.benchmark_python --runtime onnx
python -m pytest -q
python -m benchmarks.compare_runtimes
```

The generator refuses to overwrite a nonempty output directory. Use a new
`--output` directory for another experiment and pass it to preparation. Training
and export replace their own outputs in the selected artifact directory.

For Core ML on macOS:

```bash
python -m pip install -e '.[coreml]'
python -m export.export_coreml
python -m benchmarks.compare_runtimes
```

For LiteRT, use a separate Linux/Python 3.11 environment and copy the **same**
checkpoint, metadata, and test matrices into its `artifacts/` directory:

```bash
python -m pip install -e '.[test,litert]'
python -m export.export_litert --batch-size 256
python -m export.export_onnx
python -m benchmarks.compare_runtimes --batch-size 256
```

Core ML Tools 9 warns that PyTorch 2.10 is newer than its tested version (2.7);
this project's local conversion/parity run succeeded. Its warning about the
scikit-learn converter does not affect this direct PyTorch conversion.
Dependency ranges are intentionally bounded, but this is not a full environment
lock; rerun parity when changing versions.

For the larger experiment, keep outputs separate:

```bash
python -m src.generate_data --rows 5000000 --output data/large/raw
python -m src.prepare_data --data data/large/raw --artifacts artifacts/large --memory-limit 2GB
python -m src.train --artifacts artifacts/large
python -m src.evaluate --artifacts artifacts/large
python -m export.export_onnx --artifacts artifacts/large
python -m benchmarks.compare_runtimes --artifacts artifacts/large --output results/large.csv
```

## Repository structure

```text
src/          data generation, SQL execution, model, training, evaluation
sql/          temporal feature engineering and cohort analytics
export/       ONNX, Core ML, LiteRT exports and parity checks
benchmarks/   Python timing and cross-runtime comparison
cpp/          CMake project and native ONNX Runtime inference
tests/        model, ONNX parity, and temporal leakage tests
data/         generated Parquet inputs (ignored)
artifacts/    matrices, metadata, checkpoints, exports (ignored)
results/      generated benchmark outputs (ignored)
```

## Current status

The initial 100,000-transaction run completed generation, SQL preparation,
15-epoch training, held-out evaluation, ONNX export, and Python ONNX inference.
The 5,000-row test set achieved ROC-AUC **0.83393** and average precision
**0.86491**. ONNX maximum probability error was **1.79e-7**. Core ML conversion
and native prediction also ran successfully, with maximum error **1.19e-7** on
256 test rows. These are synthetic-task measurements, not general benchmarks.

The C++17 executable compiled with AppleClang and ONNX Runtime 1.30.0, generated
5,000 predictions, and passed comparison with PyTorch (maximum error **1.79e-7**).
All **9 tests passed**, including the SQL temporal-boundary checks.

The cross-runtime command measured PyTorch, ONNX Runtime, native C++, and Core ML
on the complete 5,000-row test set; all exported probabilities passed the 1e-5
check. It wrote actual latency/throughput measurements to the ignored results
files. LiteRT's row remained unmeasured.

The **5-million-transaction experiment also completed** generation, SQL
preparation, 15-epoch training, evaluation, ONNX export, and Python/C++ parity.
It used 750,000 training snapshots, 250,000 validation snapshots, and 250,000 test
snapshots. Test ROC-AUC was **0.83124**, average precision **0.85944**, and maximum
ONNX probability error **1.79e-7**. Core ML was exercised on the smaller checkpoint,
not on this second model.

Local validation used Python 3.11, PyTorch 2.10.0, DuckDB 1.5.6, NumPy 2.4.6,
ONNX 1.23.0, ONNX Runtime 1.30.0, and Core ML Tools 9.0 on macOS ARM64.
LiteRT conversion has not run locally because the documented converter target
is Linux. The full three-runtime portability question remains open until that
conversion and prediction comparison run successfully.

## Next steps

- Execute and pin the LiteRT Linux conversion environment; compare the same checkpoint.
- Measure cold starts, batch-size sweeps, and deployment-device thread settings.
- Add an interpretable baseline and test temporal drift/unseen customer cohorts.
- Evaluate quantization as a separate experiment with explicit fidelity budgets.

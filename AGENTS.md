# Project guidance

Keep this a small, readable ML systems prototype. Prefer functions and direct
runtime APIs; avoid extra service layers, frameworks, notebooks, or boilerplate.

## Architecture and contracts

- Chunked synthetic Parquet transactions feed DuckDB SQL customer snapshots.
- `sql/create_features.sql` owns feature/label semantics. History is strictly
  before cutoff; labels span the following 30 days. Refund availability matters.
- Temporal splits have non-overlapping label windows. Fit scalers on train only;
  select epochs with validation, and reserve test for final measurement.
- `ConversionMLP` produces logits; `Predictor` embeds scaling and sigmoid.
- Every exported runtime accepts raw float32 `[batch, features]` and returns
  probabilities `[batch, 1]`. Preserve feature order and preprocessing exactly.
- C++ loads the ONNX export and validates numeric CSV headers against its metadata.
  Python runtime comparison invokes C++ and checks its returned probabilities.
- Float32 probability parity must satisfy max absolute error < 1e-5. Never hide
  numerical failures as optional runtime skips. Re-export after checkpoint changes.

## Development

Use Python type hints, small functions, and descriptive names. Use C++17, RAII,
and standard containers. Keep CPU benchmarks synchronous and clearly scoped.
Optional SDKs must fail with actionable messages. Never fabricate conversion
success, prediction checks, or benchmark results.

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.generate_data --rows 100000
python -m src.prepare_data
python -m src.train
python -m src.evaluate
python -m export.export_onnx
python -m pytest -q
python -m export.export_coreml   # optional coreml extra; prediction needs macOS
python -m export.export_litert   # optional litert extra; conversion needs Linux
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/sdk
cmake --build build --config Release
python -m benchmarks.compare_runtimes --cpp-executable build/predict
```

Generated data belongs in `data/`; prepared matrices, metadata, checkpoints,
exports, and native outputs in `artifacts/`; benchmark reports in `results/`.
Keep generated datasets, models, build directories, environments, SDKs, machine
paths, and credentials out of git. Preserve the README/.gitkeep exceptions.
Run the core pipeline and tests before committing changes to its contracts.
Document exactly which optional runtimes were actually executed.

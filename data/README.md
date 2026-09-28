# Generated data

Run `python -m src.generate_data --rows 100000` from the repository root.
`raw/` will contain `customers.parquet`, partitioned transaction Parquet files,
and a generation manifest. It contains synthetic data only.

Transactions cover 360 days; customer snapshots use 180 days of history and
30-day future purchase labels. See the root README and `sql/create_features.sql`
for cutoff and refund-availability semantics. Existing generator directories
are not overwritten. Use `--output data/another-run` to start a separate dataset.

Everything here except this README is ignored by git. Do not commit datasets.

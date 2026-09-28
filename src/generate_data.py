"""Generate bounded-memory Parquet transactions with persistent customer preferences."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START = np.datetime64("2024-01-01", "s")
DAYS = 360


def write_parquet(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path) -> None:
    connection.register("chunk", frame)
    connection.execute("COPY chunk TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    connection.unregister("chunk")


def generate(rows: int, output: Path, seed: int = 42, chunk_size: int = 100_000) -> None:
    if rows < 1_000 or chunk_size < 1:
        raise ValueError("Use at least 1,000 transactions and a positive chunk size.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new --output directory.")
    (output / "transactions").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    customers = max(100, rows // 20)
    # Heavy-tailed customer activity remains predictive of future purchases.
    activity = rng.lognormal(0, 1.35, customers)
    probabilities = activity / activity.sum()
    order_value = rng.lognormal(3.7, 0.6, customers)
    online_rate = rng.beta(3, 2, customers)
    discount_rate = rng.beta(2, 5, customers)
    signup = START - rng.integers(1, 1_200, customers).astype("timedelta64[D]")
    favorite_category = rng.integers(0, 6, customers)
    categories = np.array(["grocery", "home", "fashion", "electronics", "beauty", "sports"])
    with duckdb.connect() as connection:
        write_parquet(connection, pd.DataFrame({
            "customer_id": np.arange(customers), "signup_timestamp": signup,
        }), output / "customers.parquet")
        for part, offset in enumerate(range(0, rows, chunk_size)):
            size = min(chunk_size, rows - offset)
            customer = rng.choice(customers, size=size, p=probabilities)
            # Sampling a fixed number of marked events approximates heterogeneous
            # Poisson purchasing; labels will be derived from actual future events.
            timestamp = START + rng.integers(0, DAYS * 86400, size).astype("timedelta64[s]")
            discount = rng.random(size) < discount_rate[customer]
            refunded = rng.random(size) < 0.06
            refund_timestamp = timestamp + rng.integers(1, 22, size).astype("timedelta64[D]")
            refund_timestamp[~refunded] = np.datetime64("NaT")
            category = np.where(rng.random(size) < 0.7, favorite_category[customer],
                                rng.integers(0, 6, size))
            frame = pd.DataFrame({
                "customer_id": customer,
                "transaction_id": np.arange(offset, offset + size),
                "timestamp": timestamp,
                "amount": np.round(order_value[customer] * rng.lognormal(0, 0.4, size)
                                   * np.where(discount, 0.85, 1.0), 2),
                "category": categories[category],
                "discount_used": discount,
                "refunded": refunded,
                "refund_timestamp": refund_timestamp,
                "channel": np.where(rng.random(size) < online_rate[customer], "online", "store"),
                "customer_tenure_days": ((timestamp - signup[customer])
                                         / np.timedelta64(1, "D")).astype(int),
            })
            write_parquet(connection, frame, output / "transactions" / f"part-{part:05d}.parquet")
    manifest = {"rows": rows, "customers": customers, "seed": seed, "chunk_size": chunk_size,
                "start": str(START), "observation_days": DAYS,
                "observation_end": str(START + np.timedelta64(DAYS, "D"))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {rows:,} transactions for {customers:,} customers in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    generate(args.rows, args.output, args.seed, args.chunk_size)


if __name__ == "__main__":
    main()

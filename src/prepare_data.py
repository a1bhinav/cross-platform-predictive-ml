"""Execute temporal feature SQL in DuckDB; materialize compact model matrices."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET = "will_purchase_next_30d"


def prepare(raw: Path, output: Path, memory_limit: str = "2GB") -> None:
    manifest = json.loads((raw / "manifest.json").read_text())
    if manifest["start"] != "2024-01-01T00:00:00" or manifest["observation_days"] < 330:
        raise ValueError("SQL cutoffs require coverage from 2024-01-01 through day 330.")
    output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(output / "features.duckdb")) as connection:
        connection.execute("SET memory_limit = ?", [memory_limit])
        connection.execute("SET temp_directory = ?", [str(output / "duckdb_tmp")])
        connection.read_parquet(str(raw / "transactions" / "*.parquet")).create_view("transactions")
        connection.read_parquet(str(raw / "customers.parquet")).create_view("customers")
        connection.execute((ROOT / "sql/create_features.sql").read_text())
        columns = [row[0] for row in connection.execute("DESCRIBE features").fetchall()]
        names = [name for name in columns if name not in {"customer_id", "cutoff", "split", TARGET}]
        summary = {}
        for split in ("train", "validation", "test"):
            data = connection.execute(
                "SELECT * FROM features WHERE split = ? ORDER BY cutoff, customer_id", [split]
            ).fetchnumpy()
            x = np.column_stack([data[name] for name in names]).astype(np.float32)
            y = data[TARGET].astype(np.float32)
            if not np.isfinite(x).all() or len(np.unique(y)) != 2:
                raise ValueError(f"{split} needs finite features and both target classes; use more rows.")
            np.savez_compressed(output / f"{split}.npz", x=x, y=y,
                                customer_id=data["customer_id"], cutoff=data["cutoff"])
            summary[split] = {"rows": len(y), "positive_rate": float(y.mean()),
                              "first_cutoff": str(data["cutoff"].min()),
                              "last_cutoff": str(data["cutoff"].max())}
            if split == "test":
                # Nine significant digits round-trip float32, including in C++.
                np.savetxt(output / "test_features.csv", x, delimiter=",", fmt="%.9g",
                           header=",".join(names), comments="")
        connection.execute((ROOT / "sql/analytics.sql").read_text())
    metadata = {"feature_names": names, "target": TARGET, "splits": summary,
                "history_days": 180, "label_days": 30, "generator": manifest}
    (output / "data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--memory-limit", default="2GB")
    args = parser.parse_args()
    prepare(args.data, args.artifacts, args.memory_limit)


if __name__ == "__main__":
    main()

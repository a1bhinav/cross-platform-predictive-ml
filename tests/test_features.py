"""Boundary tests guard the temporal and refund information contracts."""

from pathlib import Path

import duckdb
import pytest


def test_cutoff_labels_and_refund_availability():
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE customers AS SELECT 1 AS customer_id, TIMESTAMP '2023-01-01' AS signup_timestamp")
        connection.execute("""
            CREATE TABLE transactions AS SELECT * FROM (VALUES
              (1, 1, TIMESTAMP '2024-06-28', 10.0, 'home', false, true, TIMESTAMP '2024-07-01', 'online'),
              (1, 2, TIMESTAMP '2024-06-29', 999.0, 'home', true, false, NULL, 'online'),
              (1, 3, TIMESTAMP '2024-07-29', 777.0, 'home', true, false, NULL, 'store')
            ) t(customer_id, transaction_id, timestamp, amount, category, discount_used,
                refunded, refund_timestamp, channel)
        """)
        sql = Path(__file__).resolve().parents[1] / "sql/create_features.sql"
        connection.execute(sql.read_text())
        row = connection.execute("""
            SELECT transaction_count, total_spend, refund_rate, will_purchase_next_30d
            FROM features WHERE cutoff = TIMESTAMP '2024-06-29'
        """).fetchone()
        assert row == (1, 10.0, 0.0, 1)
        # Removing the exact-cutoff purchase leaves only an event at label end:
        # that event must not count as a positive label for June 29.
        connection.execute("DELETE FROM transactions WHERE transaction_id = 2")
        connection.execute(sql.read_text())
        assert connection.execute("SELECT will_purchase_next_30d FROM features WHERE cutoff = TIMESTAMP '2024-06-29'").fetchone()[0] == 0
        assert connection.execute("SELECT refund_rate FROM features WHERE cutoff = TIMESTAMP '2024-07-29'").fetchone()[0] == pytest.approx(1.0)

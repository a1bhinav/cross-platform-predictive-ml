-- Run against the DuckDB database produced by src.prepare_data.
SELECT split, cutoff, COUNT(*) AS customers,
       AVG(will_purchase_next_30d) AS conversion_rate,
       AVG(transaction_count) AS mean_transactions,
       AVG(total_spend) AS mean_spend
FROM features GROUP BY split, cutoff ORDER BY cutoff;

SELECT split,
       CASE WHEN transactions_30d = 0 THEN 'inactive' ELSE 'active' END AS segment,
       COUNT(*) AS customers, AVG(will_purchase_next_30d) AS conversion_rate
FROM features GROUP BY split, segment ORDER BY split, segment;

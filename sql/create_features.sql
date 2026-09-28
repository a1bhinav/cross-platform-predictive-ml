-- Half-open history [cutoff - 180d, cutoff); label [cutoff, cutoff + 30d).
-- The five cutoffs have disjoint label windows. Earlier labels mature before
-- the next split begins. Customer IDs never enter the model.
CREATE OR REPLACE TABLE features AS
WITH cutoffs AS (
    SELECT TIMESTAMP '2024-01-01' + day_offset * INTERVAL '1 day' AS cutoff,
           split
    FROM (VALUES (180, 'train'), (210, 'train'), (240, 'train'),
                 (270, 'validation'), (300, 'test')) AS dates(day_offset, split)
), history AS (
    SELECT c.customer_id, d.cutoff, d.split,
        COALESCE(date_diff('second', MAX(t.timestamp), d.cutoff) / 86400.0, 180.0) AS recency_days,
        COUNT(t.transaction_id) AS transaction_count,
        COUNT(t.transaction_id) FILTER (WHERE t.timestamp >= d.cutoff - INTERVAL '30 day') AS transactions_30d,
        COUNT(t.transaction_id) FILTER (WHERE t.timestamp >= d.cutoff - INTERVAL '90 day') AS transactions_90d,
        COALESCE(SUM(t.amount), 0) AS total_spend,
        COALESCE(AVG(t.amount), 0) AS avg_order_value,
        COALESCE(MAX(t.amount), 0) AS max_order_value,
        -- Refund status is only observable after the refund actually happened.
        COALESCE(AVG(CASE WHEN t.transaction_id IS NULL THEN NULL
                         WHEN t.refunded AND t.refund_timestamp < d.cutoff THEN 1.0
                         ELSE 0.0 END), 0) AS refund_rate,
        COALESCE(AVG(CAST(t.discount_used AS DOUBLE)), 0) AS discount_rate,
        COUNT(DISTINCT t.category) AS category_diversity,
        COALESCE(AVG(CAST(t.channel = 'online' AS DOUBLE)), 0) AS online_transaction_ratio,
        date_diff('day', c.signup_timestamp, d.cutoff) AS customer_tenure_days
    FROM customers c CROSS JOIN cutoffs d
    LEFT JOIN transactions t ON t.customer_id = c.customer_id
        AND t.timestamp >= d.cutoff - INTERVAL '180 day' AND t.timestamp < d.cutoff
    WHERE c.signup_timestamp < d.cutoff
    GROUP BY c.customer_id, c.signup_timestamp, d.cutoff, d.split
), labels AS (
    SELECT t.customer_id, d.cutoff, 1 AS will_purchase_next_30d
    FROM transactions t JOIN cutoffs d
        ON t.timestamp >= d.cutoff AND t.timestamp < d.cutoff + INTERVAL '30 day'
    GROUP BY t.customer_id, d.cutoff
)
SELECT h.*, COALESCE(l.will_purchase_next_30d, 0) AS will_purchase_next_30d
FROM history h LEFT JOIN labels l USING (customer_id, cutoff);


-- Q1: WHERE and HAVING
-- Customers spending more than INR 8000
-- on completed orders individually above INR 500

SELECT customer_id,
       SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Three-table JOIN
-- Completed revenue by city and category

SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Scalar subquery
-- Products above average catalogue price

SELECT product_id,
       product_name,
       category,
       price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: LEFT JOIN
-- Customers without completed orders

SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Common Table Expressions
-- Customers spending above average

WITH customer_totals AS (
    SELECT customer_id,
           SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
overall_average AS (
    SELECT AVG(total_spend) AS avg_customer_spend
    FROM customer_totals
)
SELECT ct.customer_id,
       ROUND(ct.total_spend, 2) AS total_spend,
       ROUND(a.avg_customer_spend, 2) AS avg_customer_spend
FROM customer_totals AS ct
CROSS JOIN overall_average AS a
WHERE ct.total_spend > a.avg_customer_spend
ORDER BY ct.total_spend DESC;


-- Q6: Ranking window functions
-- Rank customers by spending within cities

WITH city_spend AS (
    SELECT c.city,
           c.customer_id,
           SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city,
       customer_id,
       ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (
           PARTITION BY city
           ORDER BY total_spend DESC
       ) AS city_rank,
       DENSE_RANK() OVER (
           PARTITION BY city
           ORDER BY total_spend DESC
       ) AS dense_rank,
       ROW_NUMBER() OVER (
           PARTITION BY city
           ORDER BY total_spend DESC
       ) AS row_number
FROM city_spend
ORDER BY city, city_rank, customer_id;


-- Q7: Running total
-- Daily completed revenue and cumulative revenue

WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2) AS daily_revenue,
       ROUND(
           SUM(revenue) OVER (
               ORDER BY day
               ROWS BETWEEN UNBOUNDED PRECEDING
               AND CURRENT ROW
           ), 2
       ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;

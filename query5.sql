--Revenue Trend Over Time
SELECT Year, SUM(Revenue) AS total_revenue
FROM iot_data
GROUP BY Year
ORDER BY Year ASC;

--Year-over-Year Growth per Company
SELECT Year, Company, AVG([YoY Growth]) AS avg_growth
FROM iot_data
GROUP BY Year, Company
ORDER BY Year ASC, avg_growth DESC;

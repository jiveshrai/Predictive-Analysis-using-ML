--Total Units Sold by Company
SELECT Company, SUM([Units Sold]) AS total_units
FROM iot_data
GROUP BY Company
ORDER BY total_units DESC;

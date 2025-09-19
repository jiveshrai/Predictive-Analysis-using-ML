-- Top Companies by Average Market Share
SELECT Company, AVG([Market Share]) AS avg_market_share
FROM iot_data
GROUP BY Company
ORDER BY avg_market_share DESC;

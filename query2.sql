--Top Companies by Revenue in the Latest Year
SELECT Company, Revenue
FROM iot_data
WHERE Year = (SELECT MAX(Year) FROM iot_data)
ORDER BY Revenue DESC;

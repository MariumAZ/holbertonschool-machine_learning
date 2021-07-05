--select rows
SELECT city, AVG(value) as avg_temp FROM temperatures group by city order avg_temp DESC;


--  displays the average temperature (Fahrenheit) by city ordered by temperature (descending).
select city, AVG(value) as avg_temp from temperatures Group by city order by avg_temp DESC;

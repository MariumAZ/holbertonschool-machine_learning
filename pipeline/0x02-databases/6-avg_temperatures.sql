-- temperatures 
--select rows from temperatures

SELECT AVG(value) 
FROM temperatures 
group by city;


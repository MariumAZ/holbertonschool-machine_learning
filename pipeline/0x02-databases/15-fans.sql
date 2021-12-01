-- ranks country origins of bands, ordered by the number of (non-unique) fans
select origin, SUM(fans) as nb_fans
from metal_bands
group by origin
order by nb_fans DESC;

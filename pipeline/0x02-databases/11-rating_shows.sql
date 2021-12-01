-- lists all shows from hbtn_0d_tvshows_rate by their rating.$
select title, SUM(rate) as rating
from tv_shows tv_s, tv_show_ratings tv_s_r 
where tv_s.id=tv_s_r.show_id
group by title
order by rating DESC;
-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

select name as genre, count(*) as number_of_shows
from tv_genres tv_g, tv_show_genres tv_s_g
where tv_g.id=tv_s_g.genre_id
group by genre
order by number_of_shows DESC;


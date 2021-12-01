--  lists all shows contained in hbtn_0d_tvshows without a genre linked.
SELECT title, genre_id 
FROM tv_shows tv_s
LEFT JOIN tv_show_genres tv_s_g ON tv_s.id=tv_s_g.show_id
LEFT JOIN tv_genres tv_g ON tv_g.id=tv_s_g.genre_id
WHERE tv_s_g.genre_id IS NULL
ORDER BY title, genre_id ;
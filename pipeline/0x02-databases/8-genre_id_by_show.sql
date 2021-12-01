-- shows contained in hbtn_0d_tvshows that have at least one genre linked.
SELECT title, genre_id FROM tv_genres tv_g, tv_show_genres tv_s_g, tv_shows tv_s
WHERE tv_g.id=tv_s_g.genre_id AND tv_s.id=tv_s_g.show_id
ORDER BY title, genre_id;
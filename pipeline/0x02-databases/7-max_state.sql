-- displays the max temperature of each state (ordered by State name).

select state, max(value)as max_temp from temperatures Group by state ;
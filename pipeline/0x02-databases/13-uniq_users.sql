-- SQL script that creates a table users

create table if not exists users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL UNIQUE,
    name VARCHAR(256),
    PRIMARY KEY (id)
)
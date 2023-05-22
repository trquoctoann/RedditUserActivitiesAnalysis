use master;
drop database Web;

CREATE DATABASE Web;
use Web;

create table textClassification
(
	tagId		 int auto_increment primary key,
	category 	 varchar(10),
	descriptions varchar(1000)
);

create table redditData
(
	id int auto_increment primary key,
	post_id	varchar(10),
	descriptions varchar(1000),
	created_utc  datetime,
	source_url varchar(1000),
	post_url varchar(100),
	category varchar(20)
);
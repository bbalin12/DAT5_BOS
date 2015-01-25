--1) Simple SELECT statements
SELECT * FROM Master;
SELECT * FROM Batting;
SELECT * FROM Pitching;
SELECT * FROM Fielding;


--2) SELECT statements designating only one column
SELECT playerID, nameGiven, birthYear FROM Master;
SELECT playerID, yearID, TeamID FROM Batting;

--2A) Giving aliases to columns
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, AB as at_bats, R as Runs, H as Hits 
FROM Batting; 
--question: at what granularity (by player? by game? by year?) is the data? 


--3) Using SELECT and WHERE to limit data
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000; 

--3a) Multiple WHERE conditions
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'; 

--3b) Listed WHERE conditions
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID in( 'aardsda01', 'abbotpa01');


--4) ORDER BY (ASC or DESC)
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY games_as_batter DESC; 
--Quesion: what are the years where this player had the most number of games at bat?

--4a) ORDER BY two columns
--say we want to know games at batter by league
SELECT playerID, yearID, TeamID, lgID as league_id, G as games, G_batting as games_as_batter, 
AB as at_bats, R as Runs, H as Hits 
FROM Batting
WHERE yearID > 2000 AND playerID = 'aardsda01'
ORDER BY league_id ASC, games_as_batter DESC;


--5) JOINs and table aliases
--5A) LEFT JOIN (on) --say we want to pull in the given name of the batter
SELECT b.playerID, b.yearID, b.TeamID, b.lgID as league_id, b.G as games, b.G_batting as games_as_batter, 
b.AB as at_bats, b.R as Runs, b.H as Hits, m.nameGiven 
FROM Batting b
LEFT JOIN Master m on b.playerID = m.playerID
WHERE b.yearID > 2000 AND b.playerID = 'aardsda01'
ORDER BY games_as_batter DESC;
--notice the query is many-on-one, so the same name is repeated.

--here's an example of when there is no data in the other table and you use a left join
SELECT b.playerID, b.yearID,b.teamID, b.G_batting as games_batting, pp.* from Batting b
LEFT JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc

--5B) INNER JOIN
--let's run our last query again, but with inner join.  Notice that there are no more null rows in the query.
SELECT b.playerID, b.yearID,b.teamID, b.G_batting as games_batting, pp.* from Batting b
INNER JOIN PitchingPost pp on b.playerID = pp.playerID
WHERE b.playerID in( 'aardsda01', 'abbotpa01')
and b.yearID > 2000
and b.yearID < 2010
order by b.yearID desc


--6) GROUP BYs -- COUNT, MIN, MAX, SUM
--Let's go back to the Batting table. Say you want to count the number of player IDs per team ID in the table.
--FIRST, let's pull in the team name by using a LEFT JOIN.  Notice I join on two colums as the data in both Teams and Batting is at the year level.
SELECT b.teamID, b.playerID, t.teamID, t.name from Batting b
LEFT JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID;

--now, let's do the counting.
SELECT t.name, COUNT(b.playerID)  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
GROUP BY t.name

-- we get a result, but it's a little ugly. One of the colums is just called COUNT(b.playerID), and the data is in no particular order.
-- use aliases and ORDER BY to make things more readable.   
SELECT t.name, COUNT(b.playerID) as num_players  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
GROUP BY t.name
ORDER BY num_players desc
--notice I took out the teamID columns for the query as they're not necessary.

--also notice that this is for all time.  Say we just want to know the number of players total per team at or after 1950.
SELECT t.name, COUNT(b.playerID) as num_players  from Batting b
LEFT OUTER JOIN Teams t ON t.teamID = b.teamID and t.yearID = b.yearID
WHERE t.yearID >=1950
GROUP BY t.name
ORDER BY num_players desc
--notice I use the greater then or equals sign here.

-- say I want to know the total number of games at bat per player. Notice the undercase 'select'.
-- SQL doesn't care if you write these in uppercase or lowercase.
SELECT b.playerID,  sum(b.G_batting) as total_games_at_bat from Batting b
GROUP BY b.playerID
order by sum(b.G_batting) desc

-- say I want to know the first year each player played
SELECT b.playerID,  min(b.yearID) as rookie_year from Batting b
GROUP BY b.playerID;


--7) DISTINCT
--Notice that playerID repeats in the Batting table.  Say you want to only see the unique player IDs in the table.  
SELECT DISTINCT playerID FROM Batting;
--Then, say you want know how many unique player IDs exist in the table.
SELECT COUNT(DISTINCT playerID) FROM Batting;
--Compare this to the number of all (unique and non-unique player IDs)
SELECT COUNT(playerID) FROM Batting;
--there are many, many more nondistinct entries than distinct ones.


--8) CASE statements
--Say you want to make a new label for players who had 20 or more games batted per year.
SELECT CASE WHEN b.G_batting >=20 THEN 1 ELSE 0 END as many_games_batted , b.*
FROM Batting b;
--notice my use of b.* to select all columns without explicitly naming them. 


--9) Subqueries and IS NOT NULL
-- say you wanted to know the batting information for each player's last year.
-- first, find the last year of each player in the Batting table -- we use the MAX function for this
SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID;

--then, we throw it into a subquery
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear;

-- now I want to only select situations where maxyear is not null -- ie that the max year matches with the year in Batting
-- so, I add to the WHERE clause
select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null;


--10) CREATE TABLE
--say I want to save the max_year data to a new table.  I use the following syntax:
CREATE TABLE lastyear as select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null;

-- once this is done, I can query from the new table as you do with any other:
SELECT * from lastyear;


-- EXCERCISES:  

-- 1) Find me the player with the most at-bats in a single season.
-- 2) Find me the name of the the player with the most at-bats in baseball history.
-- 3) Find the average number of at_bats of players in their rookie season.
-- 4) Find the average number of at_bats of players in their final season for all players born after 1980. 
-- 5) Find the average number of at_bats of Yankees players who began their second season at or after 1980.


Resources:

SQLite homepage: https://www.sqlite.org/index.html
SQLite Syntax: https://www.sqlite.org/lang.html

SQL Tutorials: 
Note: These tutorials are for all flavors of SQL, not just SQLite, so some of the functions may behave differently in SQLite.
SQL tutorial: http://www.w3schools.com/sql/ 
SQLZoo: http://sqlzoo.net/wiki/Main_Page

THEN DO HOW PANDAS CONNECTS TO SQL, READS/WRITES. 

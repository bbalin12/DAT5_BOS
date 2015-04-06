##1 Find the player with the most at-bats in a single season.
## The database in structured such that each number of at bats is coupled to its respective season (year).
## Thus, simply by selecting the player and yearID and sorting batting in descending order...
## The player with ID "rilliji01" is found to have 716 at bats during the year 2007
SELECT playerID, AB, yearID from Batting 
ORDER BY AB DESC

#2 Find the name of the the player with the most at-bats in baseball history.
## We group by the player ID and sum the at bats through as many AB entries as there are for that specific player
## Secondly, in order to find the name of the player (data that is only available in the Master Table), 
## we join the master table to the batting table and "zip" over the playerID index so that the total number of at bats points to the player name via playerID
## The individual with the most at bats in the history of baseball is "Peter Edward" at 14,053 at bats. (around 2k more than hank aaron who came in 2nd) 
SELECT b.playerID, sum(b.AB) as ab, m.nameGiven, m.playerID from Batting b 
LEFT JOIN Master m ON m.playerID == b.playerID
GROUP BY b.playerID
ORDER BY ab DESC

## 3 Find the average number of at_bats of players in their rookie season.
##NOTE: I am making two assumptiona. 1) that a rookie season is the player's first season, and 2) average AB is for that specific year (not one AB for all of bball history)
SELECT playerID, min(yearID), avg(AB) from Batting GROUP BY playerID


## 4 Find the average number of at_bats of players in their final season for all players born after 1980.
##This uses a subquery to get the max years and the corresponding at bats from those years, and takes the average since 1980 for the number of at bats for the respective player's last season
SELECT avg(b.AB) as at_bats_a from Batting b
LEFT OUTER JOIN
(Select playerID, max(yearID) as maxYear from Batting GROUP BY playerID) t1
ON t1.maxYear = b.yearID
AND t1.playerID = b.playerID
WHERE  maxYear > 1980
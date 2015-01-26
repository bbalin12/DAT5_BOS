# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 11:43:49 2015
@author: Bryan
"""
# importing the package for SQLite alongside pandas.
import sqlite3
import pandas

# connect to the baseball database. Notice I am passing the full path
# to the SQLite file.
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')

# creating an object contraining a string that has the SQL query. Notice that
# I am using triple quotes to allow my query to exist on multiple lines.
sql = """select sq1.maxyear, b.* from Batting b
LEFT OUTER JOIN 
(SELECT  playerID, max(yearID) as maxyear from Batting 
GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.maxyear
WHERE  sq1.maxyear is not null"""

# passing the connection and the SQL string to pandas.read_sql.
df = pandas.read_sql(sql, conn)
# NOTE: I can use this syntax for SQLite, but for other flavors of SQL
# (MySQL, PostgreSQL, etc.) you will have to create a SQLAlchemy engine 
# as the connection. More information on SQLAlchemy at http://www.sqlalchemy.org/. 
# Stack Overflow also has some nice examples of how to make this connection.

# closing the connection.
conn.close()

# filling NaNs
df.fillna(0, inplace = True)

# re-opening the connection to SQLite.
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
# writing the table back to the database.
# If the table already exists, I'm opting to replace it.  
df.to_sql('pandas_table', conn, if_exists = 'replace')
# You can also append to the table if it exists 
# with the option if_exists = 'append.'

# closing the connection.
conn.close()

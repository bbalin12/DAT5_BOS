# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:33:30 2015

@author: Bryan
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

# importing numpy and the KNN content in scikit-learn 
# along with SQLite and pandas.
import numpy
import sqlite3
import pandas
from sklearn.neighbors import KNeighborsClassifier

# designating model constants at the top of the file per PEP8 
# see https://www.python.org/dev/peps/pep-0008/
# this is the percent we want to hold out for our cross-validation.
CROSS_VALIDATION_AMOUNT = .2

# connect to the baseball database. Notice I am passing the full path
# to the SQLite file.
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
# creating an object contraining a string that has the SQL query. 
sql = 'SELECT playerID, ballots, votes, inducted FROM HallofFame WHERE yearID <2000;'
# passing the connection and the SQL string to pandas.read_sql.
df = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)             

# creating the derived variable 'percent_of_ballots'
df['percent_of_ballots'] = df.votes / df.ballots
df.head()

# seperate your response variable from your explanatory variables
response_series = df.inducted
explanatory_variables = df[['ballots','votes', 'percent_of_ballots']]

# designating the number of observations we need to hold out.
# notice that I'm rounding down so as to get a whole number. 
holdout_num = round(len(df.index) * CROSS_VALIDATION_AMOUNT, 0)

# creating our training and text indices
test_indices = numpy.random.choice(df.index, holdout_num, replace = False )
train_indices = df.index[~df.index.isin(test_indices)]

# our training set
response_train = response_series.ix[train_indices,]
explanatory_train = explanatory_variables.ix[train_indices,]

# our test set
response_test = response_series.ix[test_indices,]
explanatory_test = explanatory_variables.ix[test_indices,]

# instantiating the KNN classifier, with p=2 for Euclidian distnace
# see http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier for more information.
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
# fitting the data to the training set
KNN_classifier.fit(explanatory_train, response_train) 

# predicting the data on the test set. 
predicted_response = KNN_classifier.predict(explanatory_test)

# calculating accuracy
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
print number_correct / total_in_test_set
## not to shabby, eh?
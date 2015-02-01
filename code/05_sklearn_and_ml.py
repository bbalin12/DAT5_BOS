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
accuracy = number_correct / total_in_test_set
print accuracy* 100

# not to shabby, eh?

######
## now, let's do K-Fold CV.
#####

# first, let's verify that a different train/test split will result in a differetnt test set error by re-sampling our train and test indices. 
test_indices = numpy.random.choice(df.index, holdout_num, replace = False )
train_indices = df.index[~df.index.isin(test_indices)]
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
KNN_classifier.fit(explanatory_train, response_train)  
predicted_response = KNN_classifier.predict(explanatory_test)
number_correct = len(response_test[response_test == predicted_response])
total_in_test_set = len(response_test)
new_accuracy = number_correct / total_in_test_set
print new_accuracy * 100


# let's use 10-fold cross-validation to score our model. 
from sklearn.cross_validation import cross_val_score
# we need to re-instantiate the model 
KNN_classifier = KNeighborsClassifier(n_neighbors=3, p = 2)
# notice that instead of putting in my train and text groups, I'm putting 
# in the entire dataset -- the cross_val_score method automatically splits
# the data. 
scores = cross_val_score(KNN_classifier, explanatory_variables, response_series, cv=10, scoring='accuracy')
# let's print out the accuracy at each itration of cross-validation.
print scores
# now, let's get the average accuracy score. 
mean_accuracy = numpy.mean(scores)
print mean_accuracy * 100
# look at hhow his differs from the previous two accuracies we computed. 
print new_accuracy * 100
print accuracy* 100

# now, let's tune the model for the optimal number of K. 
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,  p = 2)
    scores.append(numpy.mean(cross_val_score(knn, explanatory_variables, response_series, cv=5, scoring='accuracy')))

# plot the K values (x-axis) versus the 5-fold CV score (y-axis)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(k_range, scores)
## so, the optimal value of K appears to be low -- under 5 or so. 

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier( p = 2)
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(explanatory_variables, response_series)

# check the results of the grid search and extract the optial estimator
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
best_oob_score = grid.best_score_
grid.best_params_
Knn_optimal = grid.best_estimator_


## pull in data from 2000 onwards
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
sql = 'SELECT playerID, ballots, votes, inducted FROM HallofFame WHERE yearID >2000;'
df = pandas.read_sql(sql, conn)
conn.close()

df.dropna(inplace = True)             

df['percent_of_ballots'] = df.votes / df.ballots

response_series = df.inducted
explanatory_variables = df[['ballots','votes', 'percent_of_ballots']]

optimal_knn_preds = Knn_optimal.predict(explanatory_variables)

number_correct = len(response_series[response_series == optimal_knn_preds])
total_in_test_set = len(response_series)
accuracy = number_correct / total_in_test_set

## compare actual accurac with the accuracy anticipated by our grid search.
print accuracy* 100
print best_oob_score 



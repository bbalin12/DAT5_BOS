# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 22:49:24 2015

@author: bbalin
"""

import pandas
import sqlite3

con = sqlite3.connect('../data/lahman2013.sqlite')

# open a cursor as we are executing a SQL statement that does 
# not produce a pandas DataFrame
cur = con.cursor()    

# writing the query to simplify creating our response feature.
# notice I have to aggregate at the player level as players can
# be entered into voting for the Hall of Fame for numerous years
# in a row. 
table_creation_query = """
CREATE TABLE hall_of_fame_inductees as  

select playerID, case when average_inducted = 0 then 0 else 1 end as inducted from (

select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as average_inducted from  HallOfFame hf
where yearid < 2000
group by playerID

) bb;"""

# executing the query
cur.execute(table_creation_query)
# closing the cursor
cur.close()

# now, running our query to obtain data.  Notice I am joining 
# Batting, Pitching, and Fielding.
monster_query = """
select m.nameGiven, m.weight, m.height, m.bats, m.throws, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees hfi 
left outer join master m on hfi.playerID = m.playerID
left outer join 
(
select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
sum(RBI) as total_RBI, sum(CS) as total_caught_stealing, sum(SO) as total_strikeouts, sum(IBB) as total_intentional_walks
from Batting
group by playerID
HAVING max(yearID) > 1950 and min(yearID) >1950 
)
batting on batting.playerID = hfi.playerID
left outer join
(
 select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, 
sum(H) as total_pitching_hits, sum(er) as total_pitching_earned_runs, sum(so) as total_strikeouts, 
avg(ERA) as average_ERA, sum(WP) as total_wild_pitches, sum(HBP) as total_hit_by_pitch, sum(GF) as total_games_finished,
sum(R) as total_runs_allowed
from Pitching
group by playerID
) 
pitching on pitching.playerID = hfi.playerID 
LEFT OUTER JOIN
(
select playerID, sum(G) as total_games_fielded, sum(InnOuts) as total_time_in_field_with_outs, 
sum(PO) as total_putouts, sum(E) as total_errors, sum(DP) as total_double_plays
from Fielding
group by playerID
) 
fielding on fielding.playerID = hfi.playerID

where batting.playerID is not null
"""

df = pandas.read_sql(monster_query, con)
con.close()

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)

df.head(10)
df.columns

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

# creating binary features for bats and throws
# notice that I'm creating the binary variable as the 'left' option
# why am I doing this?
df['bats_left'] = 0
df.bats_left[df.bats == 'L'] = 1

df['throws_left'] = 0
df.throws_left[df.throws =='L'] = 1

# dropping the categorical Series
df.drop(['throws', 'bats'], 1, inplace = True)

# getting summary statistics on the data
df.describe()


## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['nameGiven', 'inducted']]
explanatory_df = df[explanatory_features]

# dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns

## doing the same for response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]
# if there were any, we need to make sure that we only keep indices 
# that are the union of the explanatory and response features post-dropping.


# imputing NaNs with the mean value for that column.  We will 
# go over this in further detail in next week's class.
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
# fitting the object on our data -- we do this so that we can save the 
# fit for our new data.
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

# create a naive Bayes classifier and get it cross-validated accuracy score. 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

# running a cross-validates score on accuracy.  Notice I set 
# n_jobs to -1, which means I'm going to use all my computer's 
# cores to find the result.
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

# let's see how accurate the model is, on average.
print accuracy_scores.mean()

# let's calculate Cohen's Kappa
mean_accuracy_score = accuracy_scores.mean()
largest_class_percent_of_total = response_series.value_counts(normalize = True)[0]
# largest_class_percent_total is around 89%.  
# So a completely naive model that predicts noone
# is inducted into the Hall of Fame is 89% correct on average.
kappa = (mean_accuracy_score - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa
## kappa is highly negative.  So, if we weigh a positive prediction to be as important as a negative one, our model isn't being that great at predicting at all. 

# calculating F1 score, which is the harmonic mean of specificity
# and sensitivity. 
f1_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

print f1_scores.mean()
# so combined two-class acccuracy doesn't look to good. 


## calculating the ROC area under the curve score. 
roc_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores.mean()

## here's the interpretability of AUC
#.90-1 = excellent 
#.80-.90 = good 
#.70-.80 = fair 
#.60-.70 = poor
#.50-.60 = fail
# so, on AUC terms, this is a fail.

# now, let's create a confusion matrix and plot an ROC curve.
# ideally, we want these to incorporate fully cross-validated
# information, but for the sake of time we're only going to 
# look at one slice.  See http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py for more information on how to really do it. 

## pulling out a training and test slice from the data.
from sklearn.cross_validation import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)

# get predictions on the test slice of the classifier. 
y_predicted = naive_bayes_classifier.fit(xTrain, yTrain).predict(xTest)

# create confusion matrix for the data
cm = pandas.crosstab(yTest, y_predicted, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
# what do the accuracy of predicting not inducted vs inducted look like (recall)
# waht does the accuracy of each prediction look like (precision)

#####
# let's plot ROC curve
#####

## extracting probabilties for the clasifier
y_probabilities = pandas.DataFrame(naive_bayes_classifier.fit(xTrain, yTrain).predict_proba(xTest))

from sklearn import metrics
# remember to pass the ROC curve method the probability of a 'True' 
# class, or column 1 in this case.
fpr, tpr, thresholds = metrics.roc_curve(yTest, y_probabilities[1])
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## looking at the ROC curve, the f1 score,
# ROC score, accuracy, and Kappa how helpful is this estimator? 


###########
## CART ###
###########

# now, let's create some classification trees. 
from sklearn import tree
# Create a decision tree classifier instance
decision_tree = tree.DecisionTreeClassifier(random_state=1)

# realize that the above code is the exact same as the code below,
# which shows the object's default values.  We can change these values
# to tune the tree.
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)


# Fit the decision tree classider
decision_tree.fit(xTrain, yTrain)

## predict on the test data, look at confusion matrix and the ROC curve
predicted_values = decision_tree.predict(xTest)

cm = pandas.crosstab(yTest, predicted_values, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

print cm
# looks a little better, right?

# getting variable iportances
importances_df = pandas.DataFrame(explanatory_colnames)
importances_df['importances'] = decision_tree.feature_importances_


# extracting decision tree probabilities
predicted_probs = pandas.DataFrame(decision_tree.predict_proba(xTest))

# now, let's plot the ROC curve and compare it to Naive Bayes (which will be in green)
fpr_cart, tpr_cart, thresholds_cart = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## which ROC curve looks better? 

## now let's do 10-fold CV, compute accuracy, f1, AUC, and Kappa
accuracy_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_cart.mean()
# mean accuracy of 90% or so. let's compare this to the naive Bayes mean accuracy.
print accuracy_scores.mean()
# so, which model is more accurate? 

# let's calculate Cohen's Kappa
mean_accuracy_score_cart = accuracy_scores_cart.mean()
# recall we already calculated the largest_class_percent_of_total above.
kappa_cart = (mean_accuracy_score_cart - largest_class_percent_of_total) / (1-largest_class_percent_of_total)
print kappa_cart
# so Kappa of 0.096.  What does this say in absolute terms of the 
# ability of the model to predict better than just random selection?

# let's compare to Naive Bayes. 
print kappa
# which is better?

# calculating F1 score, which is the weighted average of specificity
# and sensitivity. 
f1_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

#compare F1 of decision tree and naive bayes
print f1_scores_cart.mean()
print f1_scores.mean()


## calculating the ROC area under the curve score. 
roc_scores_cart = cross_val_score(decision_tree, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# let's compare the decision tree with Naive Bayes.
print roc_scores_cart.mean()
print roc_scores.mean()


# now, let's fine-tune the tree model.
from sklearn.grid_search import  GridSearchCV

# Conduct a grid search for the best tree depth
decision_tree = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)

grid = GridSearchCV(decision_tree, param_grid, cv=10, scoring='roc_auc')
grid.fit(explanatory_df, response_series)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# now let's calculate other accuracy metrics for the best estimator.
best_decision_tree_est = grid.best_estimator_

## lets compare on best accuracy
accuracy_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='accuracy', n_jobs = -1)

print accuracy_scores_best_cart.mean()
print accuracy_scores_cart.mean()
#accuracy scores look identical. So, Cohen's Kappa will be identical. 

# calculating F1 score, which is the weighted average of specificity
# and sensitivity. 
f1_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='f1', n_jobs = -1)

#compare F1 scores
print f1_scores_best_cart.mean()
print f1_scores_cart.mean()
## they're identical

## calculating the ROC area under the curve score. 
roc_scores_best_cart = cross_val_score(best_decision_tree_est, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_best_cart.mean()
print roc_scores_cart.mean()

# Now let's plot the ROC curve of the  best grid estimator vs 
# our older decision tree classifier.
predicted_probs_cart_best = pandas.DataFrame(best_decision_tree_est.predict_proba(xTest))

fpr_cart_best, tpr_cart_best, thresholds_cart_best = metrics.roc_curve(yTest, predicted_probs[1])
plt.figure()
plt.plot(fpr, tpr, color = 'g')
plt.plot(fpr_cart, tpr_cart, color = 'b')
plt.plot(fpr_cart_best, tpr_cart_best, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

## what does this tell us? 

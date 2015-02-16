# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:43:37 2015

@author: Bryan
"""
import pandas
import sqlite3

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)

# first, let's create a categorical feature that shows the dominant team 
# played per player
con = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
query = 'select playerID, teamID from Batting'
df = pandas.read_sql(query, con)
con.close()

# use pandas.DataFrame.groupby and an annonymous lambda function
# to pull the mode team for each player
majority_team_by_player = df.groupby(['playerID']).agg(lambda x:x.value_counts().idxmax())

## write the data back to the database
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
majority_team_by_player.to_sql('dominant_team_per_player', conn, if_exists = 'replace')
# closing the connection.
conn.close()

## using the new table as part of the monster query from last class
monster_query = """
select m.nameGiven, d.teamID, m.weight, m.height, m.bats, m.throws, hfi.inducted, batting.*, pitching.*, fielding.* from hall_of_fame_inductees hfi 
left outer join master m on hfi.playerID = m.playerID
left outer join 
(
select playerID,  sum(AB) as total_at_bats, sum(H) as total_hits, sum(R) as total_runs, sum(HR) as total_home_runs, sum(SB) as total_stolen_bases,
sum(RBI) as total_RBI, sum(CS) as total_caught_stealing, sum(SO) as total_hitter_strikeouts, sum(IBB) as total_intentional_walks
from Batting
group by playerID
HAVING max(yearID) > 1950 and min(yearID) >1950 
)
batting on batting.playerID = hfi.playerID
left outer join
(
 select playerID, sum(G) as total_games_pitched, sum(SO) as total_shutouts, sum(sv) as total_saves, sum(IPouts) as total_outs_pitched, 
sum(H) as total_pitching_hits, sum(er) as total_pitching_earned_runs, sum(so) as total_pitcher_strikeouts, 
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

LEFT OUTER JOIN dominant_team_per_player d on d.playerID = hfi.playerID
where batting.playerID is not null
"""

con = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(monster_query, con)
con.close()

## getting an intial view of the data for validation
df.head(10)
df.columns

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

#############
## repeating the same preprocessing from the previous lesson
############

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

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

######
# now, let's find if any of the categorical features need 'binnng'
#####
# first, fill the NANs in the feature (this lets us see if there are features
# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')

# lets' create the heuristic that a level in the feature must exist in more
# than 1% of the training data to be retained. 
for col in string_features:
    # get the value_count of the column
    sizes = string_features[col].value_counts(normalize = True)
    # get the names of the levels that make up less than 1% of the dataset
    values_to_delete = sizes[sizes<0.01].index
    string_features[col].ix[string_features[col].isin(values_to_delete)] = "Other"

# let's verify if the replacement happened
string_features.teamID.value_counts(normalize = True)

## let's wrap that in a function for re-use 
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return col
##

######
## now, let's encode the categorical features.
######
# creating the 'catcher' data frame that will hold the encoded data
encoded_data = pandas.DataFrame(index = string_features.index)
for col in string_features.columns:
    ## calling pandas.get_dummies to turn the column into a sequene of 
    ## binary variables. Notice I'm using the 'prefix' feature to include the 
    ## original name of the column
    data = pandas.get_dummies(string_features[col], prefix=col.encode('ascii', 'replace'))
    encoded_data = pandas.concat([encoded_data, data], axis=1)

# let's verify that the encoding occured.
encoded_data.head()

## let's also wrap this into a function.
def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns


## now, let's fill the NANs in our nuemeric features.
# as before, let's impute using the mean strategy.
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## now that we've encoded our qualitative variables and filled the NaNs in our numeric variables, let's merge both DataFrames back together.

explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()

#now, let's find features with no variance 
toKeep = []
toDelete = []
## loop through the DataFrame's columns
for col in explanatory_df:
    ## if the value_counts method returns more than one uniqe entity,
    ## append the column name to 'toKeep'
    if len(explanatory_df[col].value_counts()) > 1:
        toKeep.append(col)
    ## if not, append to 'toDelete'.
    else:
        toDelete.append(col)

# let's see if there's zero variance in an features
print toKeep
print toDelete
# doesn't look like it.

## let's wrap this into a function for future use. 
def find_zero_var(df):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """   
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            toKeep.append(col)
        else:
            toDelete.append(col)
        ##
    return {'toKeep':toKeep, 'toDelete':toDelete} 
##

########
# now, let's look for columns with perfect correlation
#######

# first, let's create a correlation matrix diagram for the first 26 features.
toChart = explanatory_df.ix[:,0:25].corr()
toChart.head()

import matplotlib.pyplot as plt
import numpy
plt.pcolor(toChart)
plt.yticks(numpy.arange(0.5, len(toChart.index), 1), toChart.index)
plt.xticks(numpy.arange(0.5, len(toChart.columns), 1), toChart.columns, rotation=-90)
plt.colorbar()
plt.show()
# if you want to be audacious, try plotting the entire dataset.

# let's use an automated method to see what's perfectly correlated,
# either positively or negatively.
corr_matrix = explanatory_df.corr()
# substitude the entire matrix for a triangular matrix for faster
# computation
corr_matrix.ix[:,:] =  numpy.tril(corr_matrix.values, k = -1)
## create catcher objects to find lists of what is perfectly correlated
already_in = set()
result = []
for col in corr_matrix:
    perfect_corr = corr_matrix[col][abs(numpy.round(corr_matrix[col],10)) == 1.00].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)
# notice that throws R and throws L are perfectly correlated -- they should  be.
print result
# creating a list of what to remove as all but the first column to appear
# in each correlation grouping.
toRemove = []
for item in result:
    toRemove.append(item[1:(len(item)+1)])
# flattenign the list of lists
toRemove = sum(toRemove, [])

#now, let's drop the columns we've identified from our explanatory features. 
explanatory_df.drop(toRemove, 1, inplace = True)

# let's combine all of this into a nice function.
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
###

##############
# scaling data
#############
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


########
# Imputing missing values
#######
# recall that we used a 'mean' strategy for imputation before. This created some strange results for our values.  So, let's try out another method.
from sklearn.preprocessing import Imputer
## re-creating the numeric_features dataframe.
numeric_features = df.ix[:, df.dtypes != 'object']
## inputting the median observation
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), 
                                    columns = numeric_features.columns)

########
# Recursive feature elimination
#######
from sklearn.feature_selection import RFECV
from sklearn import tree

# create new class with a .coef_ attribute.
class TreeClassifierWithCoef(tree.DecisionTreeClassifier):
    def fit(self, *args, **kwargs):
        super(tree.DecisionTreeClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

# create your tree based estimator
decision_tree = TreeClassifierWithCoef(criterion = 'gini', splitter = 'best', max_features = None, max_depth = None, min_samples_split = 2, min_samples_leaf = 2, max_leaf_nodes = None, random_state = 1)

## set up the estimator. Score by AUC
rfe_cv = RFECV(estimator=decision_tree, step=1, cv=10,
              scoring='roc_auc', verbose = 1)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))

# printing out scores as we increase the number of features -- the farter
# down the list, the higher the number of features considered.
print rfe_cv.grid_scores_

## let's plot out the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()
# notice you could have just as well have included the 10 most important 
# features and received similar accuracy.

# you can pull out the features used this way:
features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

#you can extract the final selected model object this way:
final_estimator_used = rfe_cv.estimator_

# you can also combine RFE with grid search to find the tuning 
# parameters and features that optimize model accuracy metrics.
# do this by passing the RFECV object to GridSearchCV.
from sklearn.grid_search import  GridSearchCV

# doing this for a small range so I can show you the answer in a reasonable
# amount of time.
depth_range = range(4, 6)
# notice that in param_grid, I need to prefix estimator__ to my paramerters.
param_grid = dict(estimator__max_depth=depth_range)
# notice that this will take quite a bit longer to compute.
rfe_grid_search = GridSearchCV(rfe_cv, param_grid, cv=10, scoring='roc_auc')
rfe_grid_search.fit(explanatory_df, response_series)

print rfe_grid_search.grid_scores_
rfe_grid_search.best_params_

# let's plot out the results.
grid_mean_scores = [score[1] for score in rfe_grid_search.grid_scores_]

plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.plot(rfe_grid_search.best_params_['estimator__max_depth'], rfe_grid_search.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')
plt.grid(True)

# now let's pull out the winning estimator.
best_decision_tree_rfe_grid = rfe_grid_search.best_estimator_
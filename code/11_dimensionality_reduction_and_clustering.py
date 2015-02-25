# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 20:30:10 2015

@author: Bryan
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


query = """
select s.teamID, sum(salary) as total_salaries, sum(R) as total_runs from salaries s
 inner join Batting b on s.playerId = b.playerID 
 where R is not null and s.yearID > 2000
group by s.teamID
order by total_runs desc
"""

con = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
df = pandas.read_sql(query, con)
con.close()

## scale the data
# scaling data
data = df[['total_salaries', 'total_runs']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

# plot the scaled data
plt = df.plot(x='total_salaries', y='total_runs', kind='scatter')

## annotating with team names.
for i, txt in enumerate(df.teamID):
    plt.annotate(txt, (df.total_salaries[i],df.total_runs[i]))
plt.show()

########
# K-Means
########

# plotting the data, it looks like there's 3 clusters -- one 
# 'big' cluster, another of low-preforming teams, and the Yankees.
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters=3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.total_salaries, df.total_runs, s=60, c=labels)
#eh, it's okay.  Not great.

########
# DBSCAN
########

from sklearn.cluster import DBSCAN

## getting around a bug that doesn't let you fit to a dataframe
# by coercing it to a NumPy array.
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.total_salaries, df.total_runs, s=60, c=labels)
# blah, not a ton better.

########
# hirearchical clustering
########
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

## increasing the figure size of my plots for better readabaility.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

distanceMatrix = pdist(data)

## adjust color_threshold to get the number of clusters you want.
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=1, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

# notice I moved the threshold to 4 so I can get 3 clusters. 
dend = dendrogram(linkage(distanceMatrix, method='complete'), 
           color_threshold=4, 
           leaf_font_size=10,
           labels = df.teamID.tolist())

########
###  PCA
########

import sqlite3
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt



# including our functions from last week up here for use. 
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
#

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
#
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


# putting a setting into pandas that lets you print out the entire
# DataFrame when you use the .head() method
pandas.set_option('display.max_columns', None)


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


# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')
# cleaning up string features
string_features = cleanup_data(string_features)
# binarizing string features 
encoded_data = get_binary_values(string_features)
## imputing features
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()


#now, let's find features with no variance 
no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


#######
# PCA
######
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(explanatory_df)

# extracting the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting the first to principal components
pca_df.plot(x = 0, y= 1, kind = 'scatter')


# making a scree plot
variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component': pca_df.columns.tolist()})
# adding one to principal components (since there is no 0th component)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x = 'principal component', y= 'variance')
## looks like variance stops getting explained after the first 
# two principal components.

pca_df_small = pca_df.ix[:,0:1]

## getting cross-val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

print roc_scores_rf_pca.mean()
## 91% accuracy.

# Let's compare this to the original adata
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()
## 95% accuracy - so PCA actually created information LOSS!


#########################
# SUPPORT VECTOR MACHINES
#########################
from sklearn.svm import SVC

## first, running quadratic kernel without PCA
svm = SVC(kernel='poly')

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm.mean()
## 92% acccuracy


# let's try with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_svm_pca.mean()
## 86% acccuracy -- so PCA did worse AGAIN


# let's do a grid search on the optimal kernel
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
svm_grid.fit(explanatory_df, response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
## looks like rbf won out
print svm_grid.best_score_
## best estiamtor was 94% accurate -- so just a hair below RFs.
# Note: remember, SVMs are more accurate than RFs with trending data!
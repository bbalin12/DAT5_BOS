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
from scipy.spatial.distance import pdist,
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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

## retrieve the labels (need to reverse due to the dist matrix)
labels = dend['color_list']
plt.scatter(df.total_salaries, df.total_runs, s=60, c=labels)
df.teamID

fl = fcluster(cl,numclust,criterion='maxclust')

from sklearn.cluster import AgglomerativeClustering
hirear_cluster = AgglomerativeClustering(n_clusters=3)
hirear_cluster.fit(data)
labels = hirear_cluster.labels_
plt.scatter(df.total_salaries, df.total_runs, s=60, c=labels)

## very similar to K-means
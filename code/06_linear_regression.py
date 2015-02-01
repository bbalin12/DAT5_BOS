# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:33:30 2015

@author: Bryan
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas
# importing statsmodels to run the linear regression
# scikit-learn also has a linear model method, but the statsmodels version
# has more user-friendly output.
import statsmodels.formula.api as smf

# connect to the baseball database. 
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
# SQL
sql = """select yearID, sum(R) as total_runs, sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, sum(IBB) as total_intentional_walks
from Batting 
where yearID > 1954
and yearid < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)      

# starting out with the most obvious connection -- more runs means more hits. 
est = smf.ols(formula='total_runs ~ total_hits', data=df).fit()
# now, let's print out the results.
print est.summary()
# notice the R-squared, coefficeints, and p-values. 
# how would you interpret the covariates?

# let's pull out the r-squared
print est.rsquared
# 97%.  Not to shabby! 

# let's create a y-hat column in our dataframe. 
df['yhat'] = est.predict(df)

# now, let's plot how well the model fits the data. 
plt = df.plot(x='total_hits', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat, color='blue',
         linewidth=3)
# looks pretty good.

# let's get a look at the residuals to see if there's heteroskedasticity
df['residuals'] = df.total_runs - df.yhat

plt = df.plot(x='total_hits', y='residuals', kind='scatter')
## there doesn't seem to be any noticable clear heteroskedasticity. 

# let's calculate RMSE -- notice you use two multiply signs for exponenets
# in Python
RMSE = (((df.residuals) ** 2) ** (1/2)).mean()
# so, on average, the model is of by about 651 runs for each observation.

# lets understand the percent by which the model deviates from actuals on average
percent_avg_dev = RMSE / df.total_runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(percent_avg_dev*100, 1))
# looks like in-sample deviation is 4% on average. 


# now, let's see the effects of stolen_bases on toral_ruls -- do we think that,
# on average, more stolen bases equates to more runs? 

# first, let's plot their interaction.
plt = df.plot(x='stolen_bases', y='total_runs', kind='scatter')

# now, let's run some regressions.
sb_est = smf.ols(formula='total_runs ~ stolen_bases', data=df).fit()
print sb_est.summary()
# how would you interpret the covariates? 
# notice the R-squared, coefficeints, and p-values. 

# let's pull out the r-squared
print sb_est.rsquared
# is it better or worse than using hits? 

# let's get the yhat of the stolen_bases regression
df['sb_yhat'] = sb_est.predict(df)

# let's see how the stolen bases model fits the data
plt = df.plot(x='stolen_bases', y='total_runs', kind='scatter')
plt.plot(df.stolen_bases, df.sb_yhat, color='blue',
         linewidth=3)

# let's get a look at the residuals to see if there's heteroskedasticity
df['sb_residuals'] = df.total_runs - df.sb_yhat

plt = df.plot(x='stolen_bases', y='sb_residuals', kind='scatter')
## there doesn't seem to be any noticable clear heteroskedasticity. 

# let's calculate RMSE -- notice you use two multiply signs for exponenets
# in Python
RMSE_sb = (((df.sb_residuals) ** 2) ** (1/2)).mean()

# is RMSE higher or lower for stolen bases? 
print RMSE_sb
print RMSE

# lets understand the percent by which the model deviates from actuals on average
sb_percent_avg_dev = RMSE_sb / df.total_runs.mean()
# notice I'm using string formatting in when I print the results.
print 'average deviation: {0}%'.format(round(sb_percent_avg_dev*100, 1))
# looks like in-sample deviation is 11% on average -- higher than the 4% of 
# hits.

# let's now investigate the relationship of total runs with time.
# First, we encode a dummy feature for years past 1995. 
df['post_1995'] = 0
df.post_1995[df.yearID>1995] = 1
# Now,create a dummy feature for years between 1985 and 1995. 
df['from_1985_to_1995'] = 0
df.from_1985_to_1995[(df.yearID>1985) & (df.yearID<=1995)] = 1
# do we need to create another dummy feaure for all the other years? 

# let's run the formula.
bin_est = smf.ols(formula='total_runs ~ from_1985_to_1995 + post_1995', data=df).fit()
print bin_est.summary()
# interpret the results for me, please.

# lets plot these predictions against actuals
df['binary_yhat'] = bin_est.predict(df)
plt = df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.yearID, df.binary_yhat, color='blue',
         linewidth=3)

# let's combine all three factors together: total hits, stolen bases, and year.
large_est = smf.ols(formula='total_runs ~ total_hits + stolen_bases + from_1985_to_1995 + post_1995', data=df).fit()
print large_est.summary()
# many of the covariates (regressors) have p-values above 0.05.  
# Should we include them? 

large_rsquared = large_est.rsquared
print large_rsquared
print est.rsquared
# so, the large formula rsquared is higher than the rsquared for just hits.
# does this mean that the large formula is more predictive?

## let's caclulate residuals and RMSE. 
df['large_yhat'] = large_est.predict(df)
df['large_residuals'] = df.total_runs - df.large_yhat

RMSE_large = (((df.large_residuals) ** 2) ** (1/2)).mean()

print 'average deviation for large equation: {0}'.format(
                                            round(RMSE_large, 4))

print 'average deviation for just hits: {0}'.format(
                                            round(RMSE, 4))

## RMSe looks better.  Is it really more predictive?
# let's plot the fit of just hits and the full equation.
plt = df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.yearID, df.yhat, color='blue',
         linewidth=3)
plt.plot(df.yearID, df.large_yhat, color='red',
         linewidth=3)

# let's look at data after 2005.
conn = sqlite3.connect('/Users/Bryan/Documents/SQLite/lahman2013.sqlite')
# creating an object contraining a string that has the SQL query. 
sql = """select yearID, sum(R) as total_runs, sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, sum(IBB) as total_intentional_walks
from Batting 
where 
yearid >= 2005
group by yearID
order by yearID ASC"""

# passing the connection and the SQL string to pandas.read_sql.
df_post_2005 = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# re-create the dummy features for the new data.
df_post_2005['post_1995'] = 1
df_post_2005['from_1985_to_1995'] = 0

# let's predict both modes on the post_2005 data.
df_post_2005['yhat'] = est.predict(df_post_2005)
df_post_2005['large_yhat'] = large_est.predict(df_post_2005)

# creating the residuals
df_post_2005['hits_residuals'] = df_post_2005.total_runs - df_post_2005.yhat
df_post_2005['large_residuals'] = df_post_2005.total_runs - df_post_2005.large_yhat

# calculating  RMSE
RMSE_large = (((df_post_2005.large_residuals) ** 2) ** (1/2)).mean()
RMSE_hits =  (((df_post_2005.hits_residuals) ** 2) ** (1/2)).mean()

print 'average deviation for large equation: {0}'.format(
                                            round(RMSE_large, 4))

print 'average deviation for just hits: {0}'.format(
                                            round(RMSE_hits, 4))
# what does this show you?  
# We were OVERFITTING our data with the large equaiton!
                                            
# lets plot how bad the overfit was.
plt = df_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat, color='blue',
         linewidth=3)
plt.plot(df_post_2005.yearID, df_post_2005.large_yhat, color='red',
         linewidth=3)

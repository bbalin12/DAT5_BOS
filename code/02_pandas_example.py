# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 12:40:20 2015

@author: Bryan
"""

# importing pandas
import pandas

# assigning the drinks URL to a string object named data_url
data_url = 'https://raw.githubusercontent.com/bbalin12/DAT5_BOS/master/data/drinks.csv'

# passing data_url to pandas' read_csv method.
# and assigning the result to an object named drinks
drinks = pandas.read_csv(data_url)

# confirming that drinks is a pandas DataFrame
type(drinks)

# inspecing the 'drinks' DataFrame by viewing its first 5 rows
drinks.head()
# inspecing the 'drinks' DataFrame by viewing its last 5 rows
drinks.tail()
# summarizing the columns of the data frame
drinks.describe()

#############################
# INDEXING AND SELECTING DATA
#############################

# get names of columns
drinks.columns

# select just the country column
drinks['country']

# select the country column via a method call
drinks.country

# select the first three rows of the country column
# notice the index starts with zero
drinks['country'][0:3]

# select the country column and the continent column
# notice the double bracets here - we are passing a list of strings 
# to the dataframe to get the two columns
drinks[['country', 'continent']]

# select the first three rows of the country and continent columns
drinks[['country', 'continent']][0:3]

# do the same thing, but with numeric indexes only
drinks.ix[0:3, 0]

# select just the country column with numeric indices only
drinks.ix[:,0]

# select the first thre rows of the country column using boolean
drinks.country[drinks.index < 3]


################################
# ASSIGNING AND REASSIGNING DATA
################################

# arbitrarily create a new column
drinks['light_drinker'] = 0

# look bac at the DataFrame and see the light_drinker column having values
# of all zero. 
drinks.head()
# reassign the first three rows of the 'kight_drinker' column to zero
# notice I can now access beer_servings as a method of the 'drinks' object
drinks.light_drinker[0:3] = 1
# re-inspect the DataFrame.  Notice that there's 1 under light_drinker
# for the first three 
drinks.head()
# reverse the change
drinks.light_drinker[0:3] = 0
# confirm the change
drinks.head()

# show all columns of the drinks DataFrame where the beer servings 
# column equals 1
drinks[drinks.beer_servings == 1]

# just show the beer_sevings column where beer_servings equals 1
drinks.beer_servings[drinks.beer_servings == 1]

# just show the light_drinker column where beer_servings equals 1
drinks.light_drinker[drinks.beer_servings == 1]

# reassign all instances of the light_drinker to 1 if beer_servings == 1
drinks.light_drinker[drinks.beer_servings == 1] = 1

# confirm that that light_drinker now equals 1 when beer_servings == 1
drinks[drinks.light_drinker == 1]

# reassign light_drinker to 1 where beer_servings is less than 2
drinks.light_drinker[drinks.beer_servings < 2] = 1

# re-confirm the change
drinks[drinks.light_drinker == 1]
drinks.country[drinks.light_drinker == 1]

################################
# DESCRIBING AND SUMMARIZING DATA
################################

# Examine the data types of all columns
drinks.dtypes
drinks.info()

# Calculate the average 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# Only show European countries OR countries with 'wine_servings' greater than 300
drinks[(drinks.continent=='EU') | (drinks.wine_servings > 300)]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol').tail(10)

# Determine which country has the highest value for 'beer_servings'
drinks[drinks.beer_servings==drinks.beer_servings.max()].country

# see mean beer servings by continent
drinks.groupby('continent').beer_servings.mean()

# missing values are often just excluded
drinks.describe(include='all')              # excludes missing values
drinks.continent.value_counts(dropna=False) # includes missing values (new in pandas 0.14.1)

# find missing values in a Series
drinks.continent.isnull()           # True if NaN, False otherwise
drinks.continent.notnull()          # False if NaN, True otherwise
drinks[drinks.continent.notnull()]  # only show rows where continent is not NaN
drinks.continent.isnull().sum()     # count the missing values

# find missing values in a DataFrame
drinks.isnull()             # DataFrame of booleans
drinks.isnull().sum()       # calculate the sum of each column

# drop missing values
drinks.dropna()             # drop a row if ANY values are missing
drinks.dropna(how='all')    # drop a row only if ALL values are missing

# fill in missing values
drinks.continent.fillna(value='NA')                 # does not modify 'drinks'
drinks.continent.fillna(value='NA', inplace=True)   # modifies 'drinks' in-place
drinks.fillna(drinks.mean())                        # fill in missing values using mean


################################
# PLOTTING DATA
################################

# bar plot of number of countries in each continent
plt = drinks.continent.value_counts().plot(kind='bar', title='Countries per Continent')
plt.set_xlabel('Continent')
plt.set_ylabel('Count')


# bar plot of average number of beer servings (per adult per year) by continent
drinks.groupby('continent').beer_servings.mean().plot(kind='bar')


# histogram of beer servings (shows the distribution of a numeric column)
drinks.beer_servings.hist(bins=20)
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')

# density plot of beer servings (smooth version of a histogram)
drinks.beer_servings.plot(kind='density', xlim=(0,500))

# grouped histogram of beer servings (shows the distribution for each group)
drinks.beer_servings.hist(by=drinks.continent)
drinks.beer_servings.hist(by=drinks.continent, sharex=True)
drinks.beer_servings.hist(by=drinks.continent, sharex=True, sharey=True)
drinks.beer_servings.hist(by=drinks.continent, layout=(2, 3))   # change layout (new in pandas 0.15.0)

# boxplot of beer servings by continent (shows five-number summary and outliers)
drinks.boxplot(column='beer_servings', by='continent')

# scatterplot of beer servings versus wine servings
drinks.plot(kind='scatter', x='beer_servings', y='wine_servings', alpha=0.3)

# same scatterplot, except all European countries are colored red
import numpy as np
colors = np.where(drinks.continent=='EU', 'r', 'b')
drinks.plot(x='beer_servings', y='wine_servings', kind='scatter', c=colors)

# scatterplot matrix of all numerical columns
pandas.scatter_matrix(drinks)


# -*- coding: utf-8 -*-

##import the packages we're going to use -- scikit-learn (machine learning)
##and pandas (data analysis)
import sklearn
import pandas

###importing the iris dataset
from sklearn import datasets
iris = datasets.load_iris()

##coercing the iris dataset (stored as a scikit-learn object)
## to a Pandas DataFrame.  Included in the iris object are two attirbures -- 
## its data, and the names of the columns of the data.  I pass both 
## to the dataframe.
data = pandas.DataFrame(data = iris.data, columns = iris.feature_names)

##now, let's explore the data. As you can see, printing the data shows you
##what's in the dataframe. 
print data

###say you want to predict which species of flower each observation (ie row)
##that the data you just saw corresponds to.  Luckily enough, this data 
##already exists! 
target = iris.target
print target

##you can see that the species of flower is represented by an integer.  
##the corresponding names for the species can be found here:
print iris.target_names

##ok, let's do some machine learning! 



###dividing the data and the target variable into train and test groups
##remember that Python indices start at zero! 
train_data = data[0:125]
print train_data
##notice I do not put an end to the data I want -- Pandas automatically 
##interprets it as 'the rest' of the data above 125!
test_data = data[125:]
print test_data


train_target = target[0:125]
test_target = target[125:]


###building the classifier -- we'll use a random forest to start. 
#importing the random forest classifier
from sklearn.ensemble import RandomForestClassifier
##creating the random forest classifier object.
random_forest_object = RandomForestClassifier()
##now, fit the object to the training data and the target 'answers' to create
##the meta-structure. 
random_forest_object.fit(train_data, train_target)

##now, let's predict the species of the remaining test data
test_predictions = random_forest_object.predict(test_data)
test_predictions

##now, let's compare the test predictions with what they actually are
test_results = pandas.DataFrame({'predicted':test_predictions, 'actual':test_target})

##let's look at the results -- not to shabby, eh!  We'll go over the statistics
## used to evaluate the predicitve power of the model in the next lession. 

print test_results

test_results.predicted.value_counts()
test_results.predicted.value_counts(normalize = True)

###and volia, you've run your first machine learning algorithm! 
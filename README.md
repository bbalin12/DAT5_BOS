## DAT5 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science) in Boston, MA (20 January 2015 - 07 April 2015). View student work in the [student repository](https://github.com/bbalin12/DAT5_BOS_students).

**Instructor:** Bryan Balin. **Teaching Assistant:** Harish Krishnamurthy.

**Office hours:** Wednesday 5-6pm; Friday 5-6:30pm, [@Boston Public Library](http://www.bpl.org/); as needed Tuesdays and Thursdas @GA. 

**[Course Project information](project.md)**

Tuesday | Thursday
--- | ---
1/20: [Introduction](#class-1-introduction) | 1/22: [Python & Pandas](#class-2-python--pandas)
1/27: [Git and GitHub](#class-3-git-and-github) | 1/29: [SQL](#class-4-sql)
2/3: [Supervised Learning and Model Evaluation](#class-5-supervised-learning-and-model-evaluation) | 2/5: [Linear Regression](#class-6-linear-regression) <br>**Milestone:** Begin Work on Question and Data Set
2/10:  [Logistic Regression](#class-7-logistic-regression)| 2/12: [Naive Bayes, Decision Trees, and Classification Model Evaluation](#class-8-naive-bayes-decision-trees-and-classification-model-evaluation)
2/17: [Data Cleaning and Manipulation](#class-9-data-cleaning-and-manipulation) | 2/19:  Makeup: Logistic Regression
2/24: [Ensemble Methods and Neural Networks](#class-10-neural-networks-and-ensemble-methods)| 2/26: [Unsupervised Learning and Dimensionality Reduction](#class-10-unsupervised-learning-and-dimensionality-reduction) <br>**Milestone:** Data Exploration and Analysis Plan 
3/3:  [Amazon Web Services & Apache Hive](#class-13-amazon-web-services--apache-hive)| 3/5: [Amazon Elastic MapReduce](#class-14-natural-language-processing) 
3/10: [Natural Language Processing](#class-13-recommendation-systems)<br>**Milestone:** First Draft | 3/12: [Advanced scikit-learn](#class-16-advanced-scikit-learn)
3/17: **No Class**  | 3/19: [Web Scraping and Data Retrieval Methods](#class-17-web-scraping-and-data-retrieval-methods)
3/24: [Recommenders](#class-18-recommenders) | 3/26: [Course Review, Companion Tools](#class-19-course-review-companion-tools)<br>**Milestone:** Second Draft (Optional)
3/31: [TBD](#class-20-tbd) | 4/2: [Project Presentations](#class-21-project-presentations)
4/7: [Project Presentations](#class-22-project-presentations) |



### Installation and Setup
* Install the [Anaconda distribution](http://continuum.io/downloads) of Python 2.7x.
* Install [Git](http://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and create a [GitHub](https://github.com/) account.
* Once you receive an email invitation from [Slack](https://slack.com/), join our "datbos05 team" and add your photo!


### Class 1: Introduction
* Introduction to General Assembly
* Course overview: our philosophy and expectations ([slides](slides/01_course_overview.pdf))
* Data science overview ([slides](slides/01_intro_to_data_science.pdf))
* Tools: check for proper setup of Anaconda, overview of Slack

**Homework:**
* Resolve any installation issues before next class.

**Optional:**
* Review the [code](code/00_python_refresher.py) for a recap of some Python basics.
* Read [Analyzing the Analyzers](http://cdn.oreillystatic.com/oreilly/radarreport/0636920029014/Analyzing_the_Analyzers.pdf) for a useful look at the different types of data scientists.
* Check out the [PyData Boston Meetup page](http://www.meetup.com/PyData-Boston/) to become acquainted with the local data community.


### Class 2: Python & Pandas
[slides](slides/02_python_in_data_science.pdf).  [Python refresher code](code/00_python_refresher.py).  [Python code](code/02_python_example.py). [Pandas code](code/02_pandas_example.py).

* Brief overview of Python
* Brief overview of Python environments: Python scripting, IPython interpreter, Spyder
* Working with data in Pandas
    * Loading and viewing data
    * Indexing and selecting data
    * Assigning, reassigning, and splitting data
    * Describing and summarizing data
    * Plotting data

**Homework:**
* Do the [class homework](/homework/02_pandas.md) by Tuesday.
* Read through the [project page](project.md) in detail.
* Review a few [projects from past Data Science courses](https://github.com/justmarkham/DAT-project-examples) to get a sense of the variety and scope of student projects.

**Optional:**
* If you need more practice with Python, review the "Python Overview" section of [A Crash Course in Python](http://nbviewer.ipython.org/gist/rpmuller/5920182), work through some of [Codecademy's Python course](http://www.codecademy.com/en/tracks/python), or work through [Google's Python Class](https://developers.google.com/edu/python/) and its exercises.
* For more project inspiration, browse the [student projects](http://cs229.stanford.edu/projects2013.html) from Andrew Ng's [Machine Learning course](http://cs229.stanford.edu/) at Stanford.

**Resources:**
* [Online Python Tutor](http://pythontutor.com/) is useful for visualizing (and debugging) your code.


### Class 3: Git and GitHub
* [slides](slides/03_git_and_github.pdf)
* [GitHub for Windows](https://windows.github.com/)
* [GitHub for Mac](https://mac.github.com/)
* [Student directory for this class](https://github.com/bbalin12/DAT5_BOS_students)

**Homework:**
* Check for proper setup of Git by forking the [data science project examples](https://github.com/bbalin12/DAT-project-examples.git) and pulling the fork to your local hard drive.
* Download the following for Class 4:
	* [SQLite](http://www.sqlite.org/download.html).  Please make sure to download the precompiled binaries for your OS, NOT the source code. 
	* [Sublime Text Editor](http://www.sublimetext.com/).
	* [DB Visualizer](http://www.dbvis.com/).  Please download the free version. 
	* [Baseball archive for SQLite](https://github.com/jknecht/baseball-archive-sqlite/blob/master/lahman2013.sqlite). 


### Class 4: SQL
[slides](slides/04_sql_tutorial.pdf)
[Python code](code/04_sql_and_pandas.py)
[SQL code](code/04_sql_tutorial.sql)

[Overview of the baseball archive](data/baseball_database_description.txt)
* Installation of SQLite, Sublime, DB Visualizer, and our dataset
* The SELECT statement
* The WHERE clause
* ORDER BY
* LEFT JOIN and INNER JOIN
* GROUP BY
* DISTINCT
* CASE statements
* Subqueries and IS NOT NULL
* CREATE TABLE
* Using Pandas and SQL Seamlessly

**Homework:** 
* Complete the in-class excercises, if you haven't already:
	* Find the player with the most at-bats in a single season.
	* Find the name of the the player with the most at-bats in baseball history.
	* Find the average number of at_bats of players in their rookie season.
	* Find the average number of at_bats of players in their final season for all players born after 1980. 
	* Find the average number of at_bats of Yankees players who began their second season at or after 1980.
	* Pass the SQL in the previous bullet into a pandas DataFrame and write it back to SQLite.


* Create full, working queries to answer at least four novel questions you have about the dataset using the following concepts:
	* The WHERE clause
	* ORDER BY
	* LEFT JOIN and INNER JOIN
	* GROUP BY
	* SELECT DISTINCT
	* CASE statements
	* Subqueries and IS NOT NULL

* Using Pandas, (1) query the Baseball dataset, (2) transform the data in some way, and (3) write a new table back to the databse.

* Commit and Sync your SQL and Pandas files to your GitHub fork and issue a pull request.

**Resources:**
	* [SQLite homepage](https://www.sqlite.org/index.html)
	* [SQLite Syntax](https://www.sqlite.org/lang.html)

**SQL Tutorials:**
	* Note: These tutorials are for all flavors of SQL, not just SQLite, so some of the functions may behave differently in SQLite.
	* [SQL tutorial](http://www.w3schools.com/sql/)
	* [SQLZoo](http://sqlzoo.net/wiki/Main_Page)


### Class 5: Supervised Learning and Model Evaluation
[slides](slides/05_ml_knn.pdf)
[code](code/05_sklearn_and_ml.py)
* Overview of machine learning
* Supervised vs unsupervised learning
* Classification with K-nearest meighbors
* Training and test sets
* K-fold cross validation
* Tuning model paramaters via grid search

**Homework (due 2/10):** 
* Build a preditive model that predicts whether a player was inducted to the Baseball Hall of Fame before 2000 using their batting, pitching, and fielding results- not the number of votes they received.  Please make sure to use K-fold cross validaiton and grid search to find your best model. 
* Begin thinking about your class project.  It can be from anywhere you want -- your work, school or general interest. 

### Class 6: Linear Regression
[slides](slides/06_linear_regression.pdf)
[code](code/06_linear_regression.py)
* Overview of regression models
* Estimating linear regression coefficients
* Determining model relevance
* Interpreting model coefficients
* Gotchas
* Categorical features

**Homework (due 2/10):** 
* Using the Baseball dataset, build a linear regression model that predicts how many runs a player will have in a given year.
	* Begin with more than one possible model; each of which should have at least one categorical dummy feature and at least two continuous explanatory features.
	* Make sure you check for heteroskedasticity in your models.
	* Decide whether to include or take out the model's features depending on whether they may be collinear or insignificant. 
	* Interpret the model's coefficients and intercept.
	* Calculate the models' R-squared and in-sample RMSE.
	* Make sure to use a holdout group or k-fold CV to calculate out-of-sample RMSE for your model group.
	* Decide on the best model to use, and justify why you made that choice. 

### Class 7: Logistic Regression
[code](code/07_logistic_regression)
* Comparison to Linear Regression
* Probability review
* logistic regression
* logistic function

** Homework (due 03/02):**
* Using the baseball dataset, build a logistic regression model that predicts who is likely to be inducted into Hall of Fame.
  	* Start with considering as many explanatory variables.
	* What factors are signficant? Reduce your explanatory variables to the ones that are significant.
	* Compare performance between having many variables vs having a smaller set.
	* Cross validate your model and print out the coeffecients of the best model.
	* Considering any two features, generate a scatter plot with a class separable line showing the classification.

### Class 8: Naive Bayes, Decision Trees, and Classification Model Evaluation
[slides](slides/08_naive_bayes_decision_trees_model_evaluation.pdf)
[code](code/08_naive_bayes_decision_trees_model_evaluation.py)
* Bayes Theorem
* Naive Bayes Classification
* Decision Tree Classification
* Classification Model Evaluation
	* Confusion Matrices
	* F1 Score
	* Cohen's Kappa
	* Receiver-Operating Characteristic Curves

**Homework (due 2/17):**
* Build a Naive Bayes and decision tree model for your set of features you have extracted from the Baseball Dataset to predict whether a player was inducted into the Baseball Hall of Fame before the year 2000.  
* If you haven't already, build a logistic regression model for the same data. 
* Compare the n-fold cross-validated accuracy, F1 score, and AUC for your three new models, and compare it to the KNN model you built for class 5.
* Decide which of these models is the most accurate. 
* For your best performing model, print a confusion matrix and ROC curve in your iPython interpreter for all k cross validation slices. 
	* See [this link](http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py) for an example of how this is done for an ROC curve. 

### Class 9: Data Cleaning and Manipulation
[slides](slides/09_data_cleaning_and_manipulation.pdf)
[code](code/09_data_cleaning_and_manipulation.py)
* Encoding & Binning Categorical Data
* Finding Features with Perfect Correlation or No Variation
* Feature Standardization and Imputation
* Recursive Feature Elimination

**Homework (due 2/24):**
* Join your SQL query used in last class' homework (to predict Baseball Hall of Fame indution) with the table we created in today's class (called dominant_team_per_player). 
* Pick at least one additional categorical feature to include in your data.
* Bin and encode your categorical features.
* Remove features with perfect correlation and/or no variation.
* Scale your data and impute for your numeric NaNs.
* Perform recursive feature elimination on the data.
* Decide whether to use grid search to find your 'optimal' model.
* Bring in data after the year 2000, and preform the same transformations on the data you did with your training data.
* Predict Hall of Fame induction after the year 2000. 


### Class 10: Ensemble Methods and Neural Networks
[slides](slides/10_ ensemble_methods_and_neural_networks.pdf)
[code](code/10_ensemble_methods_and_neural_networks.py)
* Random Forests
* Boosting Trees
* Neural Networks

**Homework (due 3/3):**
* Run a Random Forest (RF), Boosting Trees (GBM), and Neural Network (NN) classifier on the data you assembled in the homework from class 9.
* See which of the methods you've used so far (RF, GBM, NN, Decision Tree, Logistic Regression, Naive Bayes) is the most accurate (measured by ROC AUC).
* Use grid seach to optimize your NN's tuning parameters for learning_rate, iteration_range, and compoents, as well as any others you'd like to test. 

### Class 11: Unsupervised Learning and Dimensionality Reduction
[slides](slides/11_dimensionality_reduction_and_clustering.pdf)
[code](code/11_dimensionality_reduction_and_clustering)
* K-Means Clustering
* DBSCAN
* Hirearchical Clustering
* Principal Component Analysis
* Support Vector Machines

**Homework (due 3/3):**

* Find some sort of attribute in the Baseball dataset that sits on a two-dimenstional plane and has discrete clusters.
* Perform K-Means and DBSCAN clustering.
* Determine which better represents your data and the intiution behind why the model was the best for your dataset.
* Plot a dendrogram of your dataset.
* Using the data from the last homework, perform principal component analysis on your data.
* Decide how many components to keep
* Run a Boosting Tree on the components and see if in-sample accuracy beats a classifier with the raw data not trasnformed by PCA.
* Run a support vector machine on your data
* Tune your SVM to optimize accuracy
* Like what you did in class 9, bring in data after 2000, and preform the same transformations on the data you did with your training data.
* Compare Random Forest, Boosting Tree, and SVM accuracy on the data after 2000.
* If you wish, see if using PCA improves the accuracy of any of your models.


### Class 13: Recommendation Systems
[slides](slides/13_recommendation_systems.pdf)
[code](code/13_recommender_systems.py)
* Collaborative Filtering
* Content Based Filtering
* Hybrid Filtering

** Homework
* [Non-Personalized Recommenders](http://nbviewer.ipython.org/github/python-recsys/recsys-tutorial/blob/master/tutorial/0-Introduction-to-Non-Personalized-Recommenders.ipynb)
* [yhat](http://help.yhathq.com/v1.0/docs/)

### Class 14: Natural Language Processing 
[slides](slides/14_natural_language_processing.pdf)
* vectorizing text features
* vector normalization
* TF-IDF
* cosine similarity
* Talk about projects

** Homework 
* [Text Mining](http://nbviewer.ipython.org/github/robertlayton/authorship_tutorials/blob/master/pyconau2014/PyCon%20AU%202014%20--%20Text%20mining%20online%20data%20with%20scikit-learn.ipynb)

** Links
* [Google n-grams](http://googleresearch.blogspot.com/2006/08/all-our-n-gram-are-belong-to-you.html)
* [Twitter sentiment analysis](http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)

### Class 15: Amazon Web Services, Elastic MapReduce and Apache Hive

Application presentation.

Question review.

Check out [Latency numbers every programmer should know](https://gist.github.com/hellerbarde/2843375). (Disk is slow!)

[Slides](slides/15_map_reduce.pdf) on map-reduce.

Walk-through for doing map-reduce on Amazon Elastic MapReduce (EMR):

#### The AWS Command Line Interface (CLI)

Amazon provides an [AWS CLI](https://aws.amazon.com/cli/) for interacting with many of their services, including [S3](http://aws.amazon.com/s3/). It installs easily with [pip](https://pypi.python.org/pypi/pip). You'll need an [AWS](http://aws.amazon.com/) account and an [access key](https://console.aws.amazon.com/iam/home?#security_credential) to configure it.

```bash
pip install awscli
aws configure
```

Now you can easily move files into and out of S3 buckets:

```bash
aws s3 cp myfile s3://mybucket
aws s3 sync s3://mybucket .
```

And so on. (See `aws s3 help` etc.)

#### Streaming map-reduce with Python

This example uses tweets as the data. The tweets were loaded into Python and then written to disk as stringified dicts.  A manageable chunk containing just 11 tweets is available: [https://s3.amazonaws.com/gadsdc-twitter/out03.txt](https://s3.amazonaws.com/gadsdc-twitter/out03.txt)

Here are simple [map](code/L15/map.py) and [reduce](code/L15/reduce.py) scripts. You can run locally:

```bash
cat input | ./map.py | sort | ./reduce.py > output
```

You can run cluster [streaming jobs on Amazon EMR](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/CLI_CreateStreaming.html) through the AWS console.

More things to try implementing this way:

 * What words are "trending" in your tweets?
 * What were the most popular hashtags?
 * How many tweets came in per hour?
 * What tweets / which people were most re-tweeted?
 * Can you induce a graph from "conversations"?

There is a [command line interface](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-cli-reference.html) for [Elastic Map Reduce](https://aws.amazon.com/elasticmapreduce/) as well, but it's a bit old, and depends on Ruby 1.8.7.


#### More abstraction

[Pig](http://pig.apache.org/) lets you write [Pig Latin](http://pig.apache.org/docs/r0.7.0/piglatin_ref2.html) scripts for doing complex map-reduce tasks more easily. [Hortonworks](http://hortonworks.com/) has an introductory [tutorial](http://hortonworks.com/hadoop-tutorial/how-to-process-data-with-apache-pig/). [Mortar](http://www.mortardata.com/) has a [tutorial](http://help.mortardata.com/technologies/pig/learn_pig) as well. You can also run [Pig on Amazon EMR](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-pig-launch.html).

[PigPen](https://github.com/Netflix/PigPen/wiki) "is map-reduce for Clojure. It compiles to Apache Pig, but you don't need to know much about Pig to use it."

[Hive](http://hive.apache.org/) adds some more structure to data and let's you write [HiveQL](https://cwiki.apache.org/confluence/display/Hive/LanguageManual), which is very close to SQL. You can also run [Hive on Amazon EMR](http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-hive.html).

 * [mrjob](https://github.com/Yelp/mrjob) is a Python library from [Yelp](http://www.yelp.com/) that wraps map-reduce and can run jobs on EMR.
 * [Luigi](https://github.com/spotify/luigi) is a Python library from [Spotify](https://www.spotify.com/us/) that lets you write map-reduce workflows more easily.
 * [Cascading](http://www.cascading.org/) is a layer on top of Hadoop that has further layers such as [Scalding](https://github.com/twitter/scalding) ([Scala](http://www.scala-lang.org/)) from [Twitter](https://twitter.com/) - yet another way to simplify working with map-reduce.
 * [RHadoop](https://github.com/RevolutionAnalytics/RHadoop/wiki) provides an interface for running `R` on Hadoop.

There's also big graph processing as in [Giraph](http://giraph.apache.org/), which is inspired by Google's [Pregel](http://dl.acm.org/citation.cfm?id=1807184).

Totally separate from Hadoop, [MongoDB](http://www.mongodb.org/) has an internal implementation of map-reduce.


#### Beyond Map-Reduce

Cloudera's [Impala](http://www.cloudera.com/content/cloudera/en/products-and-services/cdh/impala.html) is inspired by Google's [Dremel](http://research.google.com/pubs/pub36632.html). Of course there's also [Drill](http://incubator.apache.org/drill/). And if you want to get Dremel straight from the source, you can buy it as a service from Google as [BigQuery](https://cloud.google.com/products/bigquery/).

[Spark](http://spark.apache.org/) keeps things in memory to be much faster. This is especially useful for iterative processes. See, for example, their [examples](https://spark.incubator.apache.org/examples.html), which feature their nice Python API. There's also [Shark](http://shark.cs.berkeley.edu/), which gives much faster HiveQL query performance. You can [run Spark/Shark on EMR](https://aws.amazon.com/articles/Elastic-MapReduce/4926593393724923) too.

There's also distributed stream processing as in [Storm](http://storm.incubator.apache.org/).


#### `sklearn` for huge data?

Not exactly. But there are some projects that step in that direction:

[Mahout](http://mahout.apache.org/) is a project for doing large scale machine learning. It was originally mostly map-reduce oriented, but in April 2014 announced a move toward Spark.

[MLlib](http://spark.apache.org/docs/0.9.0/mllib-guide.html) is the machine learning functionality directly on Spark, which is actively growing.


### After

Optional:

 * UC Berkeley's [AMP Camp](http://ampcamp.berkeley.edu/) provides great resources for learning a range of technologies including Spark. (Berkeley's [AMP Lab](https://amplab.cs.berkeley.edu/software/) is responsible for a lot of these cool technologies.)
 * [HUE](http://gethue.com/) "Hadoop User Experience" "is a Web interface for analyzing data with Apache Hadoop" that you might find inside various vendors' platforms, or separately.
 * This [paper](http://arxiv.org/pdf/1402.6076v1.pdf) describes large-scale machine learning in a very real-world advertising setting.
 * See also [Ad Click Prediction: a View from the Trenches at Google](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/41159.pdf).
 * [Writing an Hadoop MapReduce Program in Python](http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/) is a streaming walk-through that runs Hadoop directly.
 * [An elastic-mapreduce streaming example with python and ngrams on AWS](http://dbaumgartel.wordpress.com/2014/04/10/an-elastic-mapreduce-streaming-example-with-python-and-ngrams-on-aws/) is another walk-through that uses the EMR CLI.
 * Check out an [overview](http://highlyscalable.wordpress.com/2012/02/01/mapreduce-patterns/) of algorithms over map-reduce.
 * For more on doing joins with map-reduce, see this [thesis](http://www.inf.ed.ac.uk/publications/thesis/online/IM100859.pdf).
 * [Read about](http://www.cs.stanford.edu/people/ang//papers/nips06-mapreducemulticore.pdf) doing ML faster by using more cores, using map-reduce.
 * Go through an old Twitter [deck](http://www.slideshare.net/kevinweil/hadoop-pig-and-twitter-nosql-east-2009) on why Pig is good.
 * Read about why [Spark is a Crossover Hit For Data Scientists](http://blog.cloudera.com/blog/2014/03/why-apache-spark-is-a-crossover-hit-for-data-scientists/).

### Class 16: NLP Bootcamp - LSI
[Slides](https://docs.google.com/presentation/d/1wv6r2p7lXKepoeX4WIvkqBMThELkI5irQYYtZWfyX70/pub?slide=id.g3aa5c6c7c_05) on NLP

The entire class will be spent working together on the LSI Workshop in the IPython Notebook below, with occasional references to the guest lecture slides above.
[LSI Workshop](code/L16)


### Class 17: Web Scraping and Data Retrieval Methods


### Class 18: Recommenders


### Class 19: Course Review, Companion Tools


### Class 20: TBD


### Class 21: Project Presentations


### Class 22: Project Presentations


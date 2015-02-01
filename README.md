## DAT5 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science) in Boston, MA (20 January 2015 - 07 April 2015). View student work in the [student repository](https://github.com/bbalin12/DAT5_BOS_students).

**Instructor:** Bryan Balin. **Teaching Assistant:** Harish Krishnamurthy.

**Office hours:** Wednesday 5-6pm; Friday 5-6:30pm, [@Boston Public Library](http://www.bpl.org/); as needed Tuesdays and Thursdas @GA. 

**[Course Project information](project.md)**

Tuesday | Thursday
--- | ---
1/20: [Introduction](#class-1-introduction) | 1/22: [Python & Pandas](#class-2-python-&-pandas)
1/27: [Git and GitHub](#class-3-git-and-github) | 1/29: [SQL](#class-4-SQL)
2/3: [Supervised Learning and Model Evaluation](#class-5-Supervised-Learning-and-Model-Evaluation) | 2/5: [Linear and Logistic Regression](#class-6-Linear-and-Logistic-Regression) <br>**Milestone:** Begin Work on Question and Data Set
2/10:  [Naive Bayes, Neural Networks, and Ensemble Methods](#class-7-Naive Bayes-Neural-Networks-and-Ensemble-Methods)| 2/12: [Categorical Data and Feature Selection/Elimination](#class-8-Categorical-Data-and-Feature-Selection/Elimination)
2/17: [Unsupervised Learning and Dimensionality Reduction](#class-9-Unsupervised-Learning-and-Dimensionality-Reduction) | 2/19: [Clustering and Visualization](#class-10-Clustering-and-Visualization)<br>**Milestone:** Data Exploration and Analysis Plan
2/24: [Working a Data Problem](#class-11-working-a-data-problem) | 2/26: [Amazon Web Services & Apache Hive](#class-12-Amazon-Web Services-& Apache-Hive)<br>**Milestone:** Deadline for Topic Changes
3/3: [Amazon Elastic MapReduce](#class-13-Amazon-Elastic-MapReduce) | 3/5: [Natural Language Processing](#class-14-natural-language-processing)
3/10: [Decision Trees and Ensembles](#class-15-decision-trees-and-ensembles)<br>**Milestone:** First Draft | 3/12: [Advanced scikit-learn](#class-16-advanced-scikit-learn)
3/17: **No Class**  | 3/19: [Web Scraping and Data Retrieval Methods](#class-17-Web-Scraping-and-Data-Retrieval-Methods)
3/24: [Recommenders](#class-18-recommenders) | 3/26: [Course Review, Companion Tools](#class-19-course-review-companion-tools)<br>**Milestone:** Second Draft (Optional)
3/31: [TBD](#class-20-tbd) | 4/2: [Project Presentations](#class-21-project-presentations)
4/7: [Project Presentations](#class-22-project-presentations) |


### Class 5: Supervised Learning and Model Evaluation


### Class 6: Linear and Logistic Regression


### Class 7: Naive Bayes, Neural Networks, and Ensemble Methods


### Class 8: Categorical Data and Feature Selection/Elimination


### Class 9: Unsupervised Learning and Dimensionality Reduction


### Class 10: Clustering and Visualization


### Class 11: Working a Data Problem


### Class 12: Amazon Web Services & Apache Hive


### Class 13: Amazon Elastic MapReduce


### Class 14: Natural Language Processing


### Class 15: Decision Trees and Ensembles


### Class 16: Advanced scikit-learn


### Class 17: Web Scraping and Data Retrieval Methods


### Class 18: Recommenders


### Class 19: Course Review, Companion Tools




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
[slides](slides/05_ml_knn.pdf.pdf)

### Class 6: Linear and Logistic Regression


### Class 7: Naive Bayes, Neural Networks, and Ensemble Methods


### Class 8: Categorical Data, Feature Selection


### Class 9: Unsupervised Learning and Dimensionality Reduction


### Class 10: Clustering and Visualization


### Class 11: Working a Data Problem


### Class 12: Amazon Web Services & Apache Hive


### Class 13: Amazon Elastic MapReduce


### Class 14: Natural Language Processing


### Class 15: Decision Trees and Ensembles


### Class 16: Advanced scikit-learn


### Class 17: Web Scraping and Data Retrieval Methods


### Class 18: Recommenders


### Class 19: Course Review, Companion Tools


### Class 20: TBD


### Class 21: Project Presentations


### Class 22: Project Presentations


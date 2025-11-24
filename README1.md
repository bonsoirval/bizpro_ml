<center><h1>BIZPRO</h1></center>
https://www.bizpro.ng/
This project is a complete, runnable CRISP-DM example using the scikit-learn breast cancer dataset.
It includes:
- Full Python scripts for each CRISP-DM phase
- A training pipeline that logs experiments to MLflow (local file backend)
- A simple Flask API for serving predictions
- Monitoring sketch for drift detection
- Makefile targets to run common tasks
- GitHub Actions CI workflow that runs tests
- requirements.txt listing packages used

## Quick start (local)
1. Create venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run EDA:
```bash
make run-eda
```

3. Prepare data:
```bash
make prep
```

4. Train (uses MLflow local tracking):
```bash
make train
# This creates artifacts/ and logs in mlruns/
```

5. Evaluate:
```bash
make evaluate
```

6. Serve API (for manual testing):
```bash
make serve
# then POST to http://127.0.0.1:5000/predict
```

## Project layout
See detailed README files inside each numbered directory for explanations.







### This is outside of crisp-ml

7 Steps to a Successful Data Science Project
Beginners Guide on Completing a Data Science Project from Scratch

Amit Bharadwa
Feb 6, 2021
8 min read
Share
Photo by Clark Tibbs on Unsplash
Photo by Clark Tibbs on Unsplash
Data science projects are essential for anyone breaking into the field and to build a personal portfolio. No matter if you’re an absolute beginner or a seasoned professional, a logical approach will help your projects become a success. This post describes an easy seven-step method you can apply to your projects to tackle them confidently.

The method is as follows:

Problem Statement
Data Collection
Data Cleaning
Exploratory Data Analysis (EDA)
Feature Engineering
Modelling
Communication
Okay! so that’s the methodology. Now let’s go into more detail about each of these steps and tackle them with helpful tips and tricks.

### 1. Problem Statement
Whether it’s a business problem or a personal project that you are working on, a well-defined problem can save you a lot of time and trouble. The objective of a problem statement is to state the problem you are trying to solve clearly. If it is done well, it can be defined in a couple of sentences.

Remember your problem statement has to be SMART.

Specific: The problem statement has to be detailed and specific to the problem you are solving.
Measurable: Are there any metrics you can track so that you can tell if it is successful at the end of the project?
Action: What specific actions can you take to solve your problem?
Relevant: There are multiple ways to solve a problem but focus on the most relevant method.
Timebound: Have you added a time constraint to when your problem should be solved?
e.g. How can XYZ reduce their failure rates below 5% by the end of the year, through manufacturing and analysing product performance?

A problem statement that follows a SMART guideline will set you up on a successful track to meeting your end goal. More importantly, after you have finished the problem statement, you will have a much better idea of your project’s finer details.

### 2 Data Collection:
Data collection is the process of gathering and measuring information on targeted variables of interest in an organised system, which then allows you to answer relevant questions and decide future outcomes.

A few examples of data collection methods include:

A government institution
Kaggle
Company database server
Self-collected data
No matter where you obtain your data from, KEEP IN MIND: garbage in, garbage out

Make sure your data is Relevant and Validated. If your data is not suitable for the problem you are solving, your results will be useless no matter how good your model is. sQUALITY IS KEY!

Data collection can take time so don’t rush this step!

### 3. Data Cleaning
Around 80% of your time will be spent cleaning data. You cannot overlook this step!

Cleaning your data is a process of ensuring your data is in the correct format; consistent and errors are identified and dealt with appropriately.

The actions below lead to a cleaner dataset:

Remove duplicate values ( This is usually the case when combining multiple datasets)
Remove irrelevant observations (observations need to be specific to the problem you are solving)
Address missing values (e.g. Imputation techniques, drop features/observations)
Reformat data types (e.g. boolean, numeric, Datetime)
Filter unwanted outliers (if you have a legitimate reason)
Reformat strings (e.g. remove white spaces, mislabeled/misspelt categories)
Validate (does the data make sense? does the data adhere to the defined business rules? )
Cleaning your data will allow for higher-quality information and ultimately lead to a more conclusive and accurate decision.

### 4. Exploratory Data Analysis (EDA)
As the name suggests, during EDA, you get a deeper understanding of the data. During this step, you want to understand your data’s statistical characteristics, create visualisations, ** and test hypothesise**s.

This is where you show your creative side!

There are four main types of EDA:

Univariate non-graphical: make observations of the population and understand sample distributions of a single variable. (e.g. the measure of spread, the measure of central tendency, outlier detection)
Univariate graphical: graphical analysis on a single variable. (e.g. Histograms, Boxplots, Stem and leaf)
Multivariate non-graphical: techniques which show the relationship between two or more variables. (e.g. covariance, correlations)
Multivariate graphical: graphically show the relationship between two or more variables. (e.g. bar plots, scatterplots)
Remember the aim of EDA is to find underlying patterns within the data, detect outliers and test assumptions with the final aim of finding a model that fits the data well.

### 5. Feature Engineering
A feature is an attribute of a dataset that is useful to the problem you are solving. If a feature has no impact on the problem you are solving, it is not part of the problem.

So what is feature engineering?

Feature engineering is defined as the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.

The better the features you create and choose for your predictive models’, the better your results will be. Feature engineering is an art where you decompose or aggregate, raw data to help solve your problem; however, there are many approaches to this process.

Feature Extraction: select and/or combine variables into features to reduce the dimensionality of your dataset. (e.g. Principle Component Analysis, Nonlinear dimensionality reduction, unsupervised clustering methods)
Feature Selection: select the features which contribute most to the problem you are solving. (e.g. Variance thresholding, Pearson correlation, LASSO)
Feature Construction: the process of manually building more efficient features **** from raw data. _(e.g. Dynamic aggregation of relational attribute_s)
Feature Learning: the **** automatic identification and use of features. _(e.g. Restricted Boltzmann machine, K-means clusterin_g)
Using feature importance scoring methods, you can estimate how useful the feature will be. Features are given scores so they can be ranked based on these scores.

Methods include:

the correlation coefficient between the feature and the target variable (the feature that you are trying to predict)
co-integration between two time series (for time-series data)
predictive models have embedded feature selection methods(e.g. Random Forest, Gradient Boosting Machine)
Chi-Squared test (between target and numerical variable)
Recursive Feature Elimination (EXPLAIN)
It is normal to find yourself returning to this step multiple times.

Feature engineering is an iterative process. It can look something like:

Brainstorm feature ideas.
create features based on the problem (e.g. feature extraction/construction)
choose features based on feature importance scores
Calculate model accuracy using the chosen features on unseen data.
Repeat steps until a suitable model is chosen.

### 6. Modelling
All machine learning models are classified either as a Supervised or Unsupervised learning problem.

_A Supervised problem is where a function maps an input to an output based on input-output pairs. The machine learning model learns from the input-output training data to make predictions on unseen data (test data). An Unsupervised problem is where a model looks for patterns within an unlabelled dataset._

Supervised learning problems are labelled as a Regression ( output variable is a real value) or Classification (output variable is a category) problem—more on the difference between them here. You can identify which metric/metrics you will use to compare models’ accuracy by labelling your problem.

Regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), R-Squared, Adjusted R-Squared
Classification metrics: Accuracy, Precision, Recall, F1-Score
Preprocessing
Data preprocessing helps to enhance your data quality by organizing raw data in a suitable format to build and train a machine learning model.

The first step is splitting your data into train and test datasets. This is important as you don’t want to contaminate the training data with the test data.
The second step is to Standardize or Normalize your data if the model’s algorithm is sensitive to unscaled data.
For the third step, the training and test data needs to be split into target variable (what you are trying to predict) and predictor variables (the features you are using to predict the target variable).
Machine learning models
As previously mentioned, machine learning models are classified as supervised or unsupervised. I will outline some models that are used in these categories.

Supervised – Regression: Linear Regression, Multivariate Linear Regression, Support Vector Regression (SVR), Random Forest, Neural Networks
Supervised – Classification: Logistic Regression, **** Support Vector Machines (SVM), Random Forest, Neural Networks, k-Nearest Neighbor __ (k_N_N)
Unsupervised: K-means clustering, Principal Component Analysis (PCA), Singular Value Decomposition (SVD)
After building a few machine learning models, the models needs to be trained by tuning the hyperparameters to optimize model performance. The hyperparameters are parameters used to control the learning process and reduce a predefined loss function. By comparing the predefined metrics for each model, an optimal model can be chosen.

### 7. Communication
Lastly, it is essential to communicate your results. This can be done through a presentation, formal report or even a blog post. The point is the world has to see the amazing work you have done. A few key points to remember:

Don’t overcrowd your slides (6 items max)
Use relevant visualisations
Know your audience
Make sure it flows
Data science is about communicating your results well. Do it with passion, use a storytelling approach and show your audience why your findings are so interesting.


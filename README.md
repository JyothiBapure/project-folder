Problem Statement

This project implements 6 ML classifiers to Predict whether an individual earns more than $50K per year based on demographic and work-related features 
such as age, education, occupation, and hours worked per week. This prediction is useful for economic studies, policy making, and financial planning

Dataset Description

The dataset used in this project is the Adult Income Dataset from the UCI Machine Learning Repository.
Total Instances: ~48,000 records
Target Variable: income
<=50K → 0
>50K → 1

| Feature Name   | Description               |
| -------------- | ------------------------- |
| age            | Age of the individual     |
| workclass      | Employment type           |
| fnlwgt         | Final sampling weight     |
| education      | Highest education level   |
| education-num  | Numerical education level |
| marital-status | Marital status            |
| occupation     | Occupation category       |
| relationship   | Relationship status       |
| race           | Race                      |
| sex            | Gender                    |
| capital-gain   | Capital gains             |
| capital-loss   | Capital losses            |
| hours-per-week | Working hours per week    |
| native-country | Country of origin         |

Pre-Processing Step
Removal of missing values (?)
One-hot encoding for categorical features
Stratified train–test split
Feature scaling applied where required (Logistic Regression, KNN)

c. Models Used

The following six machine learning models were implemented and evaluated using the same preprocessing pipeline to ensure fair comparison:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest Classifier
XGBoost Classifier

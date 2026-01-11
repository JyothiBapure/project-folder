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

Models Used

The following six machine learning models were implemented and evaluated using the same preprocessing pipeline to ensure fair comparison:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest Classifier
XGBoost Classifier

Comparison Table with the evaluation metrics calculated for all the 6 models

| ML Model Name                 | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ----------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression           | 0.8473     | 0.9023     | 0.7341     | 0.6065     | 0.6642     | 0.5709     |
| Decision Tree Classifier      | 0.8523     | 0.9003     | 0.7701     | 0.5799     | 0.6616     | 0.5789     |
| K-Nearest Neighbor Classifier | 0.8210     | 0.8398     | 0.6577     | 0.5859     | 0.6197     | 0.5046     |
| Naive Bayes Classifier        | 0.7872     | 0.8248     | 0.6608     | 0.2983     | 0.4110     | 0.3389     |
| Random Forest Classifier      | 0.8550     | 0.9147     | 0.7889     | 0.5699     | 0.6618     | 0.5848     |
| XGBoost Classifier            | 0.8666     | 0.9233     | 0.7751     | 0.6538     | 0.7093     | 0.6273     |


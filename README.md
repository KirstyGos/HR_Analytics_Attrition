# HR_Analytics_Attrition

**The Goal**
The goal of this project is to predict whether an employee will leave a company or not. The data is from Kaggle entitled IBM HR Analytics Employee Attrition Analysis

**Libraries**
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
    - LogisticRegression
    - MinMaxScaler
    - RandomForestClassifier
    - train_test_split
    - accuracy_score, classification_report
    - confusion_matrix
    - SMOTETomek
    - DecisionTreeClassifier
    - SMOTE
    - accuracy_score, precision_score, recall_score, f1_score

**The steps:**
- Import libraries
- Goal Description
- Read the data into Python
- Undertsand the data
- Clean the data
- Preprocessing
    - MinMaxScaler
    - Get dummies
- EDA
    - Distplot
    - Correlation matrix
    - Kernel density estimation
- Modeling
    - Train test split
    - Logisitc Regression
    - Evaluation: confusion matrix, Roc Curve, Classification report
- Dealing with imbalanced data
    - SMOTE for oversampling
    - Tomek links for undersampling
- Modeling round 2
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine
- Hyperparameter tuning
    - Logistic Regression
- Modeling round 3
    - MLPClassifier
- Hyperparameter tuning round 2
    - Random Forest
 
**Description**
The first step in the process was to clean the data and then perform some EDA and prprocessing. 
Next was to apply the processed data to a Logisitc Regression model and evaluate the model.

As Logisitc Regression was not the best model, it has a very good accuracy score, but is not very good at predicting class 1 (attrition = no).
I therefore applied some sampling techniques:
I first tried SMOTE to oversample the the minority class; this didn’t improve very much in the classification report.
Next I applied TomekLinks to undersample the data, but thsi also didn’t have th desired effect.

I then went on to apply some other models: Decision Tree, Random Foresta and Support Vector Machine. None of these models provided very good result either. Most of the models have high accuracy scores, but are not very good at predicting class 1 (attrition = no).

As Logistic regression was the best model, I applied some hyperparameter tuning to see if I could improve the performance. Unfortunately, this also didn’t help with predicting class 1 (atterition = 1).

This led me to try a new model called MLPClassifier. This model didn’t provide much better results unfortunately. 

I then decided to apply hyperparameter tuning to the Random Forest model as this model usually provides quite good results. But alas, no improvement there either.

Some other techniques that would be attempted next would be feature engineering, feature selection, gathering more data and trying different models.


**MySQL:**
The dataframe used for this is called 'attrition'
Here you can find 3 different queries:
- The relationship between attrition and job satisfaction (line 6)
- The relationship between attrition and gender (line 14)
- The employee attrition rate (line 39) 

**Tableau:**
You can find the following analysis:
- Age distribution
- Attrition
- Attrition rate
- Average monthly income per job group
- Average monthly income per education level
- HR Analytics Attrition dashboard

**Streamlit:**
Streamlit didn’t work with my models, so I came up with a solution for the purpose of the presentation
Appppppy.py is the relevant file
 

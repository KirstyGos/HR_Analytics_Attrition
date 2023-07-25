#Import the libraries

# Basic libraries
import streamlit as st
import pandas as pd
import numpy as np
#%matplotlib inline
#import matplotlib.pyplot as plt
#import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Other
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/Users/KirstyG/Desktop/Ironhack/HR_Analytics_Attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data

data = data.rename(columns={'Age': 'age',
                       'Attrition': 'attrition',
                       'BusinessTravel': 'business_travel',
                       'DailyRate': 'daily_travel',
                       'Department': 'department',
                       'DistanceFromHome': 'distance_from_home',
                       'Education': 'education',
                       'EducationField': 'education_field',
                       'EmployeeCount': 'employee_count',
                       'EmployeeNumber': 'employee_number',
                       'EnvironmentSatisfaction': 'environment_satisfaction',
                       'Gender': 'gender',
                       'HourlyRate': 'hourly_rate',
                       'JobInvolvement': 'job_involvement',
                       'JobLevel': 'job_level',
                       'JobRole': 'job_role',
                       'JobSatisfaction': 'job_satisfaction',
                       'MaritalStatus': 'marital_status',
                       'MonthlyIncome': 'monthly_income',
                       'MonthlyRate': 'monthly_rate',
                       'NumCompaniesWorked': 'num_companies_worked',
                       'Over18': 'over_18',
                       'OverTime': 'over_time',
                       'PercentSalaryHike': 'percent_salary_hike',
                       'PerformanceRating': 'performance_rating',
                       'RelationshipSatisfaction': 'relationship_satisfaction',
                       'StandardHours': 'standard_hours',
                       'StockOptionLevel': 'stock_option_level',
                       'TotalWorkingYears': 'total_working_years',
                       'TrainingTimesLastYear': 'training_times_last_year',
                       'WorkLifeBalance': 'work_life_balance',
                       'YearsAtCompany': 'years_at_company',
                       'YearsInCurrentRole': 'years_in_current_role',
                       'YearsSinceLastPromotion': 'years_since_last_promotion',
                       'YearsWithCurrManager': 'years_with_curr_manager'})
data.head(5)

# we can drop employee_number as it has no use in the model
data = data.drop('employee_number', axis = 1)
data

#split numericals and categoricals
data_cat = data.select_dtypes(include = object)
data_num = data.select_dtypes(include = np.number)

# scale numerical features
transformer = MinMaxScaler().fit(data_num)
data_num_minmax = transformer.transform(data_num)
data_num_norm = pd.DataFrame(data_num_minmax,columns= data_num.columns)
data_num_norm.head()

# encode the categorical feature
data_cat_dumm = pd.get_dummies(data_cat, drop_first = True)
data_cat_dumm.head()

data_cat_dumm = data_cat_dumm.rename(columns = {'attrition_Yes': 'attrition'})
data_cat_dumm

# drop employee_count and standard_hours
data_num = data_num.drop(['employee_count', 'standard_hours'], axis = 1)
data_num

# concat the dataframes
concatenated_data = pd.concat([data_num_norm, data_cat_dumm], axis = 1)
concatenated_data.head()

# define X and y for the model, our target is attrition
X = concatenated_data.drop('attrition', axis = 1)
y = concatenated_data['attrition']

# Split the data into training and testing sets=
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# use logistic regression to train model
classification = LogisticRegression(random_state = 0).fit(X_train, y_train)

# predictions with testing dataset
predictions = classification.predict(X_test)

# classification report
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# apply SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

y_smote.value_counts()

# round 2 of logistic regression to train model
classification_smote = LogisticRegression(random_state = 0).fit(X_smote, y_smote)

# predictions with testing dataset
predictions_smote = classification_smote.predict(X_test)

# classification report with SMOTE
report_smote = classification_report(y_test, predictions_smote)
print("Classification Report:\n", report_smote)

# Tomek links
smt = SMOTETomek(random_state = 42)
X_tomek, y_tomek = smt.fit_resample(X_smote, y_smote)

# round 3 of logistic regression to train model
classification_tomek = LogisticRegression(random_state = 0).fit(X_tomek, y_tomek)

# predictions with testing dataset
predictions_tomek = classification_tomek.predict(X_test)

# classification report with Tomek Links
report_tomek = classification_report(y_test, predictions_tomek)
print("Classification Report with Tomek links:\n", report_tomek)

from sklearn.metrics import classification_report


def apply_models(X, y, test_size=0.2, random_state=42):
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Support Vector Machine': SVC(random_state=random_state)
    }

    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store the evaluation metrics in the results dictionary
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(
            f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Generate the classification report and print it
        report = classification_report(y_test, y_pred)
        print(f"\nClassification Report for {model_name}:\n{report}\n")

    return results


# Usage example:
results = apply_models(X, y)


# STREAMLIT
st.title("Your Favourite HR Company")
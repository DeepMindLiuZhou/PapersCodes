import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load the data
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Display the first few rows of the datasets
    print("Train Data Preview:")
    print(train_data.head())
    
    print("\nTest Data Preview:")
    print(test_data.head())
    
    # Print the shape of the datasets
    print("\nTrain Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    
    # Display data types and count of non-null values
    print("\nTrain Data Info:")
    print(train_data.info())
    
    print("\nTest Data Info:")
    print(test_data.info())
    
    # Generate summary statistics for numerical columns
    print("\nTrain Data Summary Statistics (Numerical):")
    print(train_data.describe())
    
    print("\nTest Data Summary Statistics (Numerical):")
    print(test_data.describe())
    
    # Generate summary statistics for categorical columns
    print("\nTrain Data Summary Statistics (Categorical):")
    print(train_data.describe(include='object'))
    
    print("\nTest Data Summary Statistics (Categorical):")
    print(test_data.describe(include='object'))
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Define numerical features
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    
    # Plot histograms for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(train_data[feature].dropna(), kde=True)
        plt.title(f'Histogram of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/histogram_{feature}.png')
        plt.close()
    
    # Plot box plots for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=train_data[feature].dropna())
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/boxplot_{feature}.png')
        plt.close()
    
    
    # Define categorical features
    categorical_features = ['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    
    # Generate frequency tables for categorical features
    for feature in categorical_features:
        print(f"\nFrequency Table for {feature}:")
        print(train_data[feature].value_counts())
    
    # Plot bar charts for categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=train_data[feature])
        plt.title(f'Bar Chart of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/barchart_{feature}.png')
        plt.close()
    
    
    # Summarize key findings from the univariate analysis
    summary_report = """
    ### Preliminary EDA Summary Report ###
    
    **Key Findings:**
    1. **Missing Values:**
       - 'Age' has missing values in the training set.
       - 'Cabin' has many missing values in both training and test sets.
       - 'Embarked' has a few missing values in the training set.
    
    2. **Outliers:**
       - 'Fare' has some high outliers in the training set.
       - 'Age' has a wide range of values with potential outliers.
    
    3. **Class Distributions:**
       - 'Survived' class is imbalanced with more non-survivors (0) than survivors (1).
       - 'Pclass' has more passengers in 3rd class compared to 1st and 2nd classes.
       - 'Sex' has more males than females.
       - 'Embarked' is mostly from Southampton (S).
    
    **Initial Insights:**
    - 'Age' and 'Cabin' columns need imputation for missing values.
    - Outliers in 'Fare' and 'Age' need to be addressed.
    - Class imbalance in 'Survived' should be considered during model building.
    - Categorical features like 'Pclass', 'Sex', and 'Embarked' show distinct distributions that may be useful for feature engineering.
    
    These insights will guide the Data Cleaning phase.
    """
    
    print(summary_report)
    
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load the data
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Display the first few rows of the datasets
    print("Train Data Preview:")
    print(train_data.head())
    
    print("\nTest Data Preview:")
    print(test_data.head())
    
    # Print the shape of the datasets
    print("\nTrain Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    
    # Display data types and count of non-null values
    print("\nTrain Data Info:")
    print(train_data.info())
    
    print("\nTest Data Info:")
    print(test_data.info())
    
    # Generate summary statistics for numerical columns
    print("\nTrain Data Summary Statistics (Numerical):")
    print(train_data.describe())
    
    print("\nTest Data Summary Statistics (Numerical):")
    print(test_data.describe())
    
    # Generate summary statistics for categorical columns
    print("\nTrain Data Summary Statistics (Categorical):")
    print(train_data.describe(include='object'))
    
    print("\nTest Data Summary Statistics (Categorical):")
    print(test_data.describe(include='object'))
    
    # Define numerical features
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    
    # Plot histograms for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(train_data[feature].dropna(), kde=True)
        plt.title(f'Histogram of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/histogram_{feature}.png')
        plt.close()
    
    # Plot box plots for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=train_data[feature].dropna())
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/boxplot_{feature}.png')
        plt.close()
    
    # Define categorical features
    categorical_features = ['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    
    # Generate frequency tables for categorical features
    for feature in categorical_features:
        print(f"\nFrequency Table for {feature}:")
        print(train_data[feature].value_counts())
    
    # Plot bar charts for categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=train_data[feature])
        plt.title(f'Bar Chart of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/pre_eda/images/barchart_{feature}.png')
        plt.close()
    
    # Summarize key findings from the univariate analysis
    summary_report = """
    ### Preliminary EDA Summary Report ###
    
    **Key Findings:**
    1. **Missing Values:**
       - 'Age' has missing values in the training set.
       - 'Cabin' has many missing values in both training and test sets.
       - 'Embarked' has a few missing values in the training set.
    
    2. **Outliers:**
       - 'Fare' has some high outliers in the training set.
       - 'Age' has a wide range of values with potential outliers.
    
    3. **Class Distributions:**
       - 'Survived' class is imbalanced with more non-survivors (0) than survivors (1).
       - 'Pclass' has more passengers in 3rd class compared to 1st and 2nd classes.
       - 'Sex' has more males than females.
       - 'Embarked' is mostly from Southampton (S).
    
    **Initial Insights:**
    - 'Age' and 'Cabin' columns need imputation for missing values.
    - Outliers in 'Fare' and 'Age' need to be addressed.
    - Class imbalance in 'Survived' should be considered during model building.
    - Categorical features like 'Pclass', 'Sex', and 'Embarked' show distinct distributions that may be useful for feature engineering.
    
    These insights will guide the Data Cleaning phase.
    """
    
    print(summary_report)
    


if __name__ == "__main__":
    generated_code_function()
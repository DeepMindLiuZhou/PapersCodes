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
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/test.csv')
    
    # Make copies of the original DataFrames
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    # Handle missing values for 'Age'
    fill_missing_values(train_df_copy, columns=['Age'], method='mean')
    fill_missing_values(test_df_copy, columns=['Age'], method='mean')
    
    # Handle missing values for 'Cabin'
    fill_missing_values(train_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    fill_missing_values(test_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    
    # Handle missing values for 'Embarked'
    fill_missing_values(train_df_copy, columns=['Embarked'], method='mode')
    fill_missing_values(test_df_copy, columns=['Embarked'], method='mode')
    
    # Handle missing values for 'Fare'
    fill_missing_values(train_df_copy, columns=['Fare'], method='mean')
    fill_missing_values(test_df_copy, columns=['Fare'], method='mean')
    
    pass
    
    # Treat outliers for 'Fare'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Fare'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Fare'], factor=1.5, method='clip')
    
    # Treat outliers for 'Age'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Age'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Age'], factor=1.5, method='clip')
    
    pass
    
    # Convert data types
    convert_data_types(train_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(train_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    convert_data_types(test_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(test_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    # Remove duplicates
    remove_duplicates(train_df_copy, inplace=True)
    
    pass
    
    # Save cleaned datasets
    train_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_train.csv', index=False)
    test_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import re
    
    # Load the cleaned data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_test.csv')
    
    # Make copies of the DataFrames
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    # Function to extract titles from names
    def extract_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""
    
    # Extract titles
    train_df_copy['Title'] = train_df_copy['Name'].apply(extract_title)
    test_df_copy['Title'] = test_df_copy['Name'].apply(extract_title)
    
    # Standardize titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 
        'Master': 'Master', 'Dr': 'Rare', 'Rev': 'Rare', 
        'Col': 'Rare', 'Major': 'Rare', 'Mlle': 'Miss', 
        'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare', 
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 
        'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
    }
    
    train_df_copy['Title'] = train_df_copy['Title'].map(title_mapping)
    test_df_copy['Title'] = test_df_copy['Title'].map(title_mapping)
    
    # Create FamilySize feature
    train_df_copy['FamilySize'] = train_df_copy['SibSp'] + train_df_copy['Parch'] + 1
    test_df_copy['FamilySize'] = test_df_copy['SibSp'] + test_df_copy['Parch'] + 1
    
    # Create IsAlone feature
    train_df_copy['IsAlone'] = 0
    train_df_copy.loc[train_df_copy['FamilySize'] == 1, 'IsAlone'] = 1
    test_df_copy['IsAlone'] = 0
    test_df_copy.loc[test_df_copy['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Create FarePerPerson feature
    train_df_copy['FarePerPerson'] = train_df_copy['Fare'] / train_df_copy['FamilySize']
    test_df_copy['FarePerPerson'] = test_df_copy['Fare'] / test_df_copy['FamilySize']
    
    print("New features created successfully.")
    
    
    # Binning Age
    age_bins = [0, 12, 20, 40, 60, 80]
    age_labels = ['Child', 'Teen', 'Adult', 'MiddleAged', 'Senior']
    train_df_copy['AgeGroup'] = pd.cut(train_df_copy['Age'], bins=age_bins, labels=age_labels, right=False)
    test_df_copy['AgeGroup'] = pd.cut(test_df_copy['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Binning Fare
    fare_bins = [-1, 7.91, 14.454, 31, 512]
    fare_labels = ['Low', 'Medium', 'High', 'VeryHigh']
    train_df_copy['FareGroup'] = pd.qcut(train_df_copy['Fare'], q=4, labels=fare_labels)
    test_df_copy['FareGroup'] = pd.qcut(test_df_copy['Fare'], q=4, labels=fare_labels)
    
    print("Existing features transformed successfully.")
    
    
    # Handle categorical variables with one-hot encoding
    categorical_columns = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
    train_df_encoded = one_hot_encode(train_df_copy, columns=categorical_columns, handle_unknown='ignore')
    test_df_encoded = one_hot_encode(test_df_copy, columns=categorical_columns, handle_unknown='ignore')
    
    print("Categorical variables handled successfully.")
    
    
    # Normalize or standardize numerical features
    numerical_columns = ['Age', 'Fare', 'FarePerPerson']
    train_df_scaled = scale_features(train_df_encoded, columns=numerical_columns, method='standard')
    test_df_scaled = scale_features(test_df_encoded, columns=numerical_columns, method='standard')
    
    print("Numerical features normalized/standardized successfully.")
    
    
    # Save the processed datasets
    train_df_scaled.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/processed_train.csv', index=False)
    test_df_scaled.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/processed_test.csv', index=False)
    
    print("Processed datasets saved successfully.")
    


if __name__ == "__main__":
    generated_code_function()
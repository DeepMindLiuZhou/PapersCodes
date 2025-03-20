import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load data
    train_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/train.csv')
    test_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/test.csv')
    
    # Remove duplicates using the provided tool
    train_data = remove_duplicates(data=train_data)
    test_data = remove_duplicates(data=test_data)
    
    # Print the shape to verify the removal of duplicates
    pass
    
    
    # Detect and handle outliers in train and test data using the provided tool
    train_data = detect_and_handle_outliers_iqr(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    test_data = detect_and_handle_outliers_iqr(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    
    # Print a summary to verify outliers handling
    pass
    
    
    # Ensure consistency in data types for train and test data using the provided tool
    train_data = convert_data_types(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    train_data = convert_data_types(data=train_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    test_data = convert_data_types(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    test_data = convert_data_types(data=test_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    # Print data types to verify consistency
    pass
    
    
    # Align categorical values for train and test data
    train_data['Geography'] = pd.Categorical(train_data['Geography'])
    train_data['Gender'] = pd.Categorical(train_data['Gender'])
    
    test_data['Geography'] = pd.Categorical(test_data['Geography'])
    test_data['Gender'] = pd.Categorical(test_data['Gender'])
    
    # Print unique values to verify alignment
    pass
    
    
    # Save cleaned data
    train_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv', index=False)
    test_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    
    # Load cleaned data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv')
    
    # Ensure working on a copy
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Create AgeGroup feature
    age_bins = [18, 30, 50, np.inf]
    age_labels = ['Young', 'Middle-aged', 'Senior']
    train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels)
    test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels)
    
    # Create HasBalance feature
    train_df['HasBalance'] = np.where(train_df['Balance'] > 0, 1, 0)
    test_df['HasBalance'] = np.where(test_df['Balance'] > 0, 1, 0)
    
    # Create Region_Balance_Interaction feature
    train_df['Region_Balance_Interaction'] = train_df['Geography'] + '_' + train_df['HasBalance'].astype(str)
    test_df['Region_Balance_Interaction'] = test_df['Geography'] + '_' + test_df['HasBalance'].astype(str)
    
    # Create Active_Card_User feature
    train_df['Active_Card_User'] = np.where((train_df['HasCrCard'] == 1) & (train_df['IsActiveMember'] == 1), 1, 0)
    test_df['Active_Card_User'] = np.where((test_df['HasCrCard'] == 1) & (test_df['IsActiveMember'] == 1), 1, 0)
    
    # Verify new features
    print(train_df[['AgeGroup', 'HasBalance', 'Region_Balance_Interaction', 'Active_Card_User']].head())
    print(test_df[['AgeGroup', 'HasBalance', 'Region_Balance_Interaction', 'Active_Card_User']].head())
    
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize CreditScore
    scaler = MinMaxScaler()
    train_df['CreditScore'] = scaler.fit_transform(train_df[['CreditScore']])
    test_df['CreditScore'] = scaler.transform(test_df[['CreditScore']])
    
    # Log transform EstimatedSalary
    train_df['EstimatedSalary'] = np.log1p(train_df['EstimatedSalary'])
    test_df['EstimatedSalary'] = np.log1p(test_df['EstimatedSalary'])
    
    # Verify transformations
    print(train_df[['CreditScore', 'EstimatedSalary']].head())
    print(test_df[['CreditScore', 'EstimatedSalary']].head())
    
    
    from sklearn.preprocessing import LabelEncoder
    
    # One-Hot Encode Geography
    train_df = pd.get_dummies(train_df, columns=['Geography'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Geography'], drop_first=True)
    
    # Encode Gender
    le = LabelEncoder()
    train_df['Gender'] = le.fit_transform(train_df['Gender'])
    test_df['Gender'] = le.transform(test_df['Gender'])
    
    # Verify encoding
    print(train_df.head())
    print(test_df.head())
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Standardize Age and Balance
    scaler_age_balance = StandardScaler()
    train_df[['Age', 'Balance']] = scaler_age_balance.fit_transform(train_df[['Age', 'Balance']])
    test_df[['Age', 'Balance']] = scaler_age_balance.transform(test_df[['Age', 'Balance']])
    
    # Verify standardization
    print(train_df[['Age', 'Balance']].head())
    print(test_df[['Age', 'Balance']].head())
    
    
    # Save processed data
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/processed_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/processed_test.csv', index=False)
    
    print("Processed data saved successfully.")
    


if __name__ == "__main__":
    generated_code_function()
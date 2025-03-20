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
    print("Train data shape after removing duplicates:", train_data.shape)
    print("Test data shape after removing duplicates:", test_data.shape)
    
    
    # Detect and handle outliers in train and test data using the provided tool
    train_data = detect_and_handle_outliers_iqr(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    test_data = detect_and_handle_outliers_iqr(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    
    # Print a summary to verify outliers handling
    print("Outliers handled in train and test data.")
    
    
    # Ensure consistency in data types for train and test data using the provided tool
    train_data = convert_data_types(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    train_data = convert_data_types(data=train_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    test_data = convert_data_types(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    test_data = convert_data_types(data=test_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    # Print data types to verify consistency
    print("Data types after conversion in train data:\n", train_data.dtypes)
    print("Data types after conversion in test data:\n", test_data.dtypes)
    
    
    # Align categorical values for train and test data
    train_data['Geography'] = pd.Categorical(train_data['Geography'])
    train_data['Gender'] = pd.Categorical(train_data['Gender'])
    
    test_data['Geography'] = pd.Categorical(test_data['Geography'])
    test_data['Gender'] = pd.Categorical(test_data['Gender'])
    
    # Print unique values to verify alignment
    print("Unique values in Geography (train):", train_data['Geography'].unique())
    print("Unique values in Geography (test):", test_data['Geography'].unique())
    print("Unique values in Gender (train):", train_data['Gender'].unique())
    print("Unique values in Gender (test):", test_data['Gender'].unique())
    
    
    # Save cleaned data
    train_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv', index=False)
    test_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv', index=False)
    
    print("Cleaned data saved successfully.")
    


    
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
    


    
    import pandas as pd
    
    # Load processed data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/processed_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/processed_test.csv')
    
    # Ensure working on a copy
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Verify the presence of the 'Exited' column
    if 'Exited' not in train_df.columns:
        raise KeyError("The target column 'Exited' is missing from the training data.")
    
    # Separate the target variable
    y_train = train_df['Exited']
    X_train = train_df.drop(columns=['Exited', 'id', 'CustomerId', 'Surname'])
    X_test = test_df.drop(columns=['id', 'CustomerId', 'Surname'])
    
    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    
    # Align the train and test dataframes by the columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Verify the shapes of the datasets
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Define the models to be trained
    selected_models = ["logistic regression", "random forest", "XGBoost"]
    
    # Train and select the best model
    best_model = train_and_validation_and_select_the_best_model(
        X=X_train,
        y=y_train,
        problem_type='binary',
        selected_models=selected_models
    )
    
    print(f"The best performing model is: {best_model}")
    
    # The validation is already performed within the previous step using the train_and_validation_and_select_the_best_model tool.
    # No additional code is required for this step as the tool handles validation internally.
    
    # Make predictions on the test set
    predictions = best_model.predict_proba(X_test)[:, 1]  # get the probability of the positive class
    
    # Prepare the submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Exited': predictions
    })
    
    # Save the submission file
    submission_file_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/submission.csv'
    submission_df.to_csv(submission_file_path, index=False)
    
    print(f"Submission file saved to {submission_file_path}")
    


if __name__ == "__main__":
    generated_code_function()
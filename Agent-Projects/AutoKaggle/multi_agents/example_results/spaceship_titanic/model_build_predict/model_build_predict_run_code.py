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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Make copies of the dataframes
    train_clean = train.copy()
    test_clean = test.copy()
    
    # Define columns to fill missing values
    categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Fill missing values for categorical columns using mode
    train_clean = fill_missing_values(train_clean, categorical_columns, method='mode')
    test_clean = fill_missing_values(test_clean, categorical_columns, method='mode')
    
    # Fill missing values for numerical columns using median
    train_clean = fill_missing_values(train_clean, numerical_columns, method='median')
    test_clean = fill_missing_values(test_clean, numerical_columns, method='median')
    
    pass
    
    
    # Define columns to treat outliers
    outlier_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Treat outliers by clipping them to the acceptable range
    train_clean = detect_and_handle_outliers_iqr(train_clean, outlier_columns, factor=1.5, method='clip')
    test_clean = detect_and_handle_outliers_iqr(test_clean, outlier_columns, factor=1.5, method='clip')
    
    pass
    
    
    # Set expense values to 0 for CryoSleep passengers
    expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Update the training set
    train_clean.loc[train_clean['CryoSleep'] == True, expense_features] = 0
    
    # Update the test set
    test_clean.loc[test_clean['CryoSleep'] == True, expense_features] = 0
    
    pass
    
    
    # Remove duplicates
    train_clean = remove_duplicates(train_clean)
    test_clean = remove_duplicates(test_clean)
    
    # Convert data types
    boolean_columns = ['CryoSleep', 'VIP']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_columns = ['HomePlanet', 'Cabin', 'Destination', 'Name']
    
    # Convert boolean columns
    train_clean = convert_data_types(train_clean, boolean_columns, target_type='bool')
    test_clean = convert_data_types(test_clean, boolean_columns, target_type='bool')
    
    # Convert numerical columns
    train_clean = convert_data_types(train_clean, numerical_columns, target_type='float')
    test_clean = convert_data_types(test_clean, numerical_columns, target_type='float')
    
    # Convert categorical columns
    train_clean = convert_data_types(train_clean, categorical_columns, target_type='str')
    test_clean = convert_data_types(test_clean, categorical_columns, target_type='str')
    
    pass
    
    
    # Save cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_test.csv'
    
    train_clean.to_csv(cleaned_train_path, index=False)
    test_clean.to_csv(cleaned_test_path, index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    
    # Load cleaned data
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_test.csv'
    train = pd.read_csv(cleaned_train_path)
    test = pd.read_csv(cleaned_test_path)
    
    # Make copies of the dataframes
    train_fe = train.copy()
    test_fe = test.copy()
    
    # STEP 1: Create new features
    try:
        # Create binary flags for expenditure columns
        expenditure_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in expenditure_columns:
            train_fe[f'{col}_flag'] = train_fe[col] > 0
            test_fe[f'{col}_flag'] = test_fe[col] > 0
    
        # Create age groups
        age_bins = [0, 18, 35, 50, 65, np.inf]
        age_labels = ['0-18', '19-35', '36-50', '51-65', '65+']
        train_fe['AgeGroup'] = pd.cut(train_fe['Age'], bins=age_bins, labels=age_labels)
        test_fe['AgeGroup'] = pd.cut(test_fe['Age'], bins=age_bins, labels=age_labels)
    
        # Decompose Cabin feature
        def decompose_cabin(cabin):
            if pd.isna(cabin):
                return pd.Series([np.nan, np.nan, np.nan])
            parts = cabin.split('/')
            return pd.Series(parts)
    
        train_fe[['Cabin_deck', 'Cabin_number', 'Cabin_side']] = train_fe['Cabin'].apply(decompose_cabin)
        test_fe[['Cabin_deck', 'Cabin_number', 'Cabin_side']] = test_fe['Cabin'].apply(decompose_cabin)
    
        pass
    except Exception as e:
        pass
    
    # Save intermediate datasets
    intermediate_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_train_step1.csv'
    intermediate_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_test_step1.csv'
    train_fe.to_csv(intermediate_train_path, index=False)
    test_fe.to_csv(intermediate_test_path, index=False)
    
    
    # STEP 2: Transform existing features
    try:
        # Load intermediate data
        train_fe = pd.read_csv(intermediate_train_path)
        test_fe = pd.read_csv(intermediate_test_path)
    
        # Apply log transformation to expenditure features
        for col in expenditure_columns:
            train_fe[f'{col}_log'] = np.log1p(train_fe[col])
            test_fe[f'{col}_log'] = np.log1p(test_fe[col])
    
        def one_hot_encode(df, columns):
            one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
            encoded_df = pd.DataFrame(one_hot_encoder.fit_transform(df[columns]))
            encoded_df.columns = one_hot_encoder.get_feature_names_out(columns)
            df = df.drop(columns, axis=1)
            df = pd.concat([df, encoded_df], axis=1)
            return df
    
        def label_encode(df, columns):
            label_encoder = LabelEncoder()
            for column in columns:
                df[column] = label_encoder.fit_transform(df[column].astype(str))
            return df
    
        # One-hot encode categorical features with fewer unique values
        ohe_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
        train_fe = one_hot_encode(train_fe, ohe_columns)
        test_fe = one_hot_encode(test_fe, ohe_columns)
    
        # Label encode categorical features with many unique values
        le_columns = ['Cabin_deck', 'Cabin_side', 'AgeGroup']
        train_fe = label_encode(train_fe, le_columns)
        test_fe = label_encode(test_fe, le_columns)
    
        pass
    except Exception as e:
        pass
    
    # Save intermediate datasets
    intermediate_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_train_step2.csv'
    intermediate_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_test_step2.csv'
    train_fe.to_csv(intermediate_train_path, index=False)
    test_fe.to_csv(intermediate_test_path, index=False)
    
    
    # STEP 3: Normalize or standardize numerical features
    try:
        # Load intermediate data
        train_fe = pd.read_csv(intermediate_train_path)
        test_fe = pd.read_csv(intermediate_test_path)
    
        # Scale numerical features
        numerical_columns = ['Age', 'RoomService_log', 'FoodCourt_log', 'ShoppingMall_log', 'Spa_log', 'VRDeck_log']
        train_fe = scale_features(train_fe, numerical_columns, method='standard')
        test_fe = scale_features(test_fe, numerical_columns, method='standard')
    
        pass
    except Exception as e:
        pass
    
    # Save intermediate datasets
    intermediate_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_train_step3.csv'
    intermediate_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/intermediate_test_step3.csv'
    train_fe.to_csv(intermediate_train_path, index=False)
    test_fe.to_csv(intermediate_test_path, index=False)
    
    
    # STEP 4: Feature selection
    try:
        # Load intermediate data
        train_fe = pd.read_csv(intermediate_train_path)
        test_fe = pd.read_csv(intermediate_test_path)
    
        # Perform feature selection based on correlation with the target
        selected_features = correlation_feature_selection(train_fe, target='Transported', threshold=0.5)
    
        # Select only the relevant features in the train and test datasets
        train_fe = train_fe[selected_features + ['Transported']]
        test_fe = test_fe[selected_features]
    
        pass
    except Exception as e:
        pass
    
    # Save final processed datasets
    processed_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/processed_train.csv'
    processed_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/processed_test.csv'
    train_fe.to_csv(processed_train_path, index=False)
    test_fe.to_csv(processed_test_path, index=False)
    
    pass
    


    
    import pandas as pd
    
    # Load processed data
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/processed_train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/processed_test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate target variable
    y = train_df.pop('Transported')
    
    # Drop non-numeric columns and target variable
    X_train = train_df.drop(columns=['PassengerId', 'Name', 'Cabin'])
    X_test = test_df.drop(columns=['PassengerId', 'Name', 'Cabin'])
    
    print("Data Preparation for Model Training completed successfully.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y shape:", y.shape)
    
    
    # Using the provided automated tool for model training and selection
    best_model = train_and_validation_and_select_the_best_model(
        X=X_train,
        y=y,
        problem_type='binary',
        selected_models=["logistic regression", "random forest", "XGBoost"]
    )
    
    print("Model Training completed successfully.")
    print("Best Model:", best_model)
    
    
    # The best model has already been selected in the previous step.
    # Document the selected model and its performance metrics.
    print("Model Validation and Selection completed successfully.")
    print("Selected Best Model:", best_model)
    
    
    # Make predictions on the test dataset using the best model
    predictions = best_model.predict(X_test)
    
    # Prepare the submission file
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predictions
    })
    
    submission_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print("Prediction on Test Data completed successfully.")
    print("Submission file saved at:", submission_path)
    


if __name__ == "__main__":
    generated_code_function()
import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    import numpy as np
    
    # Load the data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/test.csv')
    
    # Define columns with missing values
    numerical_cols_with_missing = [
        'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
    ]
    
    categorical_cols_with_missing = [
        'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
        'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
    ]
    
    # Handle missing values for numerical columns using median
    train_df = fill_missing_values(train_df, numerical_cols_with_missing, method='median')
    test_df = fill_missing_values(test_df, numerical_cols_with_missing, method='median')
    
    # Handle missing values for categorical columns using 'Missing'
    train_df = fill_missing_values(train_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    test_df = fill_missing_values(test_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    
    # Remove columns with more than 50% missing values if needed
    train_df = remove_columns_with_missing_data(train_df, thresh=0.5)
    test_df = remove_columns_with_missing_data(test_df, thresh=0.5)
    
    pass
    
    # Define columns to check for outliers
    outlier_columns = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
    
    # Handle outliers by capping them to the IQR bounds
    train_df = detect_and_handle_outliers_iqr(train_df, outlier_columns, factor=1.5, method='clip')
    test_df = detect_and_handle_outliers_iqr(test_df, outlier_columns, factor=1.5, method='clip')
    
    pass
    
    # Convert data types if necessary
    numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = train_df.select_dtypes(include=[object]).columns.tolist()
    
    train_df = convert_data_types(train_df, numerical_columns, target_type='float')
    # Assuming numerical_columns is a list of numerical columns including 'SalePrice'
    existing_columns = [col for col in numerical_columns if col in test_df.columns]
    if existing_columns:
        test_df = convert_data_types(test_df, existing_columns, target_type='float')
    else:
        pass
    
    train_df = convert_data_types(train_df, categorical_columns, target_type='str')
    
    # Ensure categorical_columns only contains columns present in both train_df and test_df
    common_categorical_columns = [col for col in categorical_columns if col in train_df.columns and col in test_df.columns]
    
    # Now call the convert_data_types function with the cleaned list
    test_df = convert_data_types(test_df, common_categorical_columns, target_type='str')
    
    # Drop 'FireplaceQu' from train_df if it is not in test_df
    if 'FireplaceQu' not in test_df.columns and 'FireplaceQu' in train_df.columns:
        train_df = train_df.drop(columns=['FireplaceQu'])
    
    # Remove duplicates if any
    train_df = remove_duplicates(train_df, columns=None)
    test_df = remove_duplicates(test_df, columns=None)
    
    pass
    
    # Save cleaned datasets
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    
    # Load the cleaned datasets
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test.csv')
    
    # Function to create new features
    def create_new_features(df):
        df_copy = df.copy()
        df_copy['HouseAge'] = df_copy['YrSold'] - df_copy['YearBuilt']
        df_copy['YearsSinceRemod'] = df_copy['YrSold'] - df_copy['YearRemodAdd']
        df_copy['TotalBath'] = df_copy['FullBath'] + 0.5 * df_copy['HalfBath'] + df_copy['BsmtFullBath'] + 0.5 * df_copy['BsmtHalfBath']
        df_copy['TotalPorchSF'] = df_copy['OpenPorchSF'] + df_copy['EnclosedPorch'] + df_copy['3SsnPorch'] + df_copy['ScreenPorch']
        df_copy['TotalSF'] = df_copy['TotalBsmtSF'] + df_copy['1stFlrSF'] + df_copy['2ndFlrSF']
        df_copy['OverallQual_SF'] = df_copy['OverallQual'] * df_copy['GrLivArea']
        return df_copy
    
    # Create new features for both train and test datasets
    train_df = create_new_features(train_df)
    test_df = create_new_features(test_df)
    
    # Save the datasets with new features
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_with_new_features.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_with_new_features.csv', index=False)
    
    pass
    
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Load the datasets with new features
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_with_new_features.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_with_new_features.csv')
    
    # Function to apply log1p transformation
    def log_transform(df, columns):
        df_copy = df.copy()
        for col in columns:
            df_copy[col] = np.log1p(df_copy[col])
        return df_copy
    
    # Columns to transform
    log_transform_columns = ['LotArea', 'GrLivArea']
    train_log_transform_columns = log_transform_columns + ['SalePrice']
    
    # Apply log transformation
    train_df_transformed = log_transform(train_df, train_log_transform_columns)
    test_df_transformed = log_transform(test_df, log_transform_columns)
    
    # Save the datasets
    train_df_transformed.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_log_transformed.csv', index=False)
    test_df_transformed.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_log_transformed.csv', index=False)
    
    pass
    
    
    from sklearn.preprocessing import LabelEncoder
    
    # Load the transformed datasets
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_log_transformed.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_log_transformed.csv')
    
    # Function to label encode ordinal features
    def label_encode_features(df, columns):
        df_copy = df.copy()
        le = LabelEncoder()
        for col in columns:
            df_copy[col] = le.fit_transform(df_copy[col])
        return df_copy
    
    # Ordinal features to label encode
    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
    
    # Apply label encoding
    train_df_encoded = label_encode_features(train_df, ordinal_features)
    test_df_encoded = label_encode_features(test_df, ordinal_features)
    
    # Function to one-hot encode nominal features
    def one_hot_encode_features(train_df, test_df, columns, target_column='SalePrice'):
        train_df_copy = pd.get_dummies(train_df, columns=columns)
        
        # Ensure the target column is removed from the test dataset
        if target_column in test_df.columns:
            test_df = test_df.drop(columns=[target_column])
        
        test_df_copy = pd.get_dummies(test_df, columns=columns)
        
        # Align the columns of test_df with train_df
        test_df_copy = test_df_copy.reindex(columns=train_df_copy.columns, fill_value=0)
        
        # Ensure the target column is removed after reindexing
        if target_column in test_df_copy.columns:
            test_df_copy = test_df_copy.drop(columns=[target_column])
        
        return train_df_copy, test_df_copy
    
    # Nominal features to one-hot encode
    nominal_features = ['MSZoning', 'Street', 'Alley', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'Condition1', 'Condition2', 'BldgType']
    
    # Make sure the target column is not present in the test dataset before encoding
    if 'SalePrice' in test_df_encoded.columns:
        test_df_encoded = test_df_encoded.drop(columns=['SalePrice'])
    
    train_df_encoded, test_df_encoded = one_hot_encode_features(train_df_encoded, test_df_encoded, nominal_features)
    
    # Save the encoded datasets
    train_df_encoded.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_encoded.csv', index=False)
    test_df_encoded.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_encoded.csv', index=False)
    
    pass
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Load the encoded datasets
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train_encoded.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test_encoded.csv')
    
    # Ensure the target column is removed from the test dataset before standardizing
    if 'SalePrice' in test_df.columns:
        test_df = test_df.drop(columns=['SalePrice'])
    
    # Columns to standardize
    numerical_features = ['LotArea', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'HouseAge', 'OverallQual_SF']
    
    # Ensure all numerical features exist in both datasets
    numerical_features = [feature for feature in numerical_features if feature in train_df.columns and feature in test_df.columns]
    
    # Standardize the numerical features
    scaler = StandardScaler()
    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])
    
    # Save the standardized datasets
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/processed_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/processed_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    
    # Load the datasets
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/processed_train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/processed_test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate the target variable
    y_train = train_df['SalePrice']
    X_train = train_df.drop(columns=['SalePrice'])
    
    # Identify non-numeric columns
    non_numeric_columns = X_train.select_dtypes(exclude=['number']).columns
    
    # Remove non-numeric columns from both datasets
    X_train = X_train.drop(columns=non_numeric_columns)
    X_test = test_df.drop(columns=non_numeric_columns)
    
    # Ensure consistent features between train and test sets
    assert set(X_train.columns) == set(X_test.columns), "Mismatch in features between train and test sets"
    
    print("Data Preparation completed successfully.")
    
    
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }
    
    print("Model Selection completed successfully. Models selected:")
    print("\n".join(models.keys()))
    
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    # Function to calculate RMSE
    def rmse_cv(model, X, y):
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
        return rmse
    
    # Train and validate the models
    model_performance = {}
    for name, model in models.items():
        score = rmse_cv(model, X_train, y_train)
        model_performance[name] = score
        print(f"{name} RMSE: {score.mean():.4f} (+/- {score.std():.4f})")
    
    # Identify the best performing model
    best_model_name = min(model_performance, key=lambda k: model_performance[k].mean())
    best_model = models[best_model_name]
    
    print(f"Best model: {best_model_name}")
    
    
    import numpy as np
    import pandas as pd
    
    # Fit the best model on the entire training set
    best_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = best_model.predict(X_test)
    
    # Reverse the log transformation
    predictions = np.exp(predictions)
    
    # Prepare the submission file
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions
    })
    
    # Save the submission file
    submission_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Predictions and submission file prepared successfully. Submission saved to: {submission_path}")
    


if __name__ == "__main__":
    generated_code_function()
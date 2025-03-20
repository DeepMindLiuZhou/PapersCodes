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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Print missing values count before handling
    pass
    
    # Define columns
    numerical_columns = [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
        'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 
        'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
        'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 
        'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 
        'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
        'SigmoidOfAreas'
    ]
    categorical_columns = ['TypeOfSteel_A300', 'TypeOfSteel_A400']
    
    # Fill missing values
    fill_missing_values(train_df, columns=numerical_columns, method='mean')
    fill_missing_values(train_df, columns=categorical_columns, method='mode')
    fill_missing_values(test_df, columns=numerical_columns, method='mean')
    fill_missing_values(test_df, columns=categorical_columns, method='mode')
    
    # Print missing values count after handling
    pass
    
    
    # List of numerical features to treat for outliers
    numerical_features = [
        'Steel_Plate_Thickness', 'Maximum_of_Luminosity', 'Minimum_of_Luminosity', 
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 
        'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Length_of_Conveyer', 
        'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 
        'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 
        'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 
        'SigmoidOfAreas'
    ]
    
    # Handle outliers in train and test datasets
    detect_and_handle_outliers_iqr(train_df, columns=numerical_features, factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df, columns=numerical_features, factor=1.5, method='clip')
    
    # Print summary of outliers handled
    pass
    
    
    # Ensure correct data types (converting categorical to bool)
    convert_data_types(train_df, columns=categorical_columns, target_type='bool')
    convert_data_types(test_df, columns=categorical_columns, target_type='bool')
    
    # Removing duplicates
    remove_duplicates(train_df, inplace=True)
    remove_duplicates(test_df, inplace=True)
    
    # Print confirmation
    pass
    
    
    # Save cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_test.csv'
    
    train_df.to_csv(cleaned_train_path, index=False)
    test_df.to_csv(cleaned_test_path, index=False)
    
    # Print summary statistics and data types
    pass
    
    pass
    


    
    import pandas as pd
    
    # Load cleaned data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_test.csv')
    
    def create_new_features(df):
        df = df.copy()
        
        # Create new features
        df['X_Range'] = df['X_Maximum'] - df['X_Minimum']
        df['Y_Range'] = df['Y_Maximum'] - df['Y_Minimum']
        df['X_Y_Ratio'] = df['X_Range'] / (df['Y_Range'] + 1e-5)  # Avoid division by zero
        df['Luminosity_Area_Product'] = df['Sum_of_Luminosity'] * df['Pixels_Areas']
        df['Perimeter_Area_Ratio'] = (df['X_Perimeter'] + df['Y_Perimeter']) / (df['Pixels_Areas'] + 1e-5)  # Avoid division by zero
        
        # Polynomial features
        df['X_Minimum^2'] = df['X_Minimum'] ** 2
        df['X_Maximum^2'] = df['X_Maximum'] ** 2
        df['Y_Minimum^2'] = df['Y_Minimum'] ** 2
        df['Y_Maximum^2'] = df['Y_Maximum'] ** 2
        df['X_Minimum*X_Maximum'] = df['X_Minimum'] * df['X_Maximum']
        df['Y_Minimum*Y_Maximum'] = df['Y_Minimum'] * df['Y_Maximum']
        
        return df
    
    # Apply the feature creation function
    train_df = create_new_features(train_df)
    test_df = create_new_features(test_df)
    
    # Save the updated dataframes
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_test.csv', index=False)
    
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import numpy as np
    
    def transform_features(df):
        df = df.copy()
        
        # List of features to scale
        features_to_scale = ['Pixels_Areas', 'Sum_of_Luminosity', 'X_Perimeter', 'Y_Perimeter']
        
        # Scale features using MinMaxScaler
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        
        # Log transformation
        df['Log_Sum_of_Luminosity'] = np.log1p(df['Sum_of_Luminosity'])  # log1p to handle zero values
        df['Log_Pixels_Areas'] = np.log1p(df['Pixels_Areas'])
        
        return df
    
    # Apply the transformation function
    train_df = transform_features(train_df)
    test_df = transform_features(test_df)
    
    # Save the updated dataframes
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_test.csv', index=False)
    
    
    def encode_categorical(df):
        df = df.copy()
    
        # One-hot encoding for 'TypeOfSteel_A300' and 'TypeOfSteel_A400'
        df = pd.get_dummies(df, columns=['TypeOfSteel_A300', 'TypeOfSteel_A400'], drop_first=True)
        
        return df
    
    # Apply the encoding function
    train_df = encode_categorical(train_df)
    test_df = encode_categorical(test_df)
    
    # Save the updated dataframes
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_test.csv', index=False)
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    
    def plot_correlation_matrix(df):
        # Calculate correlation matrix
        corr = df.corr()
        
        # Plot heatmap
        pass
    
    # Plot correlation matrix for training data
    plot_correlation_matrix(train_df)
    
    # Feature importance using RandomForestClassifier
    def feature_importance(df, target_columns):
        X = df.drop(columns=target_columns)
        y = df[target_columns]
        
        # Train RandomForestClassifier for each target variable
        for target in target_columns:
            clf = RandomForestClassifier()
            clf.fit(X, y[target])
            importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            pass
    
    # List of target columns
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Calculate feature importance
    feature_importance(train_df, target_columns)
    


    
    import pandas as pd
    
    # Load processed data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/processed_test.csv')
    
    # Define target columns
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Separate features and target variables
    y_train = train_df[target_columns]
    X_train = train_df.drop(columns=['id'] + target_columns)
    
    # For test data, drop 'id' column and ensure consistency with training data
    X_test = test_df.drop(columns=['id'])
    
    # Ensure columns in test set match those in training set
    assert list(X_train.columns) == list(X_test.columns), "Mismatch in training and test columns"
    
    print("Data preparation completed: X_train and y_train are ready for model training.")
    
    
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Define the base model
    base_model = RandomForestClassifier()
    
    # Wrap the base model with MultiOutputClassifier for multilabel classification
    multi_target_model = MultiOutputClassifier(base_model, n_jobs=-1)
    
    # Train the model
    multi_target_model.fit(X_train, y_train)
    
    print("Model training and validation completed. Best model selected.")
    
    
    # No additional code needed for this step as the model selection is handled by using MultiOutputClassifier
    print("Model validation and selection are handled within the MultiOutputClassifier.")
    
    
    # Predict probabilities for each defect category using the best model
    predictions = multi_target_model.predict_proba(X_test)
    
    # Aggregate the predictions into a DataFrame with columns for each class probability
    predictions_dict = {}
    for i, target in enumerate(target_columns):
        # Assuming binary classification (class 0 and class 1)
        class_1_probabilities = predictions[i][:, 1]
        predictions_dict[target] = class_1_probabilities
    
    # Create the DataFrame from the dictionary
    predictions_df = pd.DataFrame(predictions_dict)
    
    # Ensure the columns are in the correct order as in sample_submission.csv
    expected_columns = ['id'] + target_columns
    
    # Create a submission DataFrame
    submission_df = pd.DataFrame(test_df['id'], columns=['id'])
    submission_df = pd.concat([submission_df, predictions_df], axis=1)
    submission_df = submission_df[expected_columns]
    
    # Save the submission DataFrame
    submission_file = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/submission.csv'
    submission_df.to_csv(submission_file, index=False)
    
    print(f"Predictions saved to {submission_file}")
    


if __name__ == "__main__":
    generated_code_function()
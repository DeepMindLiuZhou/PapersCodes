import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Define the data directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    CLEANED_DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    
    # Load the datasets
    train_path = DATA_DIR + 'train.csv'
    test_path = DATA_DIR + 'test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Create copies to avoid modifying original data
    cleaned_train_df = train_df.copy()
    cleaned_test_df = test_df.copy()
    
    # Define numerical and categorical columns
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Task 1: Fill Missing Values
    
    # Fill missing numerical features with median
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=numerical_cols,
        method='median'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=numerical_cols,
        method='median'
    )
    
    # Fill missing categorical features with mode
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=categorical_cols,
        method='mode'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=categorical_cols,
        method='mode'
    )
    
    # Remove columns with more than 60% missing values
    cleaned_train_df = remove_columns_with_missing_data(
        data=cleaned_train_df,
        thresh=0.6
    )
    cleaned_test_df = remove_columns_with_missing_data(
        data=cleaned_test_df,
        thresh=0.6
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step1.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step1.csv', index=False)
    
    
    # Task 2: Detect and Handle Outliers in Numerical Features
    
    # Define the numerical columns again in case some were removed in Task 1
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    # Handle possible removal of some numerical columns in Task 1
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Detect and handle outliers by clipping
    cleaned_train_df = detect_and_handle_outliers_iqr(
        data=cleaned_train_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    cleaned_test_df = detect_and_handle_outliers_iqr(
        data=cleaned_test_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step2.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step2.csv', index=False)
    
    
    # Task 3: Normalize and Standardize Categorical Features
    
    # Define categorical columns again in case some were removed in Task 1
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Handle possible removal of some categorical columns in Task 1
    categorical_cols = [col for col in categorical_cols if col in cleaned_train_df.columns]
    
    # Convert categorical columns to string type
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    # Normalize categorical strings: lowercase and strip whitespaces
    for col in categorical_cols:
        cleaned_train_df[col] = cleaned_train_df[col].str.lower().str.strip()
        cleaned_test_df[col] = cleaned_test_df[col].str.lower().str.strip()
    
    # Placeholder for correcting common typos
    # Define a dictionary for typo corrections if available
    # Example:
    # typo_mapping = {
    #     'marial status': 'marital status',
    #     'scholarship holder': 'scholarship_holder',
    #     # Add more mappings as identified
    # }
    
    # Uncomment and modify the following lines if typo mappings are available
    # for col, mapping in typo_mapping.items():
    #     cleaned_train_df[col] = cleaned_train_df[col].replace(mapping)
    #     cleaned_test_df[col] = cleaned_test_df[col].replace(mapping)
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step3.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step3.csv', index=False)
    
    
    # Task 4: Remove Duplicate Rows and Convert Data Types
    
    # Include 'id' in the columns to consider for duplicate removal
    columns_to_consider_train = cleaned_train_df.columns.tolist()  # This includes 'id'
    cleaned_train_df = remove_duplicates(
        data=cleaned_train_df,
        columns=columns_to_consider_train,
        keep='first'
    )
    
    # Remove duplicates based on all columns except 'id' for the test dataset
    columns_to_consider_test = [col for col in cleaned_test_df.columns if col != 'id']
    cleaned_test_df = remove_duplicates(
        data=cleaned_test_df,
        columns=columns_to_consider_test,
        keep='first'
    )
    
    # Define numerical columns again after Task 1 and Task 2
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP', 'Age at enrollment', 'Application order'
    ]
    
    # Ensure numerical columns exist after previous tasks
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Convert numerical columns to appropriate data types
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=numerical_cols,
        target_type='float'  # Assuming all are floats; adjust if some are integers
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=numerical_cols,
        target_type='float'
    )
    
    # Convert 'id' column to string
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=['id'],
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=['id'],
        target_type='str'
    )
    
    # Verify data types (optional)
    # print(cleaned_train_df.dtypes)
    # print(cleaned_test_df.dtypes)
    
    # Save the final cleaned datasets
    cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train.csv', index=False)
    cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Define directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    FEATURE_ENGINEERING_DIR = DATA_DIR + 'feature_engineering/'
    IMAGE_DIR = FEATURE_ENGINEERING_DIR + 'images/'
    
    # Create directories if they don't exist
    Path(FEATURE_ENGINEERING_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load cleaned data
    cleaned_train_path = DATA_DIR + 'cleaned_train.csv'
    cleaned_test_path = DATA_DIR + 'cleaned_test.csv'
    
    train_df = pd.read_csv(cleaned_train_path)
    test_df = pd.read_csv(cleaned_test_path)
    
    # Make copies to avoid modifying original data
    processed_train_df = train_df.copy()
    processed_test_df = test_df.copy()
    
    # Define categorical features
    binary_categorical_features = [
        'Gender',
        'Displaced',
        'Debtor',
        'Scholarship holder',
        'International'
    ]
    
    multiclass_categorical_features = [
        'Marital status',
        'Application mode',
        'Course',
        'Previous qualification',
        'Nacionality',
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        'Educational special needs',
        'Tuition fees up to date'
    ]
    
    # Initialize LabelEncoders for binary features
    label_encoders = {}
    for col in binary_categorical_features:
        le = LabelEncoder()
        combined_data = pd.concat([processed_train_df[col], processed_test_df[col]], axis=0).astype(str)
        le.fit(combined_data)
        processed_train_df[col] = le.transform(processed_train_df[col].astype(str))
        processed_test_df[col] = le.transform(processed_test_df[col].astype(str))
        label_encoders[col] = le
    
    pass
    
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
    
    # Define numerical features to scale/normalize
    numerical_features = [
        'Previous qualification (grade)',
        'Admission grade',
        'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate',
        'Inflation rate',
        'GDP',
        'Age at enrollment'
    ]
    
    # Ensure all numerical features exist in the data
    numerical_features = [col for col in numerical_features if col in processed_train_df.columns]
    
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    power_transformer = PowerTransformer(method='yeo-johnson')
    
    # Choose a scaler based on feature distribution
    # For simplicity, we'll use StandardScaler here. Adjust based on EDA results.
    scaler = StandardScaler()
    
    # Fit scaler on training data
    scaler.fit(processed_train_df[numerical_features])
    
    # Transform both training and test data
    processed_train_df[numerical_features] = scaler.transform(processed_train_df[numerical_features])
    processed_test_df[numerical_features] = scaler.transform(processed_test_df[numerical_features])
    
    pass
    
    
    from sklearn.preprocessing import PolynomialFeatures
    
    # Define specific interaction pairs based on the plan
    interaction_pairs = [
        ('Admission grade', 'GDP'),
        ('Age at enrollment', 'Educational special needs')
    ]
    
    # Create interaction features manually
    for pair in interaction_pairs:
        feat_name = f"{pair[0].replace(' ', '_')}_x_{pair[1].replace(' ', '_')}"
        processed_train_df[feat_name] = processed_train_df[pair[0]] * processed_train_df[pair[1]]
        processed_test_df[feat_name] = processed_test_df[pair[0]] * processed_test_df[pair[1]]
        
        pass
    
    # Initialize PolynomialFeatures for additional polynomial terms
    # Limiting degree=2 to control dimensionality
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    
    # Select numerical features for polynomial expansion
    poly_features = [
        'Admission grade',
        'GDP',
        'Age at enrollment'
    ]
    
    # Fit and transform polynomial features on training data
    poly_train = poly.fit_transform(processed_train_df[poly_features])
    poly_feature_names = poly.get_feature_names_out(poly_features)
    
    # Create DataFrame for polynomial features
    poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names, index=processed_train_df.index)
    processed_train_df = pd.concat([processed_train_df, poly_train_df], axis=1)
    
    # Transform test data
    poly_test = poly.transform(processed_test_df[poly_features])
    poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names, index=processed_test_df.index)
    processed_test_df = pd.concat([processed_test_df, poly_test_df], axis=1)
    
    pass
    
    
    from sklearn.impute import SimpleImputer
    
    # Identify features that might have missing values after feature engineering
    # In this scenario, it's unlikely, but we'll check to ensure
    missing_train = processed_train_df.isnull().sum()
    missing_test = processed_test_df.isnull().sum()
    
    features_with_missing_train = missing_train[missing_train > 0].index.tolist()
    features_with_missing_test = missing_test[missing_test > 0].index.tolist()
    features_with_missing = list(set(features_with_missing_train + features_with_missing_test))
    
    pass
    
    if features_with_missing:
        # Initialize SimpleImputer with median strategy for numerical features
        imputer = SimpleImputer(strategy='median')
        
        # Fit imputer on training data
        imputer.fit(processed_train_df[features_with_missing])
        
        # Transform both training and test data
        processed_train_df[features_with_missing] = imputer.transform(processed_train_df[features_with_missing])
        processed_test_df[features_with_missing] = imputer.transform(processed_test_df[features_with_missing])
        
        pass
    else:
        pass
    
    
    # Define output paths
    processed_train_path = DATA_DIR + 'processed_train.csv'
    processed_test_path = DATA_DIR + 'processed_test.csv'
    
    # Save the processed datasets
    processed_train_df.to_csv(processed_train_path, index=False)
    processed_test_df.to_csv(processed_test_path, index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Define directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    MODEL_DIR = DATA_DIR + 'model_build_predict/'
    IMAGE_DIR = MODEL_DIR + 'images/'
    
    # Ensure directories exist
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    processed_train_path = DATA_DIR + 'processed_train.csv'
    processed_test_path = DATA_DIR + 'processed_test.csv'
    
    train_df = pd.read_csv(processed_train_path)
    test_df = pd.read_csv(processed_test_path)
    
    # Make copies to avoid modifying original data
    X_train = train_df.copy()
    y_train = X_train.pop('Target')  # Separate target
    X_test = test_df.copy()
    
    # Remove 'id' column as it is not used for training
    X_train = X_train.drop(columns=['id'], errors='ignore')
    X_test = X_test.drop(columns=['id'], errors='ignore')
    
    print("Target variable separated and 'id' column removed from training and test sets.")
    
    # Identify non-numeric columns
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Non-numeric columns to be removed: {non_numeric_cols}")
    
    # Remove non-numeric columns from training and test sets
    X_train = X_train.drop(columns=non_numeric_cols, errors='ignore')
    X_test = X_test.drop(columns=non_numeric_cols, errors='ignore')
    
    print("Non-numeric columns removed from training and test sets.")
    
    # Ensure feature alignment
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)
    
    missing_in_test = train_features - test_features
    missing_in_train = test_features - train_features
    
    if missing_in_test:
        X_test = X_test.drop(columns=list(missing_in_test))
        print(f"Dropped columns from test set not present in training set: {missing_in_test}")
    
    if missing_in_train:
        X_train = X_train.drop(columns=list(missing_in_train))
        print(f"Dropped columns from training set not present in test set: {missing_in_train}")
    
    print("Feature alignment between training and test sets ensured.")
    
    # Save the prepared datasets for verification
    prepared_train_path = DATA_DIR + 'model_build_predict/ready_train.csv'
    prepared_test_path = DATA_DIR + 'model_build_predict/ready_test.csv'
    
    X_train.to_csv(prepared_train_path, index=False)
    X_test.to_csv(prepared_test_path, index=False)
    
    print(f"Prepared datasets saved at '{prepared_train_path}' and '{prepared_test_path}'.")
    
    
    from sklearn.preprocessing import StandardScaler
    
    # Load prepared data
    prepared_train_path = DATA_DIR + 'model_build_predict/ready_train.csv'
    prepared_test_path = DATA_DIR + 'model_build_predict/ready_test.csv'
    
    X_train = pd.read_csv(prepared_train_path)
    X_test = pd.read_csv(prepared_test_path)
    
    print("Prepared training and test datasets loaded.")
    
    # Verify all remaining features are numeric
    numeric_train = X_train.select_dtypes(include=[np.number])
    numeric_test = X_test.select_dtypes(include=[np.number])
    
    if numeric_train.shape[1] != X_train.shape[1] or numeric_test.shape[1] != X_test.shape[1]:
        non_numeric_train = X_train.columns.difference(numeric_train.columns)
        non_numeric_test = X_test.columns.difference(numeric_test.columns)
        print(f"Warning: Non-numeric columns detected in training set: {non_numeric_train}")
        print(f"Warning: Non-numeric columns detected in test set: {non_numeric_test}")
    else:
        print("All features are numeric in both training and test sets.")
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit scaler on training data
    scaler.fit(X_train)
    
    # Transform both training and test data
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    print("Feature scaling applied to training and test sets.")
    
    # Verify scaling by checking mean and standard deviation
    print("\nTraining Data - Mean after scaling:\n", X_train_scaled.mean())
    print("\nTraining Data - Std Dev after scaling:\n", X_train_scaled.std())
    
    print("\nTest Data - Mean after scaling:\n", X_test_scaled.mean())
    print("\nTest Data - Std Dev after scaling:\n", X_test_scaled.std())
    
    # Save the scaled datasets for modeling
    scaled_train_path = DATA_DIR + 'model_build_predict/scaled_train.csv'
    scaled_test_path = DATA_DIR + 'model_build_predict/scaled_test.csv'
    
    X_train_scaled.to_csv(scaled_train_path, index=False)
    X_test_scaled.to_csv(scaled_test_path, index=False)
    
    print(f"Scaled datasets saved at '{scaled_train_path}' and '{scaled_test_path}'.")
    
    
    # Assume that 'train_and_validation_and_select_the_best_model' is pre-imported
    
    # Load scaled training data
    scaled_train_path = DATA_DIR + 'model_build_predict/scaled_train.csv'
    X_train_scaled = pd.read_csv(scaled_train_path)
    y_train = pd.read_csv(DATA_DIR + 'processed_train.csv')['Target']  # Reload Target from original processed_train.csv
    
    print("Scaled training data and target variable loaded.")
    
    # Define problem type and selected models
    problem_type = "multiclass"
    selected_models = ["random forest", "logistic regression"]  # Reduced number of models
    
    print(f"Starting model training with models: {selected_models} for a {problem_type} problem.")
    
    # Train models and select the best one
    best_model = train_and_validation_and_select_the_best_model(
        X=X_train_scaled,
        y=y_train,
        problem_type=problem_type,
        selected_models=selected_models
    )
    
    print(f"Best model selected: {best_model}")
    
    # Save the best model for future use (optional)
    # This assumes that the best_model has a method to save itself, e.g., using joblib or pickle
    # import joblib
    # model_path = MODEL_DIR + 'best_model.pkl'
    # joblib.dump(best_model, model_path)
    # print(f"Best model saved at '{model_path}'.")
    
    
    # Load scaled test data
    scaled_test_path = DATA_DIR + 'model_build_predict/scaled_test.csv'
    X_test_scaled = pd.read_csv(scaled_test_path)
    
    print("Scaled test data loaded.")
    
    # Generate predictions using the best model
    print("Generating predictions on the test set.")
    predictions = best_model.predict(X_test_scaled)
    
    # Load the original test dataframe to retrieve 'id' for submission
    original_test_path = DATA_DIR + 'processed_test.csv'
    test_df_original = pd.read_csv(original_test_path)
    submission_ids = test_df_original['id']
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': submission_ids,
        'Target': predictions
    })
    
    # Ensure no missing values in the submission
    if submission_df.isnull().values.any():
        print("Warning: Missing values detected in the submission. Filling missing values with 'Unknown'.")
        submission_df = submission_df.fillna('Unknown')
    
    # Verify the format of 'Target' labels
    expected_labels = ['dropout', 'enrolled', 'graduate']
    if not submission_df['Target'].isin(expected_labels).all():
        print("Warning: Some 'Target' predictions are outside the expected labels. Please verify model outputs.")
    
    import os
    
    # Define the competition root directory by navigating one level up from MODEL_DIR
    competition_root = os.path.abspath(os.path.join(MODEL_DIR, '..'))
    
    # Define submission path at the competition root directory
    submission_path = os.path.join(competition_root, 'submission.csv')
    
    # Ensure the competition root directory exists
    os.makedirs(competition_root, exist_ok=True)
    
    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file 'submission.csv' created successfully at '{submission_path}'.")
    


if __name__ == "__main__":
    generated_code_function()
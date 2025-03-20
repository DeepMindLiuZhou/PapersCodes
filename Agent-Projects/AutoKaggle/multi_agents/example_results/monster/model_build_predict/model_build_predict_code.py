import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    import os
    
    # Define file paths
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    cleaned_train_file = os.path.join(data_dir, 'cleaned_train.csv')
    cleaned_test_file = os.path.join(data_dir, 'cleaned_test.csv')
    
    # Load the datasets
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    # Define numerical columns
    numerical_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
    
    # TASK 1: Detect and Handle Outliers in Numerical Features Using the IQR Method
    
    # Handle outliers in training data by capping them instead of removing
    try:
        train_df_original = train_df.copy()  # Preserve original training data for comparison
        
        train_df = detect_and_handle_outliers_iqr(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=3.0,              # Increased factor to reduce sensitivity
            method='clip'            # Changed method from 'remove' to 'clip'
        )
        print(f"Training data shape after handling outliers: {train_df.shape}")
        
        # Optional: Log the number of values capped per column
        for col in numerical_cols:
            # Compare before and after to determine if capping occurred
            original_max = train_df_original[col].max()
            original_min = train_df_original[col].min()
            capped_max = train_df[col].max()
            capped_min = train_df[col].min()
            
            if capped_max < original_max or capped_min > original_min:
                print(f"Outliers in '{col}' capped to [{capped_min}, {capped_max}].")
    except Exception as e:
        print(f"Error handling outliers in training data: {e}")
        raise
    
    # Handle outliers in testing data by clipping them
    try:
        test_df = detect_and_handle_outliers_iqr(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=1.5,
            method='clip'
        )
        print(f"Testing data shape after clipping outliers: {test_df.shape}")
    except Exception as e:
        print(f"Error handling outliers in testing data: {e}")
        raise
    
    # Save the cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 1: Outliers handled and cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving cleaned datasets: {e}")
        raise
    
    
    # Reload the cleaned datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Cleaned datasets reloaded successfully.")
    except Exception as e:
        print(f"Error reloading cleaned datasets: {e}")
        raise
    
    # TASK 2: Verify and Ensure Consistency in Categorical Features Across Datasets
    
    # Identify unique categories in both datasets
    train_colors = set(train_df['color'].dropna().unique())
    test_colors = set(test_df['color'].dropna().unique())
    
    print(f"Unique colors in training data before standardization: {train_colors}")
    print(f"Unique colors in testing data before standardization: {test_colors}")
    
    # Step 1: Standardize categories by converting to lowercase
    train_df['color'] = train_df['color'].str.lower()
    test_df['color'] = test_df['color'].str.lower()
    
    # Step 2: Re-identify unique categories after standardization
    train_colors_std = set(train_df['color'].dropna().unique())
    test_colors_std = set(test_df['color'].dropna().unique())
    
    print(f"Unique colors in training data after standardization: {train_colors_std}")
    print(f"Unique colors in testing data after standardization: {test_colors_std}")
    
    # Step 3: Verify consistency
    if not train_colors_std.issubset(test_colors_std) or not test_colors_std.issubset(train_colors_std):
        # Find discrepancies
        all_colors = train_colors_std.union(test_colors_std)
        print("Discrepancies found in 'color' categories. Handling inconsistencies...")
    
        # For this example, we'll map any unseen categories in the test set to 'unknown'
        # Identify categories in test not present in train
        unseen_in_test = test_colors_std - train_colors_std
        if unseen_in_test:
            test_df['color'] = test_df['color'].replace(list(unseen_in_test), 'unknown')
            print(f"Replaced unseen colors in test data with 'unknown': {unseen_in_test}")
    
        # Similarly, handle any categories in train not present in test, if necessary
        unseen_in_train = train_colors_std - test_colors_std
        if unseen_in_train:
            train_df['color'] = train_df['color'].replace(list(unseen_in_train), 'unknown')
            print(f"Replaced unseen colors in training data with 'unknown': {unseen_in_train}")
    else:
        print("No discrepancies found in 'color' categories. No additional handling needed.")
    
    # Step 4: Re-validate unique categories after handling discrepancies
    train_colors_final = set(train_df['color'].dropna().unique())
    test_colors_final = set(test_df['color'].dropna().unique())
    
    print(f"Final unique colors in training data: {train_colors_final}")
    print(f"Final unique colors in testing data: {test_colors_final}")
    
    # Save the standardized datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 2: Categorical feature 'color' standardized and cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving standardized datasets: {e}")
        raise
    
    
    # Reload the cleaned and standardized datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Cleaned and standardized datasets reloaded successfully.")
    except Exception as e:
        print(f"Error reloading cleaned datasets: {e}")
        raise
    
    # TASK 3: Validate and Convert Data Types for All Features
    
    # Define data type conversions for training data
    type_conversions_train = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str',
        'type': 'str'
    }
    
    # Define data type conversions for testing data
    type_conversions_test = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str'
    }
    
    # Convert data types for training data
    try:
        # Convert categorical columns first to avoid issues during numeric conversions
        train_df = convert_data_types(
            data=train_df,
            columns=['color', 'type'],
            target_type='str'
        )
        
        # Convert numerical columns
        train_df = convert_data_types(
            data=train_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        train_df['id'] = pd.to_numeric(train_df['id'], errors='coerce').astype('Int64')
        
        print("Training data types after conversion:")
        print(train_df.dtypes)
    except Exception as e:
        print(f"Error converting data types for training data: {e}")
        raise
    
    # Convert data types for testing data
    try:
        # Convert categorical columns first
        test_df = convert_data_types(
            data=test_df,
            columns=['color'],
            target_type='str'
        )
        
        # Convert numerical columns
        test_df = convert_data_types(
            data=test_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        test_df['id'] = pd.to_numeric(test_df['id'], errors='coerce').astype('Int64')
        
        print("Testing data types after conversion:")
        print(test_df.dtypes)
    except Exception as e:
        print(f"Error converting data types for testing data: {e}")
        raise
    
    # Save the datasets with updated data types
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 3: Data types validated and converted. Cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving datasets after type conversion: {e}")
        raise
    
    
    # Reload the datasets with updated data types
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        print("Datasets with updated data types reloaded successfully.")
    except Exception as e:
        print(f"Error reloading datasets for duplicate removal: {e}")
        raise
    
    # TASK 4: Confirm Absence of Duplicate Records
    
    # Remove duplicates in training data based on 'id'
    try:
        initial_train_shape = train_df.shape
        train_df = remove_duplicates(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_train_shape = train_df.shape
        print(f"Training data shape before removing duplicates: {initial_train_shape}")
        print(f"Training data shape after removing duplicates: {final_train_shape}")
    except Exception as e:
        print(f"Error removing duplicates from training data: {e}")
        raise
    
    # Remove duplicates in testing data based on 'id'
    try:
        initial_test_shape = test_df.shape
        test_df = remove_duplicates(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_test_shape = test_df.shape
        print(f"Testing data shape before removing duplicates: {initial_test_shape}")
        print(f"Testing data shape after removing duplicates: {final_test_shape}")
    except Exception as e:
        print(f"Error removing duplicates from testing data: {e}")
        raise
    
    # Verify absence of duplicates
    train_duplicates = train_df.duplicated(subset=['id']).sum()
    test_duplicates = test_df.duplicated(subset=['id']).sum()
    
    print(f"Number of duplicate 'id's in training data after removal: {train_duplicates}")
    print(f"Number of duplicate 'id's in testing data after removal: {test_duplicates}")
    
    # Save the final cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        print("TASK 4: Duplicate records removed. Cleaned datasets saved.")
    except Exception as e:
        print(f"Error saving datasets after duplicate removal: {e}")
        raise
    


    
    import pandas as pd
    import os
    import numpy as np
    import logging
    
    # Configure logging
    logging.basicConfig(
        filename='feature_engineering.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    def load_data(data_dir, train_file, test_file):
        try:
            train_df = pd.read_csv(os.path.join(data_dir, train_file))
            test_df = pd.read_csv(os.path.join(data_dir, test_file))
            logging.info("Cleaned datasets loaded successfully.")
            return train_df.copy(), test_df.copy()
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV files: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading data: {e}")
            raise
    
    def create_derived_features(df):
        try:
            df['hair_length_has_soul'] = df['hair_length'] * df['has_soul']
            df['bone_length_rotting_flesh'] = df['bone_length'] * df['rotting_flesh']
            df['bone_length_squared'] = df['bone_length'] ** 2
            df['rotting_flesh_squared'] = df['rotting_flesh'] ** 2
            epsilon = 1e-5  # To prevent division by zero
            df['bone_to_flesh_ratio'] = df['bone_length'] / (df['rotting_flesh'] + epsilon)
            df['soul_to_hair_ratio'] = df['has_soul'] / (df['hair_length'] + epsilon)
            logging.info("Derived features created successfully.")
            return df
        except KeyError as e:
            logging.error(f"Missing column during feature creation: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during feature creation: {e}")
            raise
    
    def handle_infinite_nan(df):
        try:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            logging.info("Infinite and NaN values handled successfully.")
            return df
        except Exception as e:
            logging.error(f"Error handling infinite or NaN values: {e}")
            raise
    
    def main():
        # Define file paths
        data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        cleaned_train_file = 'cleaned_train.csv'
        cleaned_test_file = 'cleaned_test.csv'
        
        # Load data
        train_df, test_df = load_data(data_dir, cleaned_train_file, cleaned_test_file)
        
        # Create derived features
        train_df = create_derived_features(train_df)
        test_df = create_derived_features(test_df)
        
        # Handle infinite and NaN values
        train_df = handle_infinite_nan(train_df)
        test_df = handle_infinite_nan(test_df)
        
        # Save the intermediate datasets
        processed_train_file = 'processed_train.csv'
        processed_test_file = 'processed_test.csv'
        
        try:
            train_df.to_csv(os.path.join(data_dir, processed_train_file), index=False)
            test_df.to_csv(os.path.join(data_dir, processed_test_file), index=False)
            logging.info("Derived features added and intermediate datasets saved successfully.")
        except Exception as e:
            logging.error(f"Error saving processed datasets: {e}")
            raise
    
    if __name__ == "__main__":
        main()
    
    
    import pandas as pd
    import os
    import logging
    from sklearn.preprocessing import OneHotEncoder
    
    def load_processed_data(data_dir, processed_train_file, processed_test_file):
        try:
            train_df = pd.read_csv(os.path.join(data_dir, processed_train_file))
            test_df = pd.read_csv(os.path.join(data_dir, processed_test_file))
            logging.info("Processed datasets loaded successfully for encoding.")
            return train_df.copy(), test_df.copy()
        except FileNotFoundError as e:
            logging.error(f"Processed file not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing processed CSV files: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading processed data: {e}")
            raise
    
    def encode_categorical(train_df, test_df, categorical_cols):
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            ohe.fit(train_df[categorical_cols])
            
            # Transform training data
            train_ohe = pd.DataFrame(
                ohe.transform(train_df[categorical_cols]),
                columns=ohe.get_feature_names_out(categorical_cols),
                index=train_df.index
            )
            
            # Transform testing data
            test_ohe = pd.DataFrame(
                ohe.transform(test_df[categorical_cols]),
                columns=ohe.get_feature_names_out(categorical_cols),
                index=test_df.index
            )
            
            # Drop original categorical columns
            train_df.drop(columns=categorical_cols, inplace=True)
            test_df.drop(columns=categorical_cols, inplace=True)
            
            # Concatenate One-Hot Encoded columns
            train_df = pd.concat([train_df, train_ohe], axis=1)
            test_df = pd.concat([test_df, test_ohe], axis=1)
            
            logging.info("Categorical variables encoded successfully using One-Hot Encoding.")
            return train_df, test_df
        except KeyError as e:
            logging.error(f"Categorical column missing during encoding: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during categorical encoding: {e}")
            raise
    
    def main():
        # Define file paths
        data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        processed_train_file = 'processed_train.csv'
        processed_test_file = 'processed_test.csv'
        
        # Load processed data
        train_df, test_df = load_processed_data(data_dir, processed_train_file, processed_test_file)
        
        # Define categorical columns
        categorical_cols = ['color']
        
        # Encode categorical variables
        train_df, test_df = encode_categorical(train_df, test_df, categorical_cols)
        
        # Save the datasets after encoding
        encoded_train_file = 'encoded_train.csv'
        encoded_test_file = 'encoded_test.csv'
        
        try:
            train_df.to_csv(os.path.join(data_dir, encoded_train_file), index=False)
            test_df.to_csv(os.path.join(data_dir, encoded_test_file), index=False)
            logging.info("Categorical variables encoded and datasets saved successfully.")
        except Exception as e:
            logging.error(f"Error saving encoded datasets: {e}")
            raise
    
    if __name__ == "__main__":
        main()
    
    
    import pandas as pd
    import os
    import logging
    from sklearn.preprocessing import StandardScaler
    
    def load_encoded_data(data_dir, encoded_train_file, encoded_test_file):
        try:
            train_df = pd.read_csv(os.path.join(data_dir, encoded_train_file))
            test_df = pd.read_csv(os.path.join(data_dir, encoded_test_file))
            logging.info("Encoded datasets loaded successfully for scaling.")
            return train_df.copy(), test_df.copy()
        except FileNotFoundError as e:
            logging.error(f"Encoded file not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing encoded CSV files: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading encoded data: {e}")
            raise
    
    def scale_features(train_df, test_df, numerical_cols):
        try:
            scaler = StandardScaler()
            scaler.fit(train_df[numerical_cols])
            
            # Transform training data
            train_scaled = scaler.transform(train_df[numerical_cols])
            train_scaled_df = pd.DataFrame(train_scaled, columns=numerical_cols, index=train_df.index)
            train_df[numerical_cols] = train_scaled_df
            
            # Transform testing data
            test_scaled = scaler.transform(test_df[numerical_cols])
            test_scaled_df = pd.DataFrame(test_scaled, columns=numerical_cols, index=test_df.index)
            test_df[numerical_cols] = test_scaled_df
            
            logging.info("Numerical features scaled successfully using StandardScaler.")
            return train_df, test_df
        except KeyError as e:
            logging.error(f"Numerical column missing during scaling: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during feature scaling: {e}")
            raise
    
    def main():
        # Define file paths
        data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        encoded_train_file = 'encoded_train.csv'
        encoded_test_file = 'encoded_test.csv'
        
        # Load encoded data
        train_df, test_df = load_encoded_data(data_dir, encoded_train_file, encoded_test_file)
        
        # Define numerical columns
        # Exclude 'id' and 'type' from scaling
        numerical_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul',
                          'hair_length_has_soul', 'bone_length_rotting_flesh',
                          'bone_length_squared', 'rotting_flesh_squared',
                          'bone_to_flesh_ratio', 'soul_to_hair_ratio']
        
        # Scale numerical features
        train_df, test_df = scale_features(train_df, test_df, numerical_cols)
        
        # Save the scaled datasets
        scaled_train_file = 'scaled_train.csv'
        scaled_test_file = 'scaled_test.csv'
        
        try:
            train_df.to_csv(os.path.join(data_dir, scaled_train_file), index=False)
            test_df.to_csv(os.path.join(data_dir, scaled_test_file), index=False)
            logging.info("Numerical features scaled and datasets saved successfully.")
        except Exception as e:
            logging.error(f"Error saving scaled datasets: {e}")
            raise
    
    if __name__ == "__main__":
        main()
    
    
    import pandas as pd
    import os
    import logging
    from sklearn.ensemble import RandomForestClassifier
    
    def load_scaled_data(data_dir, scaled_train_file, scaled_test_file):
        try:
            train_df = pd.read_csv(os.path.join(data_dir, scaled_train_file))
            test_df = pd.read_csv(os.path.join(data_dir, scaled_test_file))
            logging.info("Scaled datasets loaded successfully for feature selection.")
            return train_df.copy(), test_df.copy()
        except FileNotFoundError as e:
            logging.error(f"Scaled file not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing scaled CSV files: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading scaled data: {e}")
            raise
    
    def select_top_features(train_df, target_column, top_n=10):
        try:
            X = train_df.drop(columns=['id', target_column])
            y = train_df[target_column]
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = feature_importances.sort_values(ascending=False).head(top_n).index.tolist()
            
            logging.info(f"Top {top_n} features selected based on feature importances.")
            logging.debug(f"Feature importances: {feature_importances}")
            return top_features
        except KeyError as e:
            logging.error(f"Target or feature column missing during feature selection: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during feature selection: {e}")
            raise
    
    def filter_features(df, top_features, include_target=True):
        try:
            columns = ['id'] + top_features
            if include_target:
                columns.append('type')
            return df[columns]
        except KeyError as e:
            logging.error(f"One or more top features missing in the dataset: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during feature filtering: {e}")
            raise
    
    def main():
        # Define file paths
        data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        scaled_train_file = 'scaled_train.csv'
        scaled_test_file = 'scaled_test.csv'
        
        # Load scaled data
        train_df, test_df = load_scaled_data(data_dir, scaled_train_file, scaled_test_file)
        
        # Define target column
        target_column = 'type'
        
        # Select top features
        top_features = select_top_features(train_df, target_column, top_n=10)
        
        # Filter training data to include only top features
        train_selected = filter_features(train_df, top_features, include_target=True)
        
        # Filter testing data to include only top features (exclude target)
        test_selected = filter_features(test_df, top_features, include_target=False)
        
        # Save the final processed datasets
        final_processed_train = 'processed_train.csv'
        final_processed_test = 'processed_test.csv'
        
        try:
            train_selected.to_csv(os.path.join(data_dir, final_processed_train), index=False)
            test_selected.to_csv(os.path.join(data_dir, final_processed_test), index=False)
            logging.info("Feature selection completed and final processed datasets saved successfully.")
        except Exception as e:
            logging.error(f"Error saving final processed datasets: {e}")
            raise
    
    if __name__ == "__main__":
        main()
    


    
    import pandas as pd
    import os
    import numpy as np
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    # Assuming train_and_validation_and_select_the_best_model is pre-imported and available
    
    def load_and_prepare_data(data_dir, processed_train_file, processed_test_file):
        """
        Loads the processed training and testing data, separates the target variable,
        and prepares the feature sets for modeling.
        
        Parameters:
        - data_dir (str): Directory where the data files are located.
        - processed_train_file (str): Filename for the processed training data.
        - processed_test_file (str): Filename for the processed testing data.
        
        Returns:
        - X_train (pd.DataFrame): Feature set for training.
        - y_train (pd.Series): Target variable for training.
        - X_test (pd.DataFrame): Feature set for testing.
        - test_ids (pd.Series): IDs from the testing set for submission.
        """
        # Load the processed training and testing data
        train_path = os.path.join(data_dir, processed_train_file)
        test_path = os.path.join(data_dir, processed_test_file)
        
        X_train_full = pd.read_csv(train_path).copy()
        X_test_full = pd.read_csv(test_path).copy()
        
        # Separate the target variable
        y_train = X_train_full.pop('type')
        
        # Preserve the 'id' for submission and drop it from features
        test_ids = X_test_full['id']
        
        # Remove 'id' from training features
        X_train = X_train_full.drop('id', axis=1)
        
        # Remove 'id' from testing features
        X_test = X_test_full.drop('id', axis=1)
        
        # Ensure that the feature columns in training and testing are identical
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Mismatch in feature columns between training and testing sets.")
        
        # Output the shapes of the datasets
        print(f'X_train shape: {X_train.shape}')
        print(f'y_train shape: {y_train.shape}')
        print(f'X_test shape: {X_test.shape}')
        
        return X_train, y_train, X_test, test_ids
    
    def train_and_validate_models(X_train, y_train, selected_models, problem_type="multiclass"):
        """
        Trains and validates models using the provided tool and evaluates their performance.
        
        Parameters:
        - X_train (pd.DataFrame): Feature set for training.
        - y_train (pd.Series): Target variable for training.
        - selected_models (list): List of model names to train.
        - problem_type (str): Type of problem ('binary', 'multiclass', 'regression').
        
        Returns:
        - best_model: The best performing trained model.
        """
        # Train models and select the best one
        best_model = train_and_validation_and_select_the_best_model(
            X=X_train,
            y=y_train,
            problem_type=problem_type,
            selected_models=selected_models
        )
        
        print(f"Best Model Selected: {best_model.__class__.__name__}")
        
        return best_model
    
    def evaluate_model(model, X_train, y_train, cv_folds=5):
        """
        Evaluates the trained model using cross-validation and reports accuracy metrics.
        
        Parameters:
        - model: Trained machine learning model.
        - X_train (pd.DataFrame): Feature set for training.
        - y_train (pd.Series): Target variable for training.
        - cv_folds (int): Number of cross-validation folds.
        
        Returns:
        - cv_mean (float): Mean cross-validation accuracy.
        - cv_std (float): Standard deviation of cross-validation accuracy.
        """
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
        
        # Compute mean and standard deviation
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f'Cross-Validation Accuracy Scores: {cv_scores}')
        print(f'Mean CV Accuracy: {cv_mean:.4f}')
        print(f'Standard Deviation of CV Accuracy: {cv_std:.4f}')
        
        return cv_mean, cv_std
    
    def make_predictions(model, X_test):
        """
        Generates predictions using the trained model on the test dataset.
        
        Parameters:
        - model: Trained machine learning model.
        - X_test (pd.DataFrame): Feature set for testing.
        
        Returns:
        - predictions (np.ndarray): Predicted class labels.
        """
        predictions = model.predict(X_test)
        return predictions
    
    def create_submission(test_ids, predictions, submission_file_path):
        """
        Creates and saves the submission CSV file.
        
        Parameters:
        - test_ids (pd.Series or list): IDs corresponding to the test data.
        - predictions (np.ndarray or list): Predicted class labels.
        - submission_file_path (str): Path to save the submission CSV file.
        
        Returns:
        - None
        """
        # Create the submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_ids,
            'type': predictions
        })
        
        # Sanity check for missing values
        if submission_df['type'].isnull().sum() > 0:
            raise ValueError("There are missing values in the predictions.")
        
        # Display a sample of the submission
        print("Sample of Submission File:")
        print(submission_df.head())
        
        # Ensure the parent directory exists
        submission_dir = os.path.dirname(submission_file_path)
        if submission_dir:  # Check if there is a directory path
            os.makedirs(submission_dir, exist_ok=True)
        
        # Save the submission DataFrame to CSV
        submission_df.to_csv(submission_file_path, index=False)
        print(f"\nSubmission file saved successfully at {submission_file_path}")
    
    def main():
        # Define the data directory and filenames
        data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        processed_train_file = 'processed_train.csv'
        processed_test_file = 'processed_test.csv'
        
        # Load and prepare the data
        X_train, y_train, X_test, test_ids = load_and_prepare_data(
            data_dir, 
            processed_train_file, 
            processed_test_file
        )
        
        # Define the selected models
        selected_models = ["XGBoost", "SVM", "random forest"]
        
        # Train and validate models
        best_model = train_and_validate_models(
            X_train, 
            y_train, 
            selected_models, 
            problem_type="multiclass"
        )
        
        # Evaluate the best model
        cv_mean, cv_std = evaluate_model(best_model, X_train, y_train, cv_folds=5)
        
        # Make predictions on the test set
        predictions = make_predictions(best_model, X_test)
        
        # Define the submission file path
        # Define the correct submission file path
        submission_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
        submission_file_name = 'submission.csv'
        submission_file_path = os.path.join(submission_dir, submission_file_name)
        
        # Create and save the submission file
        create_submission(test_ids, predictions, submission_file_path)
    
    if __name__ == "__main__":
        main()
    
    
    # Example usage of create_submission
    test_ids = [3, 6, 9, 10, 13]
    predictions = ['Ghoul', 'Goblin', 'Ghoul', 'Ghost', 'Ghost']
    submission_directory = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/model_build_predict/images'
    submission_file_name = 'submission_example.csv'
    submission_file_path = os.path.join(submission_directory, submission_file_name)
    
    create_submission(test_ids, predictions, submission_file_path)
    


if __name__ == "__main__":
    generated_code_function()
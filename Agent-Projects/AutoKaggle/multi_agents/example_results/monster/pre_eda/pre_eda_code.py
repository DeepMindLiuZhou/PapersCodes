import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Define the data directory
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
    
    # Load the datasets
    train_path = f'{data_dir}train.csv'
    test_path = f'{data_dir}test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Inspect dataset shapes
    train_shape = train_df.shape
    test_shape = test_df.shape
    
    print(f"Training Data Shape: {train_shape}")
    print(f"Testing Data Shape: {test_shape}\n")
    
    # Inspect data types
    print("Training Data Types:")
    print(train_df.dtypes)
    print("\nTesting Data Types:")
    print(test_df.dtypes)
    print("\n")
    
    # Count unique values in categorical features
    categorical_features = ['color', 'type']
    print("Unique Values in Categorical Features (Training Data):")
    for feature in categorical_features:
        unique_count = train_df[feature].nunique()
        unique_values = train_df[feature].unique()
        print(f"- {feature}: {unique_count} unique values -> {unique_values}")
    
    # Since 'type' is not in test data, handle separately
    print("\nUnique Values in Categorical Features (Testing Data):")
    for feature in ['color']:
        unique_count = test_df[feature].nunique()
        unique_values = test_df[feature].unique()
        print(f"- {feature}: {unique_count} unique values -> {unique_values}")
    
    
    # Numerical and categorical features
    numerical_features = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
    categorical_features = ['color', 'type']  # Note: 'type' exists only in training data
    
    # Descriptive statistics for numerical features
    print("Descriptive Statistics for Numerical Features (Training Data):")
    numerical_stats = train_df[numerical_features].describe()
    print(numerical_stats)
    print("\n")
    
    # Frequency counts for categorical features
    print("Frequency Counts for Categorical Features (Training Data):")
    for feature in ['color', 'type']:
        counts = train_df[feature].value_counts()
        print(f"\n- {feature} Value Counts:")
        print(counts)
    
    # Frequency counts for categorical features in testing data
    print("\nFrequency Counts for Categorical Features (Testing Data):")
    color_counts_test = test_df['color'].value_counts()
    print(f"\n- color Value Counts:")
    print(color_counts_test)
    
    
    # Function to calculate outliers using IQR
    def detect_outliers_iqr(df, feature):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        return outliers.shape[0]
    
    # Missing values assessment
    print("Missing Values in Training Data:")
    missing_train = train_df.isnull().sum()
    print(missing_train)
    print("\nMissing Values in Testing Data:")
    missing_test = test_df.isnull().sum()
    print(missing_test)
    print("\n")
    
    # Duplicate rows assessment
    duplicate_train = train_df.duplicated().sum()
    duplicate_test = test_df.duplicated().sum()
    print(f"Number of Duplicate Rows in Training Data: {duplicate_train}")
    print(f"Number of Duplicate Rows in Testing Data: {duplicate_test}\n")
    
    # Uniqueness in categorical features
    print("Categorical Feature Consistency (Training Data):")
    for feature in ['color', 'type']:
        unique_values = train_df[feature].unique()
        print(f"- {feature}: {len(unique_values)} unique values -> {unique_values}")
    
    # Outlier detection in numerical features
    print("\nOutlier Detection in Numerical Features (Training Data):")
    for feature in numerical_features:
        outlier_count = detect_outliers_iqr(train_df, feature)
        print(f"- {feature}: {outlier_count} outliers detected")
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Descriptive statistics for numerical features
    print("Detailed Descriptive Statistics for Numerical Features (Training Data):")
    detailed_stats = train_df[numerical_features].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    print(detailed_stats)
    print("\n")
    
    # Correlation matrix
    print("Correlation Matrix for Numerical Features (Training Data):")
    corr_matrix = train_df[numerical_features].corr()
    print(corr_matrix)
    print("\n")
    
    # Save correlation matrix as heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    
    # Define image save path
    image_dir = f'{data_dir}pre_eda/images/'
    correlation_heatmap_path = f'{image_dir}correlation_matrix_heatmap.png'
    
    # Save the heatmap
    plt.savefig(correlation_heatmap_path)
    plt.close()
    print(f"Correlation matrix heatmap saved to {correlation_heatmap_path}\n")
    
    # Analyze relationship between 'color' and numerical features
    print("Relationship Between 'color' and Numerical Features (Training Data):")
    color_groups = train_df.groupby('color')[numerical_features].mean()
    print(color_groups)
    print("\n")
    


if __name__ == "__main__":
    generated_code_function()
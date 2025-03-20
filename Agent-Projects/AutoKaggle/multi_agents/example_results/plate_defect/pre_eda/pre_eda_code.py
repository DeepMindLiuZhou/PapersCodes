import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load train and test datasets
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Display the first few rows of the train and test datasets
    print("Train Data Sample:")
    print(train_df.head())
    
    print("\nTest Data Sample:")
    print(test_df.head())
    
    # Summary statistics for numerical features
    print("\nSummary Statistics for Train Data:")
    print(train_df.describe())
    
    print("\nSummary Statistics for Test Data:")
    print(test_df.describe())
    
    # Data types of each column
    print("\nData Types in Train Data:")
    print(train_df.dtypes)
    
    print("\nData Types in Test Data:")
    print(test_df.dtypes)
    
    # Value counts for categorical features
    categorical_features = ['TypeOfSteel_A300', 'TypeOfSteel_A400']
    
    for feature in categorical_features:
        print(f"\nValue Counts for {feature} in Train Data:")
        print(train_df[feature].value_counts())
    
        print(f"\nValue Counts for {feature} in Test Data:")
        print(test_df[feature].value_counts())
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Histograms for numerical features
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
    
    # Removing the target variables from numerical features list for histograms and boxplots
    target_variables = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    numerical_features = [feature for feature in numerical_features if feature not in target_variables]
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.histplot(train_df[feature], bins=30, kde=False)
        plt.title(f"Histogram of {feature}")
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/pre_eda/images/histogram_{feature}.png')
        plt.close()
    
    # Boxplots for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=train_df[feature], orient='h')
        plt.title(f"Boxplot of {feature}")
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/pre_eda/images/boxplot_{feature}.png')
        plt.close()
    
    # Bar plots for categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=train_df[feature], palette='viridis')
        plt.title(f"Bar Plot of {feature}")
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/pre_eda/images/barplot_{feature}.png')
        plt.close()
    
    
    import numpy as np
    
    # Check for missing values in train and test datasets
    print("Missing Values in Train Data:")
    print(train_df.isnull().sum())
    
    print("\nMissing Values in Test Data:")
    print(test_df.isnull().sum())
    
    # Unique values check for categorical features
    for feature in categorical_features:
        print(f"\nUnique Values for {feature} in Train Data:")
        print(train_df[feature].unique())
    
        print(f"\nUnique Values for {feature} in Test Data:")
        print(test_df[feature].unique())
    
    # Outlier detection using IQR method
    for feature in numerical_features:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = train_df[(train_df[feature] < (Q1 - 1.5 * IQR)) | (train_df[feature] > (Q3 + 1.5 * IQR))]
        print(f"\nOutliers in {feature}: {len(outliers)}")
    
    
    # Correlation matrix
    correlation_matrix = train_df.corr()
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix Heatmap")
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/pre_eda/images/correlation_matrix.png')
    plt.close()
    
    # Pairplot for a subset of numerical features
    subset_features = numerical_features[:5]  # Limiting to first 5 numerical features for efficiency
    sns.pairplot(train_df[subset_features])
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/pre_eda/images/pairplot_subset_features.png')
    plt.close()
    


if __name__ == "__main__":
    generated_code_function()
import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Define file paths
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/test.csv'
    
    # Load the datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Display the first few rows of the train and test datasets
    print("Train DataFrame Head:")
    print(train_df.head())
    
    print("\nTest DataFrame Head:")
    print(test_df.head())
    
    # Summary information including data types and non-null counts
    print("\nTrain DataFrame Info:")
    print(train_df.info())
    
    print("\nTest DataFrame Info:")
    print(test_df.info())
    
    
    # Basic statistical analysis for numerical features
    print("\nTrain DataFrame Numerical Features Description:")
    print(train_df.describe())
    
    # Frequency distribution for categorical features
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    print("\nTrain DataFrame Categorical Features Value Counts:")
    for feature in categorical_features:
        print(f"\nValue counts for {feature}:")
        print(train_df[feature].value_counts())
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Identify missing values
    print("\nMissing Values in Train DataFrame:")
    print(train_df.isnull().sum())
    
    print("\nMissing Values in Test DataFrame:")
    print(test_df.isnull().sum())
    
    # Boxplots for numerical features to identify potential outliers
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    plt.figure(figsize=(15, 10))
    train_df[numerical_features].boxplot()
    plt.title("Boxplot for Numerical Features in Train DataFrame")
    plt.xticks(rotation=45)
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/pre_eda/images/numerical_features_boxplot.png')
    plt.close()
    
    
    # Histograms for numerical features
    for feature in numerical_features:
        plt.figure()
        train_df[feature].hist(bins=30)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/pre_eda/images/histogram_{feature}.png')
        plt.close()
    
    # Bar charts for categorical features
    for feature in categorical_features:
        plt.figure()
        train_df[feature].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/pre_eda/images/bar_chart_{feature}.png')
        plt.close()
    
    # Pairplot for key numerical features
    sns.pairplot(train_df[numerical_features])
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/pre_eda/images/pairplot_numerical_features.png')
    plt.close()
    
    # Heatmap showing correlations between numerical features
    plt.figure(figsize=(10, 8))
    correlation_matrix = train_df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/pre_eda/images/correlation_heatmap.png')
    plt.close()
    


if __name__ == "__main__":
    generated_code_function()
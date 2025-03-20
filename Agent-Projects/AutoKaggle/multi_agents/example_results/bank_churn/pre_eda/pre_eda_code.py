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
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load the datasets
    train_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/train.csv')
    test_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/test.csv')
    
    # Display basic information about the datasets
    print("Training Data Info:")
    print(train_data.info())
    print("\nTest Data Info:")
    print(test_data.info())
    
    # Display the first few rows of the datasets
    print("\nTraining Data Head:")
    print(train_data.head())
    print("\nTest Data Head:")
    print(test_data.head())
    
    # Display basic statistics
    print("\nTraining Data Description:")
    print(train_data.describe())
    print("\nTest Data Description:")
    print(test_data.describe())
    
    # Set the style for seaborn
    sns.set(style="whitegrid")
    
    # Histograms for numerical features in training data
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/pre_eda/images/distribution_{feature}.png')
        plt.close()
    
    # Bar plots for categorical features in training data
    categorical_features = ['Geography', 'Gender']
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=train_data)
        plt.title(f'Count of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/pre_eda/images/count_{feature}.png')
        plt.close()
    
    # Box plots for numerical features in training data to identify outliers
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=train_data[feature])
        plt.title(f'Box plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/pre_eda/images/boxplot_{feature}.png')
        plt.close()
    
    # Check for missing values in training and test datasets
    print("\nMissing Values in Training Data:")
    print(train_data.isnull().sum())
    print("\nMissing Values in Test Data:")
    print(test_data.isnull().sum())
    
    # Check for negative values in numerical features
    for feature in numerical_features:
        if (train_data[feature] < 0).any() or (test_data[feature] < 0).any():
            print(f"\nInconsistency Found: Negative values in {feature}")
    
    # Filter only numerical columns from the training data
    numerical_train_data = train_data.select_dtypes(include=[np.number])
    
    # Correlation matrix for numerical features in training data
    correlation_matrix = numerical_train_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/pre_eda/images/correlation_matrix.png')
    plt.close()
    
    # Pairplot for numerical features and target variable
    sns.pairplot(train_data, vars=numerical_features, hue='Exited')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/pre_eda/images/pairplot.png')
    plt.close()
    


if __name__ == "__main__":
    generated_code_function()
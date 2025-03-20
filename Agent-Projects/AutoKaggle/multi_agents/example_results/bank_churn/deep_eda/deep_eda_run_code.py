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
    pass
    
    
    # Detect and handle outliers in train and test data using the provided tool
    train_data = detect_and_handle_outliers_iqr(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    test_data = detect_and_handle_outliers_iqr(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], factor=1.5, method='clip')
    
    # Print a summary to verify outliers handling
    pass
    
    
    # Ensure consistency in data types for train and test data using the provided tool
    train_data = convert_data_types(data=train_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    train_data = convert_data_types(data=train_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    test_data = convert_data_types(data=test_data, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], target_type='float')
    test_data = convert_data_types(data=test_data, columns=['HasCrCard', 'IsActiveMember'], target_type='int')
    
    # Print data types to verify consistency
    pass
    
    
    # Align categorical values for train and test data
    train_data['Geography'] = pd.Categorical(train_data['Geography'])
    train_data['Gender'] = pd.Categorical(train_data['Gender'])
    
    test_data['Geography'] = pd.Categorical(test_data['Geography'])
    test_data['Gender'] = pd.Categorical(test_data['Gender'])
    
    # Print unique values to verify alignment
    pass
    
    
    # Save cleaned data
    train_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv', index=False)
    test_data.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load cleaned data
    train_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_train.csv')
    test_data = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/cleaned_test.csv')
    
    # Descriptive statistics for numerical features
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    print("Descriptive Statistics for Numerical Features:\n", train_data[numerical_features].describe())
    
    # Frequency counts for categorical features
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    for feature in categorical_features:
        print(f"Value Counts for {feature}:\n", train_data[feature].value_counts())
    
    # Distribution plots for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/distribution_{feature}.png')
        plt.close()
    
    # Box plots for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_data[feature])
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/boxplot_{feature}.png')
        plt.close()
    
    
    # Proportion of exited vs non-exited customers
    print("Proportion of Exited vs Non-Exited Customers:\n", train_data['Exited'].value_counts(normalize=True))
    
    # Count plot for target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_data, x='Exited')
    plt.title('Count of Exited vs Non-Exited Customers')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/countplot_exited.png')
    plt.close()
    
    
    # Compute correlation matrix
    correlation_matrix = train_data[numerical_features + ['Exited']].corr()
    print("Correlation Matrix:\n", correlation_matrix)
    
    # Heatmap for correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/heatmap_correlation_matrix.png')
    plt.close()
    
    
    # Count plots for categorical features vs target
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train_data, x=feature, hue='Exited')
        plt.title(f'{feature} vs Exited')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/countplot_{feature}_exited.png')
        plt.close()
    
    
    # Box plots for numerical features vs target
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=train_data, x='Exited', y=feature)
        plt.title(f'{feature} vs Exited (Box Plot)')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/boxplot_{feature}_exited.png')
        plt.close()
    
    # Violin plots for numerical features vs target
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=train_data, x='Exited', y=feature)
        plt.title(f'{feature} vs Exited (Violin Plot)')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/violinplot_{feature}_exited.png')
        plt.close()
    
    
    # Pair plots for key features
    key_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Exited']
    sns.pairplot(train_data[key_features], hue='Exited', diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/pairplot_key_features.png')
    plt.close()
    
    
    # Joint plots for feature interactions with target
    interaction_pairs = [('Geography', 'Balance'), ('Age', 'CreditScore')]
    for x_feature, y_feature in interaction_pairs:
        # Check if both x_feature and y_feature are numeric
        if pd.api.types.is_numeric_dtype(train_data[x_feature]) and pd.api.types.is_numeric_dtype(train_data[y_feature]):
            plt.figure(figsize=(10, 6))
            sns.jointplot(data=train_data, x=x_feature, y=y_feature, hue='Exited')
            plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/bank_churn/deep_eda/images/jointplot_{x_feature}_{y_feature}.png')
            plt.close()
    
    
    # Summary of key insights and recommendations
    summary = """
    Key Insights:
    1. Numerical features such as Age, Balance, and EstimatedSalary show significant variation among customers.
    2. The proportion of exited customers is lower compared to non-exited customers.
    3. Correlation analysis reveals some relationships between features like Age and Balance.
    4. Categorical features like Geography and Gender have distinct distributions across the target variable.
    5. Key feature interactions show specific patterns that might be useful for feature engineering.
    
    Recommendations for Feature Engineering:
    1. Create new features based on interactions between key numerical features (e.g., Balance-to-EstimatedSalary ratio).
    2. Encode categorical features using techniques like One-Hot Encoding or Label Encoding.
    3. Consider feature scaling for numerical features to improve model performance.
    4. Explore polynomial features or interactions between numerical features to capture more complex relationships.
    5. Perform further analysis to identify potential outliers or anomalies in the data.
    
    """
    
    print(summary)
    


if __name__ == "__main__":
    generated_code_function()
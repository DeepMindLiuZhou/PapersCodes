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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Define numerical and categorical features
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Fill missing values in numerical features
    train_df = fill_missing_values(train_df, numerical_features, method='mean')
    test_df = fill_missing_values(test_df, numerical_features, method='mean')
    
    # Fill missing values in categorical features
    train_df = fill_missing_values(train_df, categorical_features, method='mode')
    test_df = fill_missing_values(test_df, categorical_features, method='mode')
    
    
    # Handle outliers in numerical features using IQR method
    train_df = detect_and_handle_outliers_iqr(train_df, numerical_features, factor=1.5, method='clip')
    test_df = detect_and_handle_outliers_iqr(test_df, numerical_features, factor=1.5, method='clip')
    
    
    # Convert categorical features to lowercase
    for feature in categorical_features:
        train_df[feature] = train_df[feature].str.lower()
        test_df[feature] = test_df[feature].str.lower()
    
    # Convert data types for numerical features and 'id' column
    train_df = convert_data_types(train_df, numerical_features, target_type='float')
    test_df = convert_data_types(test_df, numerical_features, target_type='float')
    
    train_df = convert_data_types(train_df, 'id', target_type='int')
    test_df = convert_data_types(test_df, 'id', target_type='int')
    
    
    # Remove duplicates from the dataset
    train_df = remove_duplicates(train_df, keep='first')
    test_df = remove_duplicates(test_df, keep='first')
    
    
    # Save the cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_test.csv'
    
    train_df.to_csv(cleaned_train_path, index=False)
    test_df.to_csv(cleaned_test_path, index=False)
    


    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load cleaned data
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/cleaned_test.csv'
    
    train_df = pd.read_csv(cleaned_train_path)
    test_df = pd.read_csv(cleaned_test_path)
    
    # Define numerical and categorical features
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    target_variable = 'NObeyesdad'
    
    # Univariate Analysis on Numerical Features
    for feature in numerical_features:
        print(f"Summary statistics for {feature}:")
        print(train_df[feature].describe())
        
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df[feature], kde=True)
        plt.title(f'Histogram of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/histogram_{feature}.png')
        plt.close()
        
        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[feature])
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/boxplot_{feature}.png')
        plt.close()
    
    # Univariate Analysis on Categorical Features
    for feature in categorical_features:
        print(f"Frequency counts for {feature}:")
        print(train_df[feature].value_counts())
        
        # Bar Chart
        plt.figure(figsize=(10, 6))
        sns.countplot(x=train_df[feature])
        plt.title(f'Bar Chart of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/barchart_{feature}.png')
        plt.close()
    
    # Target Variable Analysis
    print(f"Frequency counts for {target_variable}:")
    print(train_df[target_variable].value_counts())
    
    # Bar Chart for Target Variable
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_df[target_variable])
    plt.title(f'Bar Chart of {target_variable}')
    plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/barchart_{target_variable}.png')
    plt.close()
    
    
    # Bivariate Analysis: Numerical Features vs Target Variable
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[target_variable], y=train_df[feature])
        plt.title(f'Box Plot of {feature} vs {target_variable}')
        plt.xticks(rotation=45)
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/boxplot_{feature}_vs_{target_variable}.png')
        plt.close()
    
    # Bivariate Analysis: Categorical Features vs Target Variable
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=train_df[target_variable], hue=train_df[feature])
        plt.title(f'Bar Chart of {feature} vs {target_variable}')
        plt.xticks(rotation=45)
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/barchart_{feature}_vs_{target_variable}.png')
        plt.close()
    
    # Correlation Analysis
    # Encode target variable to numerical values
    train_df_encoded = train_df.copy()
    train_df_encoded[target_variable] = train_df_encoded[target_variable].astype('category').cat.codes
    
    # Calculate correlation matrix
    correlation_matrix = train_df_encoded[numerical_features + [target_variable]].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/correlation_matrix.png')
    plt.close()
    
    
    # Pairwise Feature Interaction Analysis: Numerical Features
    sns.pairplot(train_df[numerical_features + [target_variable]], hue=target_variable)
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/obesity_risks/deep_eda/images/pairplot_numerical_features.png')
    plt.close()
    
    # Chi-Square Test for Categorical Features
    from scipy.stats import chi2_contingency
    
    chi_square_results = {}
    for feature in categorical_features:
        contingency_table = pd.crosstab(train_df[feature], train_df[target_variable])
        chi_square_results[feature] = chi2_contingency(contingency_table)
    
    # Display chi-square test results
    for feature, result in chi_square_results.items():
        chi2, p, dof, ex = result
        print(f"Chi-Square Test for {feature} vs {target_variable}:")
        print(f"Chi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
        print("Expected Frequencies:")
        print(ex)
        print("\n")
    
    
    # Key Visualizations
    # (Already created in previous tasks; we will summarize the key ones)
    
    # Summary of Findings
    summary = """
    Key Insights and Patterns:
    1. Age, Height, and Weight show distinct distributions and ranges.
    2. Gender distribution is almost balanced.
    3. Family history of overweight is prevalent.
    4. High caloric food consumption is common.
    5. Most individuals consume vegetables frequently and have regular meals.
    6. Water consumption varies widely.
    7. Physical activity frequency and technology usage show diverse patterns.
    8. Alcohol consumption and transportation modes are varied.
    9. The target variable 'NObeyesdad' has a varied distribution across different levels of obesity.
    10. Correlation analysis shows certain relationships between numerical features and the target variable.
    11. Chi-square tests indicate dependencies between categorical features and the target variable.
    """
    
    print(summary)
    


if __name__ == "__main__":
    generated_code_function()
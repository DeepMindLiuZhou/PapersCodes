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
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/test.csv')
    
    # Make copies of the original DataFrames
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    # Handle missing values for 'Age'
    fill_missing_values(train_df_copy, columns=['Age'], method='mean')
    fill_missing_values(test_df_copy, columns=['Age'], method='mean')
    
    # Handle missing values for 'Cabin'
    fill_missing_values(train_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    fill_missing_values(test_df_copy, columns=['Cabin'], method='constant', fill_value='Missing')
    
    # Handle missing values for 'Embarked'
    fill_missing_values(train_df_copy, columns=['Embarked'], method='mode')
    fill_missing_values(test_df_copy, columns=['Embarked'], method='mode')
    
    # Handle missing values for 'Fare'
    fill_missing_values(train_df_copy, columns=['Fare'], method='mean')
    fill_missing_values(test_df_copy, columns=['Fare'], method='mean')
    
    print("Missing values handled successfully.")
    
    # Treat outliers for 'Fare'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Fare'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Fare'], factor=1.5, method='clip')
    
    # Treat outliers for 'Age'
    detect_and_handle_outliers_iqr(train_df_copy, columns=['Age'], factor=1.5, method='clip')
    detect_and_handle_outliers_iqr(test_df_copy, columns=['Age'], factor=1.5, method='clip')
    
    print("Outliers treated successfully.")
    
    # Convert data types
    convert_data_types(train_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(train_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    convert_data_types(test_df_copy, columns=['Pclass', 'SibSp', 'Parch'], target_type='int')
    convert_data_types(test_df_copy, columns=['Age', 'Fare'], target_type='float')
    
    # Remove duplicates
    remove_duplicates(train_df_copy, inplace=True)
    
    print("Data consistency ensured successfully.")
    
    # Save cleaned datasets
    train_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_train.csv', index=False)
    test_df_copy.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_test.csv', index=False)
    
    print("Cleaned datasets saved successfully.")
    


    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np  # Added import for numpy
    
    # Load cleaned data
    train_data_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_train.csv'
    test_data_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/cleaned_test.csv'
    
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Ensure correct data types
    train_df['Pclass'] = train_df['Pclass'].astype('int')
    train_df['SibSp'] = train_df['SibSp'].astype('int')
    train_df['Parch'] = train_df['Parch'].astype('int')
    train_df['Survived'] = train_df['Survived'].astype('int')
    
    # Statistical Summary
    print("Statistical Summary:")
    print(train_df.describe())
    
    # Univariate Analysis of Numerical Features
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df[feature].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_distribution.png')
        plt.close()
    
    # Univariate Analysis of Categorical Features
    categorical_features = ['Sex', 'Embarked']
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train_df, x=feature)
        plt.title(f'Count of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_count.png')
        plt.close()
    
    # Survival Rate Analysis
    categorical_features_with_survival = ['Sex', 'Pclass', 'Embarked']
    
    for feature in categorical_features_with_survival:
        plt.figure(figsize=(10, 6))
        survival_rate = train_df[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by='Survived', ascending=False)
        sns.barplot(x=feature, y='Survived', data=survival_rate)
        plt.title(f'Survival Rate by {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_survival_rate.png')
        plt.close()
    
    
    # Bivariate Analysis: Correlation Matrix
    
    # Select only numeric columns from train_df
    numeric_train_df = train_df.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix
    correlation_matrix = numeric_train_df.corr()
    
    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/correlation_matrix.png')
    plt.close()
    
    # Bivariate Analysis: Pairwise Plots
    sns.pairplot(train_df, hue='Survived', vars=numerical_features)
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/pairwise_plots.png')
    plt.close()
    
    # Cross-Tabulation and Stacked Bar Plots
    for feature in categorical_features_with_survival:
        cross_tab = pd.crosstab(train_df[feature], train_df['Survived'])
        cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Survived vs. Not Survived by {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_survival_stacked_bar.png')
        plt.close()
    
    # Box Plots for Numerical-Categorical Interaction
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=train_df, x='Survived', y=feature)
        plt.title(f'{feature} vs. Survived')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_boxplot_survived.png')
        plt.close()
    
    
    # Multivariate Plots: Violin Plots
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=train_df, x='Survived', y=feature)
        plt.title(f'{feature} Distribution by Survival')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/{feature}_violinplot_survived.png')
        plt.close()
    
    # Facet Grid Plots
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=30)
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/age_facetgrid_survived.png')
    plt.close()
    
    # Interaction Plots
    sns.lmplot(data=train_df, x='Fare', y='Age', hue='Survived', fit_reg=False, scatter_kws={'alpha':0.5})
    plt.title('Fare vs Age by Survival')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/titanic/deep_eda/images/fare_age_interaction.png')
    plt.close()
    
    
    # Summary Report and Recommendations
    summary_report = """
    ### Summary of Insights from In-depth EDA ###
    
    1. **Age**:
       - Younger passengers had a higher survival rate.
       - Significant age distribution differences between survivors and non-survivors.
    
    2. **Fare**:
       - Higher ticket fares are associated with higher survival rates.
       - Wide range of fare values across different classes.
    
    3. **Pclass**:
       - Passengers in higher classes (Pclass 1) had a significantly higher survival rate.
    
    4. **Sex**:
       - Female passengers had a much higher survival rate compared to males.
    
    5. **Embarked**:
       - Passengers embarked from Cherbourg (C) had a higher survival rate.
    
    ### Recommendations for Feature Engineering ###
    
    1. **Imputation**: 
       - Impute missing values for 'Age' using median values stratified by 'Pclass' and 'Sex'.
       - Consider imputing 'Embarked' based on the most frequent port.
    
    2. **Feature Scaling**:
       - Normalize or standardize 'Fare' to handle wide range of values.
    
    3. **Encoding**:
       - Encode categorical features like 'Sex' and 'Embarked' using one-hot encoding.
    
    4. **Feature Interaction**:
       - Create interaction features such as 'Pclass*Age' and 'FamilySize' (SibSp + Parch).
    
    5. **New Features**:
       - Extract titles from 'Name' and create a new 'Title' feature.
       - Create a 'CabinKnown' feature indicating whether cabin information is present.
    """
    
    print(summary_report)
    


if __name__ == "__main__":
    generated_code_function()
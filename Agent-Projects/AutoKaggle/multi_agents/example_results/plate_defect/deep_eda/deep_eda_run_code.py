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
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load cleaned data
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/cleaned_test.csv'
    
    train_df = pd.read_csv(cleaned_train_path)
    test_df = pd.read_csv(cleaned_test_path)
    
    # Task 1: Conduct Thorough Statistical Analysis on Cleaned Data
    
    # Descriptive statistics for numerical features
    numerical_columns = [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
        'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 
        'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness',
        'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 
        'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 
        'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
        'SigmoidOfAreas'
    ]
    
    print("Descriptive Statistics for Numerical Features:")
    print(train_df[numerical_columns].describe())
    
    # Frequency distribution for categorical features
    categorical_columns = ['TypeOfSteel_A300', 'TypeOfSteel_A400']
    print("\nFrequency Distribution for Categorical Features:")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(train_df[col].value_counts())
    
    # Correlation matrix for numerical features
    correlation_matrix = train_df[numerical_columns].corr()
    
    # Visualize correlation matrix using a heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    heatmap_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/correlation_matrix_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"Heatmap of the correlation matrix saved to {heatmap_path}")
    
    
    # Task 2: Explore Relationships Between Features and Target Variables
    
    target_variables = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    # Plot histograms and boxplots for numerical features, stratified by target variables
    for feature in numerical_columns[:5]:  # Limiting to first 5 for brevity
        for target in target_variables:
            plt.figure(figsize=(10, 5))
            sns.histplot(data=train_df, x=feature, hue=target, multiple='stack', kde=True)
            plt.title(f'Histogram of {feature} stratified by {target}')
            hist_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/hist_{feature}_{target}.png'
            plt.savefig(hist_path)
            plt.close()
            print(f"Histogram saved to {hist_path}")
    
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=train_df, x=target, y=feature)
            plt.title(f'Boxplot of {feature} stratified by {target}')
            boxplot_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/boxplot_{feature}_{target}.png'
            plt.savefig(boxplot_path)
            plt.close()
            print(f"Boxplot saved to {boxplot_path}")
    
    # Count plots for categorical features stratified by target variables
    for col in categorical_columns:
        for target in target_variables:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=train_df, x=col, hue=target)
            plt.title(f'Count Plot of {col} stratified by {target}')
            countplot_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/countplot_{col}_{target}.png'
            plt.savefig(countplot_path)
            plt.close()
            print(f"Count Plot saved to {countplot_path}")
    
    # Scatter plots for numerical features against target variables
    for feature in numerical_columns[:5]:  # Limiting to first 5 for brevity
        for target in target_variables:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(data=train_df, x=feature, y=target, hue=target)
            plt.title(f'Scatter Plot of {feature} vs {target}')
            scatter_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/scatter_{feature}_{target}.png'
            plt.savefig(scatter_path)
            plt.close()
            print(f"Scatter Plot saved to {scatter_path}")
    
    
    from statsmodels.graphics.factorplots import interaction_plot
    
    # Task 3: Identify Potential Feature Interactions
    
    # Interaction plots for pairs of numerical features
    for feature1 in numerical_columns[:3]:  # Limiting to first 3 pairs for brevity
        for feature2 in numerical_columns[3:6]:
            plt.figure(figsize=(10, 5))
            sns.lmplot(data=train_df, x=feature1, y=feature2, hue='Pastry', fit_reg=False)
            plt.title(f'Interaction Plot of {feature1} and {feature2}')
            interaction_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/interaction_{feature1}_{feature2}.png'
            plt.savefig(interaction_path)
            plt.close()
            print(f"Interaction Plot saved to {interaction_path}")
    
    # Cross-tabulation for categorical features
    for col1 in categorical_columns:
        for col2 in target_variables:
            crosstab_result = pd.crosstab(train_df[col1], train_df[col2])
            print(f"\nCross-tabulation between {col1} and {col2}:\n")
            print(crosstab_result)
    
    
    # Task 4: Visualize Key Insights and Patterns
    
    # Summary visualizations
    plt.figure(figsize=(10, 5))
    sns.barplot(x=train_df['TypeOfSteel_A300'], y=train_df['Pastry'])
    plt.title('Bar Plot of TypeOfSteel_A300 vs Pastry')
    summary_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/plate_defect/deep_eda/images/summary_TypeOfSteel_A300_Pastry.png'
    plt.savefig(summary_path)
    plt.close()
    print(f"Summary Bar Plot saved to {summary_path}")
    
    # Recommendations for Feature Engineering
    recommendations = """
    1. Consider creating polynomial features from 'X_Minimum', 'X_Maximum', 'Y_Minimum', and 'Y_Maximum'.
    2. Combine 'Sum_of_Luminosity', 'Minimum_of_Luminosity', and 'Maximum_of_Luminosity' to create new features capturing luminosity ranges.
    3. Use interaction terms between key indices like 'Edges_Index', 'Square_Index', and 'Orientation_Index'.
    4. Consider feature scaling for numerical features to improve model performance.
    """
    
    print("Recommendations for Feature Engineering:")
    print(recommendations)
    


if __name__ == "__main__":
    generated_code_function()
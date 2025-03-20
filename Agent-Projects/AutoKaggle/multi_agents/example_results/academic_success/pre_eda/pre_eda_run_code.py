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
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    
    # Load the train and test datasets
    train_df = pd.read_csv(f'{data_dir}train.csv')
    test_df = pd.read_csv(f'{data_dir}test.csv')
    
    # Make copies for processing
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Inspect data dimensions
    print("Train dataset shape:", train_copy.shape)
    print("Test dataset shape:", test_copy.shape)
    
    # List feature types
    print("\nTrain dataset feature types:")
    print(train_copy.dtypes)
    
    print("\nTest dataset feature types:")
    print(test_copy.dtypes)
    
    # Check for missing values in train dataset
    print("\nMissing values in Train dataset:")
    print(train_copy.isnull().sum())
    
    print("\nPercentage of missing values in Train dataset:")
    print(train_copy.isnull().mean() * 100)
    
    # Check for missing values in test dataset
    print("\nMissing values in Test dataset:")
    print(test_copy.isnull().sum())
    
    print("\nPercentage of missing values in Test dataset:")
    print(test_copy.isnull().mean() * 100)
    
    # Examine Target variable distribution
    print("\nTarget Variable Distribution in Train dataset:")
    print(train_copy['Target'].value_counts(normalize=True) * 100)
    
    
    # Define numerical features based on the data description
    numerical_features = [
        'Previous qualification (grade)',
        'Admission grade',
        'Age at enrollment',
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
        'GDP'
    ]
    
    print("\nSummary Statistics for Numerical Features:")
    print(train_copy[numerical_features].describe())
    
    print("\nSkewness of Numerical Features:")
    print(train_copy[numerical_features].skew())
    
    print("\nKurtosis of Numerical Features:")
    print(train_copy[numerical_features].kurt())
    
    # Identify outliers using IQR
    print("\nOutliers in Numerical Features (based on IQR):")
    for feature in numerical_features:
        Q1 = train_copy[feature].quantile(0.25)
        Q3 = train_copy[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = train_copy[(train_copy[feature] < lower_bound) | (train_copy[feature] > upper_bound)][feature]
        print(f"\n{feature}:")
        print(f"Number of outliers: {outliers.count()}")
        print(outliers.tolist())
    
    
    # Define categorical features based on the data description
    categorical_features = [
        'Marital status',
        'Application mode',
        'Course',
        'Previous qualification',
        'Nacionality',
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        'Displaced',
        'Educational special needs',
        'Debtor',
        'Tuition fees up to date',
        'Gender',
        'Scholarship holder',
        'International'
    ]
    
    print("\nUnivariate Analysis of Categorical Features:")
    
    for feature in categorical_features:
        print(f"\nFeature: {feature}")
        value_counts = train_copy[feature].value_counts(dropna=False)
        value_percent = train_copy[feature].value_counts(normalize=True, dropna=False) * 100
        freq_table = pd.concat([value_counts, value_percent], axis=1)
        freq_table.columns = ['Count', 'Percentage']
        print(freq_table)
        
        # Identify dominant and rare categories
        dominant_category = value_percent.idxmax()
        dominant_percentage = value_percent.max()
        rare_categories = value_percent[value_percent < 5].index.tolist()
        print(f"Dominant category: {dominant_category} ({dominant_percentage:.2f}%)")
        if rare_categories:
            print(f"Rare categories (less than 5%): {rare_categories}")
        else:
            print("No rare categories (all categories have >=5%)")
        
        # Assess data quality
        unique_values = train_copy[feature].unique()
        print(f"Unique categories ({len(unique_values)}): {unique_values}")
        # Check for inconsistencies (simple checks: unique to lower case)
        normalized_values = train_copy[feature].astype(str).str.strip().str.lower().unique()
        if len(normalized_values) != len(unique_values):
            print("Potential inconsistencies found in category names (case, leading/trailing spaces).")
        else:
            print("No inconsistencies detected in category names.")
    
    
    # Initial Correlation and Relationship Assessment
    
    print("\nCorrelation Matrix for Numerical Features:")
    corr_matrix = train_copy[numerical_features].corr()
    print(corr_matrix)
    
    # Identify highly correlated features (|corr| > 0.7)
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
    threshold = 0.7
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((feat1, feat2, corr_value))
                print(f"{feat1} and {feat2} have a correlation of {corr_value:.2f}")
    
    if not high_corr_pairs:
        print("No feature pairs with correlation above the threshold.")
    
    # Assess potential multicollinearity
    print("\nAssessment of Potential Multicollinearity:")
    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"Potential multicollinearity between {pair[0]} and {pair[1]} (Correlation: {pair[2]:.2f})")
        print("Consider removing or combining the highly correlated features to reduce multicollinearity.")
    else:
        print("No significant multicollinearity detected based on the correlation threshold.")
    
    # Crosstab Analysis for Categorical Features vs Target
    print("\nCrosstab Analysis between Categorical Features and Target Variable:")
    for feature in categorical_features:
        print(f"\nCrosstab for {feature} and Target:")
        crosstab = pd.crosstab(train_copy[feature], train_copy['Target'], normalize='index') * 100
        print(crosstab)
        # Identify strong associations (example: difference in distribution)
        print("Note: A more detailed analysis may be needed to quantify the strength of associations.")
    


if __name__ == "__main__":
    generated_code_function()
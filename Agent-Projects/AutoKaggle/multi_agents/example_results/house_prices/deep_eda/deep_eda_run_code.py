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
    
    # Load the data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/test.csv')
    
    # Define columns with missing values
    numerical_cols_with_missing = [
        'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'
    ]
    
    categorical_cols_with_missing = [
        'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
        'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'MSZoning', 'Utilities', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'
    ]
    
    # Handle missing values for numerical columns using median
    train_df = fill_missing_values(train_df, numerical_cols_with_missing, method='median')
    test_df = fill_missing_values(test_df, numerical_cols_with_missing, method='median')
    
    # Handle missing values for categorical columns using 'Missing'
    train_df = fill_missing_values(train_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    test_df = fill_missing_values(test_df, categorical_cols_with_missing, method='constant', fill_value='Missing')
    
    # Remove columns with more than 50% missing values if needed
    train_df = remove_columns_with_missing_data(train_df, thresh=0.5)
    test_df = remove_columns_with_missing_data(test_df, thresh=0.5)
    
    pass
    
    # Define columns to check for outliers
    outlier_columns = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
    
    # Handle outliers by capping them to the IQR bounds
    train_df = detect_and_handle_outliers_iqr(train_df, outlier_columns, factor=1.5, method='clip')
    test_df = detect_and_handle_outliers_iqr(test_df, outlier_columns, factor=1.5, method='clip')
    
    pass
    
    # Convert data types if necessary
    numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = train_df.select_dtypes(include=[object]).columns.tolist()
    
    train_df = convert_data_types(train_df, numerical_columns, target_type='float')
    # Assuming numerical_columns is a list of numerical columns including 'SalePrice'
    existing_columns = [col for col in numerical_columns if col in test_df.columns]
    if existing_columns:
        test_df = convert_data_types(test_df, existing_columns, target_type='float')
    else:
        pass
    
    train_df = convert_data_types(train_df, categorical_columns, target_type='str')
    
    # Ensure categorical_columns only contains columns present in both train_df and test_df
    common_categorical_columns = [col for col in categorical_columns if col in train_df.columns and col in test_df.columns]
    
    # Now call the convert_data_types function with the cleaned list
    test_df = convert_data_types(test_df, common_categorical_columns, target_type='str')
    
    # Drop 'FireplaceQu' from train_df if it is not in test_df
    if 'FireplaceQu' not in test_df.columns and 'FireplaceQu' in train_df.columns:
        train_df = train_df.drop(columns=['FireplaceQu'])
    
    # Remove duplicates if any
    train_df = remove_duplicates(train_df, columns=None)
    test_df = remove_duplicates(test_df, columns=None)
    
    pass
    
    # Save cleaned datasets
    train_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train.csv', index=False)
    test_df.to_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load cleaned data
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/cleaned_test.csv'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Univariate Analysis
    # Numerical Features
    numerical_features = ['SalePrice', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt']
    
    # Summary statistics
    print(train_data[numerical_features].describe())
    
    # Histograms
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(numerical_features):
        plt.subplot(3, 2, i+1)
        sns.histplot(train_data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/univariate_numerical_histograms.png')
    plt.close()
    
    # Categorical Features
    categorical_features = ['MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle']
    
    for feature in categorical_features:
        print(f"\nFrequency of categories in {feature}:")
        print(train_data[feature].value_counts())
    
    # Bivariate Analysis
    # Correlation matrix
    numeric_train_data = train_data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_train_data.corr()
    
    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/correlation_matrix.png')
    plt.close()
    
    # Scatter plots for significant numerical features
    important_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea', 'OverallQual']
    
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(important_features):
        plt.subplot(2, 2, i+1)
        sns.scatterplot(x=train_data[feature], y=train_data['SalePrice'])
        plt.title(f'{feature} vs SalePrice')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
    plt.tight_layout()
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/bivariate_scatter_plots.png')
    plt.close()
    
    # Box plots for key categorical features
    plt.figure(figsize=(14, 12))
    for i, feature in enumerate(categorical_features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x=train_data[feature], y=train_data['SalePrice'])
        plt.title(f'{feature} vs SalePrice')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/bivariate_box_plots.png')
    plt.close()
    
    # Feature Interactions
    interaction_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'SalePrice']
    
    # Pairwise scatter plots
    sns.pairplot(train_data[interaction_features])
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/pairwise_scatter_plots.png')
    plt.close()
    
    # Groupby interaction analysis
    interaction_analysis = train_data.groupby(['OverallQual', 'YearBuilt'])['SalePrice'].agg(['mean', 'count']).reset_index()
    print(interaction_analysis.head())
    
    # Visualizing Key Insights and Patterns
    # Bar plot for OverallQual vs SalePrice
    plt.figure(figsize=(10, 6))
    sns.barplot(x='OverallQual', y='SalePrice', data=train_data)
    plt.title('OverallQual vs SalePrice')
    plt.xlabel('OverallQual')
    plt.ylabel('SalePrice')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/deep_eda/images/overallqual_vs_saleprice.png')
    plt.close()
    
    # Recommendations for Feature Engineering
    recommendations = """
    1. Consider `OverallQual` as a significant predictor for `SalePrice`.
    2. Create interaction features between `OverallQual` and `YearBuilt`.
    3. Consider neighborhood-based features, as `Neighborhood` shows significant variance in `SalePrice`.
    4. Engineer features to capture the overall property size, combining `GrLivArea`, `TotalBsmtSF`, and `LotArea`.
    """
    
    print(recommendations)
    


if __name__ == "__main__":
    generated_code_function()
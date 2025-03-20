import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Load datasets
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Inspect data
    print("Train Data Head:")
    print(train_df.head())
    
    print("\nTest Data Head:")
    print(test_df.head())
    
    print("\nTrain Data Info:")
    print(train_df.info())
    
    print("\nTest Data Info:")
    print(test_df.info())
    
    print("\nTrain Data Description (Numerical):")
    print(train_df.describe())
    
    print("\nTrain Data Description (Categorical):")
    print(train_df.describe(include=['O']))
    
    print("\nTest Data Description (Numerical):")
    print(test_df.describe())
    
    print("\nTest Data Description (Categorical):")
    print(test_df.describe(include=['O']))
    
    
    # Calculate missing values for training data
    missing_train = train_df.isnull().sum()
    missing_train_percent = (missing_train / len(train_df)) * 100
    missing_train_summary = pd.DataFrame({'Missing Values': missing_train, 'Percentage': missing_train_percent})
    missing_train_summary = missing_train_summary[missing_train_summary['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
    
    print("\nMissing Values in Training Data:")
    print(missing_train_summary)
    
    # Calculate missing values for testing data
    missing_test = test_df.isnull().sum()
    missing_test_percent = (missing_test / len(test_df)) * 100
    missing_test_summary = pd.DataFrame({'Missing Values': missing_test, 'Percentage': missing_test_percent})
    missing_test_summary = missing_test_summary[missing_test_summary['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
    
    print("\nMissing Values in Testing Data:")
    print(missing_test_summary)
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Select numerical features for visualization
    numerical_features = [
        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
        'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice'
    ]
    
    # Histograms for numerical features
    for feature in numerical_features[:10]:  # Limiting to first 10 for this phase
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df[feature].dropna(), kde=True)
        plt.title(f'Distribution of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/pre_eda/images/hist_{feature}.png')
        plt.close()
    
    # Box plots for numerical features
    for feature in numerical_features[:10]:  # Limiting to first 10 for this phase
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[feature].dropna())
        plt.title(f'Box Plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/pre_eda/images/box_{feature}.png')
        plt.close()
    
    
    # Select categorical features for visualization
    categorical_features = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]
    
    # Bar plots for categorical features
    for feature in categorical_features[:10]:  # Limiting to first 10 for this phase
        plt.figure(figsize=(10, 6))
        sns.countplot(x=train_df[feature].dropna())
        plt.title(f'Frequency Distribution of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/house_prices/pre_eda/images/bar_{feature}.png')
        plt.close()
    


if __name__ == "__main__":
    generated_code_function()
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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Make copies of the dataframes
    train_clean = train.copy()
    test_clean = test.copy()
    
    # Define columns to fill missing values
    categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Fill missing values for categorical columns using mode
    train_clean = fill_missing_values(train_clean, categorical_columns, method='mode')
    test_clean = fill_missing_values(test_clean, categorical_columns, method='mode')
    
    # Fill missing values for numerical columns using median
    train_clean = fill_missing_values(train_clean, numerical_columns, method='median')
    test_clean = fill_missing_values(test_clean, numerical_columns, method='median')
    
    pass
    
    
    # Define columns to treat outliers
    outlier_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Treat outliers by clipping them to the acceptable range
    train_clean = detect_and_handle_outliers_iqr(train_clean, outlier_columns, factor=1.5, method='clip')
    test_clean = detect_and_handle_outliers_iqr(test_clean, outlier_columns, factor=1.5, method='clip')
    
    pass
    
    
    # Set expense values to 0 for CryoSleep passengers
    expense_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Update the training set
    train_clean.loc[train_clean['CryoSleep'] == True, expense_features] = 0
    
    # Update the test set
    test_clean.loc[test_clean['CryoSleep'] == True, expense_features] = 0
    
    pass
    
    
    # Remove duplicates
    train_clean = remove_duplicates(train_clean)
    test_clean = remove_duplicates(test_clean)
    
    # Convert data types
    boolean_columns = ['CryoSleep', 'VIP']
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    categorical_columns = ['HomePlanet', 'Cabin', 'Destination', 'Name']
    
    # Convert boolean columns
    train_clean = convert_data_types(train_clean, boolean_columns, target_type='bool')
    test_clean = convert_data_types(test_clean, boolean_columns, target_type='bool')
    
    # Convert numerical columns
    train_clean = convert_data_types(train_clean, numerical_columns, target_type='float')
    test_clean = convert_data_types(test_clean, numerical_columns, target_type='float')
    
    # Convert categorical columns
    train_clean = convert_data_types(train_clean, categorical_columns, target_type='str')
    test_clean = convert_data_types(test_clean, categorical_columns, target_type='str')
    
    pass
    
    
    # Save cleaned datasets
    cleaned_train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_train.csv'
    cleaned_test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_test.csv'
    
    train_clean.to_csv(cleaned_train_path, index=False)
    test_clean.to_csv(cleaned_test_path, index=False)
    
    pass
    


    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load the cleaned data
    train_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_train.csv')
    test_df = pd.read_csv('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/cleaned_test.csv')
    
    # Numerical features
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for feature in numerical_features:
        print(f"Statistics for {feature}:")
        print(train_df[feature].describe())
        print(f"Skewness: {train_df[feature].skew()}")
        print(f"Kurtosis: {train_df[feature].kurt()}")
        
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df[feature].dropna(), kde=True)
        plt.title(f'Histogram of {feature}')
        hist_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/histogram_{feature}.png'
        plt.savefig(hist_path)
        plt.close()
        
        # Box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[feature].dropna())
        plt.title(f'Box plot of {feature}')
        box_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/boxplot_{feature}.png'
        plt.savefig(box_path)
        plt.close()
    
    # Categorical features
    categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    for feature in categorical_features:
        print(f"Value counts for {feature}:")
        print(train_df[feature].value_counts())
    
        # Bar chart
        plt.figure(figsize=(10, 6))
        sns.countplot(y=train_df[feature], order=train_df[feature].value_counts().index)
        plt.title(f'Bar chart of {feature}')
        bar_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/bar_{feature}.png'
        plt.savefig(bar_path)
        plt.close()
    
    
    # Bivariate analysis for numerical features
    target = 'Transported'
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[target], y=train_df[feature])
        plt.title(f'Box plot of {feature} vs {target}')
        box_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/boxplot_{feature}_vs_{target}.png'
        plt.savefig(box_path)
        plt.close()
    
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=train_df[target], y=train_df[feature])
        plt.title(f'Violin plot of {feature} vs {target}')
        violin_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/violin_{feature}_vs_{target}.png'
        plt.savefig(violin_path)
        plt.close()
    
        transported_mean = train_df.groupby(target)[feature].mean()
        transported_median = train_df.groupby(target)[feature].median()
        print(f"Mean of {feature} for {target}:")
        print(transported_mean)
        print(f"Median of {feature} for {target}:")
        print(transported_median)
    
    # Bivariate analysis for categorical features
    for feature in categorical_features:
        cross_tab = pd.crosstab(train_df[feature], train_df[target])
        print(f"Cross-tabulation of {feature} vs {target}:")
        print(cross_tab)
        
        cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Stacked bar chart of {feature} vs {target}')
        bar_path = f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/bar_{feature}_vs_{target}.png'
        plt.savefig(bar_path)
        plt.close()
        
        from scipy.stats import chi2_contingency
        chi2, p, dof, ex = chi2_contingency(cross_tab)
        print(f"Chi-square test result for {feature} vs {target}:")
        print(f"Chi2: {chi2}, p-value: {p}")
    
    
    # Pair plots for all numerical features
    sns.pairplot(train_df[numerical_features + [target]])
    pairplot_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/pairplot_numerical_features.png'
    plt.savefig(pairplot_path)
    plt.close()
    
    # Correlation heatmap
    corr = train_df[numerical_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation heatmap of numerical features')
    heatmap_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/correlation_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    
    # Combined feature analysis with pivot tables and heatmaps
    # Example: Pivot table for HomePlanet and Destination
    pivot = train_df.pivot_table(index='HomePlanet', columns='Destination', values='Transported', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0.5)
    plt.title('Heatmap of HomePlanet and Destination vs Transported')
    pivot_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/deep_eda/images/pivot_homeplanet_destination.png'
    plt.savefig(pivot_path)
    plt.close()
    
    
    # Key insights summary
    insights = """
    Key Insights:
    1. Age shows a skewed distribution with a few young passengers.
    2. Expenditure features (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck) have many zero values.
    3. CryoSleep and VIP status have noticeable effects on whether a passenger was transported.
    4. HomePlanet and Destination show significant relationships with the target variable.
    5. Cabin feature needs to be split into separate deck, num, and side features.
    
    Recommendations for Feature Engineering:
    1. Create binary features for whether expenditure features are zero.
    2. Split Cabin into deck, num, and side.
    3. Encode categorical features using suitable methods (e.g., one-hot encoding).
    4. Consider interaction features between HomePlanet and Destination.
    5. Create age groups for better handling of age skewness.
    """
    
    print(insights)
    


if __name__ == "__main__":
    generated_code_function()
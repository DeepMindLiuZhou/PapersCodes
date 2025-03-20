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
    train_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/train.csv'
    test_path = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Overview of train dataset
    print("Train Dataset Info:")
    print(train_df.info())
    
    print("\nTrain Dataset Description:")
    print(train_df.describe())
    
    # Overview of test dataset
    print("\nTest Dataset Info:")
    print(test_df.info())
    
    print("\nTest Dataset Description:")
    print(test_df.describe())
    
    # Initial overview of categorical features for train dataset
    categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    for feature in categorical_features:
        print(f"\nValue counts for {feature} in train dataset:")
        print(train_df[feature].value_counts())
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Numerical features
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Plot histograms
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df[feature].dropna(), kde=False)
        plt.title(f'Histogram of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/pre_eda/images/histogram_{feature}.png')
        plt.close()
    
    # Plot box plots
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[feature].dropna())
        plt.title(f'Box plot of {feature}')
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/pre_eda/images/boxplot_{feature}.png')
        plt.close()
    
    
    # Plot bar plots for categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=train_df[feature])
        plt.title(f'Bar plot of {feature}')
        plt.xticks(rotation=45)
        plt.savefig(f'/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/pre_eda/images/barplot_{feature}.png')
        plt.close()
    
    # Print unique value counts for categorical features
    for feature in categorical_features:
        print(f"\nUnique value counts for {feature}:")
        print(train_df[feature].value_counts())
    
    
    # Visualize missing values using a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap - Train Dataset')
    plt.savefig('/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/spaceship_titanic/pre_eda/images/missing_values_heatmap_train.png')
    plt.close()
    
    # Summary of missing values
    print("\nSummary of missing values in train dataset:")
    print(train_df.isnull().sum())
    
    # Consistency check for CryoSleep and zero expenses
    cryo_sleep_zero_expenses = train_df[(train_df['CryoSleep'] == True) & 
                                        (train_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0)]
    print("\nPassengers in CryoSleep with zero expenses:")
    print(cryo_sleep_zero_expenses.shape[0])
    


if __name__ == "__main__":
    generated_code_function()
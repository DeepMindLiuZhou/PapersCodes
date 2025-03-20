import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    import os
    
    # Define file paths
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    cleaned_train_file = os.path.join(data_dir, 'cleaned_train.csv')
    cleaned_test_file = os.path.join(data_dir, 'cleaned_test.csv')
    
    # Load the datasets
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        pass
    except Exception as e:
        pass
        raise
    
    # Define numerical columns
    numerical_cols = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
    
    # TASK 1: Detect and Handle Outliers in Numerical Features Using the IQR Method
    
    # Handle outliers in training data by capping them instead of removing
    try:
        train_df_original = train_df.copy()  # Preserve original training data for comparison
        
        train_df = detect_and_handle_outliers_iqr(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=3.0,              # Increased factor to reduce sensitivity
            method='clip'            # Changed method from 'remove' to 'clip'
        )
        pass
        
        # Optional: Log the number of values capped per column
        for col in numerical_cols:
            # Compare before and after to determine if capping occurred
            original_max = train_df_original[col].max()
            original_min = train_df_original[col].min()
            capped_max = train_df[col].max()
            capped_min = train_df[col].min()
            
            if capped_max < original_max or capped_min > original_min:
                pass
    except Exception as e:
        pass
        raise
    
    # Handle outliers in testing data by clipping them
    try:
        test_df = detect_and_handle_outliers_iqr(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=numerical_cols,
            factor=1.5,
            method='clip'
        )
        pass
    except Exception as e:
        pass
        raise
    
    # Save the cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        pass
    except Exception as e:
        pass
        raise
    
    
    # Reload the cleaned datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        pass
    except Exception as e:
        pass
        raise
    
    # TASK 2: Verify and Ensure Consistency in Categorical Features Across Datasets
    
    # Identify unique categories in both datasets
    train_colors = set(train_df['color'].dropna().unique())
    test_colors = set(test_df['color'].dropna().unique())
    
    pass
    
    # Step 1: Standardize categories by converting to lowercase
    train_df['color'] = train_df['color'].str.lower()
    test_df['color'] = test_df['color'].str.lower()
    
    # Step 2: Re-identify unique categories after standardization
    train_colors_std = set(train_df['color'].dropna().unique())
    test_colors_std = set(test_df['color'].dropna().unique())
    
    pass
    
    # Step 3: Verify consistency
    if not train_colors_std.issubset(test_colors_std) or not test_colors_std.issubset(train_colors_std):
        # Find discrepancies
        all_colors = train_colors_std.union(test_colors_std)
        pass
    
        # For this example, we'll map any unseen categories in the test set to 'unknown'
        # Identify categories in test not present in train
        unseen_in_test = test_colors_std - train_colors_std
        if unseen_in_test:
            test_df['color'] = test_df['color'].replace(list(unseen_in_test), 'unknown')
            pass
    
        # Similarly, handle any categories in train not present in test, if necessary
        unseen_in_train = train_colors_std - test_colors_std
        if unseen_in_train:
            train_df['color'] = train_df['color'].replace(list(unseen_in_train), 'unknown')
            pass
    else:
        pass
    
    # Step 4: Re-validate unique categories after handling discrepancies
    train_colors_final = set(train_df['color'].dropna().unique())
    test_colors_final = set(test_df['color'].dropna().unique())
    
    pass
    
    # Save the standardized datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        pass
    except Exception as e:
        pass
        raise
    
    
    # Reload the cleaned and standardized datasets
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        pass
    except Exception as e:
        pass
        raise
    
    # TASK 3: Validate and Convert Data Types for All Features
    
    # Define data type conversions for training data
    type_conversions_train = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str',
        'type': 'str'
    }
    
    # Define data type conversions for testing data
    type_conversions_test = {
        'id': 'int',
        'bone_length': 'float',
        'rotting_flesh': 'float',
        'hair_length': 'float',
        'has_soul': 'float',
        'color': 'str'
    }
    
    # Convert data types for training data
    try:
        # Convert categorical columns first to avoid issues during numeric conversions
        train_df = convert_data_types(
            data=train_df,
            columns=['color', 'type'],
            target_type='str'
        )
        
        # Convert numerical columns
        train_df = convert_data_types(
            data=train_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        train_df['id'] = pd.to_numeric(train_df['id'], errors='coerce').astype('Int64')
        
        pass
    except Exception as e:
        pass
        raise
    
    # Convert data types for testing data
    try:
        # Convert categorical columns first
        test_df = convert_data_types(
            data=test_df,
            columns=['color'],
            target_type='str'
        )
        
        # Convert numerical columns
        test_df = convert_data_types(
            data=test_df,
            columns=['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'id'],
            target_type='float'  # Temporarily convert 'id' to float to handle NaNs
        )
        
        # Now convert 'id' to int using pandas' Int64 for nullable integers
        test_df['id'] = pd.to_numeric(test_df['id'], errors='coerce').astype('Int64')
        
        pass
    except Exception as e:
        pass
        raise
    
    # Save the datasets with updated data types
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        pass
    except Exception as e:
        pass
        raise
    
    
    # Reload the datasets with updated data types
    try:
        train_df = pd.read_csv(cleaned_train_file)
        test_df = pd.read_csv(cleaned_test_file)
        pass
    except Exception as e:
        pass
        raise
    
    # TASK 4: Confirm Absence of Duplicate Records
    
    # Remove duplicates in training data based on 'id'
    try:
        initial_train_shape = train_df.shape
        train_df = remove_duplicates(
            data=train_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_train_shape = train_df.shape
        pass
    except Exception as e:
        pass
        raise
    
    # Remove duplicates in testing data based on 'id'
    try:
        initial_test_shape = test_df.shape
        test_df = remove_duplicates(
            data=test_df.copy(),  # Work on a copy to preserve original data
            columns=['id'],
            keep='first'
        )
        final_test_shape = test_df.shape
        pass
    except Exception as e:
        pass
        raise
    
    # Verify absence of duplicates
    train_duplicates = train_df.duplicated(subset=['id']).sum()
    test_duplicates = test_df.duplicated(subset=['id']).sum()
    
    pass
    
    # Save the final cleaned datasets
    try:
        train_df.to_csv(cleaned_train_file, index=False)
        test_df.to_csv(cleaned_test_file, index=False)
        pass
    except Exception as e:
        pass
        raise
    


    
    import pandas as pd
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Define file paths
    data_dir = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/ghouls_goblins_and_ghosts_boo/'
    images_dir = os.path.join(data_dir, 'deep_eda', 'images')
    cleaned_train_file = os.path.join(data_dir, 'cleaned_train.csv')
    cleaned_test_file = os.path.join(data_dir, 'cleaned_test.csv')
    
    # Load the cleaned datasets
    train_df = pd.read_csv(cleaned_train_file)
    test_df = pd.read_csv(cleaned_test_file)
    
    # Combine train and test for unified analysis if necessary
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Define numerical features
    numerical_features = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
    
    # Create descriptive statistics
    print("Descriptive Statistics for Numerical Features:\n")
    desc_stats = train_df[numerical_features].describe().T
    desc_stats['median'] = train_df[numerical_features].median()
    desc_stats['mode'] = train_df[numerical_features].mode().iloc[0]
    desc_stats['variance'] = train_df[numerical_features].var()
    desc_stats['skewness'] = train_df[numerical_features].skew()
    desc_stats['kurtosis'] = train_df[numerical_features].kurtosis()
    print(desc_stats[['mean', 'median', 'mode', 'std', 'variance', 'skewness', 'kurtosis', 'min', '25%', '50%', '75%', 'max']])
    
    # Visualization: Histograms with Density Plots
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.histplot(train_df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        histogram_path = os.path.join(images_dir, f'{feature}_distribution.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f"Histogram for '{feature}' saved to '{histogram_path}'.")
    
    # Visualization: Box Plots
    for feature in numerical_features:
        plt.figure(figsize=(6, 8))
        sns.boxplot(y=train_df[feature], color='lightgreen')
        plt.title(f'Box Plot of {feature}')
        plt.ylabel(feature)
        boxplot_path = os.path.join(images_dir, f'{feature}_boxplot.png')
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Box plot for '{feature}' saved to '{boxplot_path}'.")
    
    # Visualization: Violin Plots
    for feature in numerical_features:
        plt.figure(figsize=(6, 8))
        sns.violinplot(y=train_df[feature], color='lightcoral')
        plt.title(f'Violin Plot of {feature}')
        plt.ylabel(feature)
        violinplot_path = os.path.join(images_dir, f'{feature}_violinplot.png')
        plt.savefig(violinplot_path)
        plt.close()
        print(f"Violin plot for '{feature}' saved to '{violinplot_path}'.")
    
    
    # Define categorical features
    categorical_features = ['color']
    target_variable = 'type'
    
    # Frequency Distribution for 'color'
    color_counts = train_df['color'].value_counts()
    color_percentages = train_df['color'].value_counts(normalize=True) * 100
    color_distribution = pd.DataFrame({'Count': color_counts, 'Percentage': color_percentages.round(2)})
    print("\nFrequency Distribution of 'color':\n")
    print(color_distribution)
    
    # Visualization: Bar Chart for 'color' Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=color_counts.index, y=color_counts.values, palette='muted')
    plt.title('Frequency of Each Color Category')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    color_bar_path = os.path.join(images_dir, 'color_frequency_bar.png')
    plt.savefig(color_bar_path)
    plt.close()
    print(f"Bar chart for 'color' distribution saved to '{color_bar_path}'.")
    
    # Association between 'color' and 'type'
    color_type_ct = pd.crosstab(train_df['color'], train_df['type'])
    print("\nContingency Table between 'color' and 'type':\n")
    print(color_type_ct)
    
    # Visualization: Grouped Bar Chart for 'color' vs 'type'
    color_type_ct.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Color Distribution Across Creature Types')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Type')
    color_type_bar_path = os.path.join(images_dir, 'color_type_grouped_bar.png')
    plt.tight_layout()
    plt.savefig(color_type_bar_path)
    plt.close()
    print(f"Grouped bar chart for 'color' vs 'type' saved to '{color_type_bar_path}'.")
    
    
    # Correlation Analysis
    pearson_corr = train_df[numerical_features].corr(method='pearson')
    spearman_corr = train_df[numerical_features].corr(method='spearman')
    
    print("\nPearson Correlation Matrix:\n")
    print(pearson_corr)
    
    print("\nSpearman Correlation Matrix:\n")
    print(spearman_corr)
    
    # Visualization: Heatmap of Pearson Correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10}, linewidths=.5)
    plt.title('Pearson Correlation Heatmap')
    heatmap_path = os.path.join(images_dir, 'pearson_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Pearson correlation heatmap saved to '{heatmap_path}'.")
    
    # Visualization: Heatmap of Spearman Correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10}, linewidths=.5)
    plt.title('Spearman Correlation Heatmap')
    spearman_heatmap_path = os.path.join(images_dir, 'spearman_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(spearman_heatmap_path)
    plt.close()
    print(f"Spearman correlation heatmap saved to '{spearman_heatmap_path}'.")
    
    # Scatter Plots for Selected Feature Pairs
    selected_pairs = [('bone_length', 'rotting_flesh'), ('hair_length', 'has_soul')]
    
    for x, y in selected_pairs:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=train_df, x=x, y=y, hue='type', palette='deep')
        plt.title(f'Scatter Plot of {x} vs {y} by Type')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend(title='Type')
        scatter_path = os.path.join(images_dir, f'scatter_{x}_vs_{y}.png')
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        print(f"Scatter plot for '{x}' vs '{y}' saved to '{scatter_path}'.")
    
    
    # Box Plots of Numerical Features by Type
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='type', y=feature, data=train_df, palette='Set2')
        plt.title(f'Box Plot of {feature} by Type')
        plt.xlabel('Type')
        plt.ylabel(feature)
        boxplot_type_path = os.path.join(images_dir, f'{feature}_boxplot_by_type.png')
        plt.tight_layout()
        plt.savefig(boxplot_type_path)
        plt.close()
        print(f"Box plot for '{feature}' by 'type' saved to '{boxplot_type_path}'.")
    
    # Violin Plots of Numerical Features by Type
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='type', y=feature, data=train_df, palette='Set3')
        plt.title(f'Violin Plot of {feature} by Type')
        plt.xlabel('Type')
        plt.ylabel(feature)
        violinplot_type_path = os.path.join(images_dir, f'{feature}_violinplot_by_type.png')
        plt.tight_layout()
        plt.savefig(violinplot_type_path)
        plt.close()
        print(f"Violin plot for '{feature}' by 'type' saved to '{violinplot_type_path}'.")
    
    # Summary Dashboard: Assembling Key Plots
    # Since creating a multi-plot dashboard programmatically can be complex and may exceed the image limit,
    # we'll create individual key visuals and ensure the most insightful are highlighted.
    
    print("\nKey insights visualizations have been saved individually in the images directory.")
    


if __name__ == "__main__":
    generated_code_function()
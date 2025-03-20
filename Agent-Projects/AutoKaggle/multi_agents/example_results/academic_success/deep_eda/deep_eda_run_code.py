import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import pandas as pd
    
    # Define the data directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    CLEANED_DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    
    # Load the datasets
    train_path = DATA_DIR + 'train.csv'
    test_path = DATA_DIR + 'test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Create copies to avoid modifying original data
    cleaned_train_df = train_df.copy()
    cleaned_test_df = test_df.copy()
    
    # Define numerical and categorical columns
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Task 1: Fill Missing Values
    
    # Fill missing numerical features with median
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=numerical_cols,
        method='median'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=numerical_cols,
        method='median'
    )
    
    # Fill missing categorical features with mode
    cleaned_train_df = fill_missing_values(
        data=cleaned_train_df,
        columns=categorical_cols,
        method='mode'
    )
    cleaned_test_df = fill_missing_values(
        data=cleaned_test_df,
        columns=categorical_cols,
        method='mode'
    )
    
    # Remove columns with more than 60% missing values
    cleaned_train_df = remove_columns_with_missing_data(
        data=cleaned_train_df,
        thresh=0.6
    )
    cleaned_test_df = remove_columns_with_missing_data(
        data=cleaned_test_df,
        thresh=0.6
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step1.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step1.csv', index=False)
    
    
    # Task 2: Detect and Handle Outliers in Numerical Features
    
    # Define the numerical columns again in case some were removed in Task 1
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    # Handle possible removal of some numerical columns in Task 1
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Detect and handle outliers by clipping
    cleaned_train_df = detect_and_handle_outliers_iqr(
        data=cleaned_train_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    cleaned_test_df = detect_and_handle_outliers_iqr(
        data=cleaned_test_df,
        columns=numerical_cols,
        factor=1.5,
        method='clip'
    )
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step2.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step2.csv', index=False)
    
    
    # Task 3: Normalize and Standardize Categorical Features
    
    # Define categorical columns again in case some were removed in Task 1
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    # Handle possible removal of some categorical columns in Task 1
    categorical_cols = [col for col in categorical_cols if col in cleaned_train_df.columns]
    
    # Convert categorical columns to string type
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=categorical_cols,
        target_type='str'
    )
    
    # Normalize categorical strings: lowercase and strip whitespaces
    for col in categorical_cols:
        cleaned_train_df[col] = cleaned_train_df[col].str.lower().str.strip()
        cleaned_test_df[col] = cleaned_test_df[col].str.lower().str.strip()
    
    # Placeholder for correcting common typos
    # Define a dictionary for typo corrections if available
    # Example:
    # typo_mapping = {
    #     'marial status': 'marital status',
    #     'scholarship holder': 'scholarship_holder',
    #     # Add more mappings as identified
    # }
    
    # Uncomment and modify the following lines if typo mappings are available
    # for col, mapping in typo_mapping.items():
    #     cleaned_train_df[col] = cleaned_train_df[col].replace(mapping)
    #     cleaned_test_df[col] = cleaned_test_df[col].replace(mapping)
    
    # Save the cleaned datasets for verification (optional)
    # cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train_step3.csv', index=False)
    # cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test_step3.csv', index=False)
    
    
    # Task 4: Remove Duplicate Rows and Convert Data Types
    
    # Include 'id' in the columns to consider for duplicate removal
    columns_to_consider_train = cleaned_train_df.columns.tolist()  # This includes 'id'
    cleaned_train_df = remove_duplicates(
        data=cleaned_train_df,
        columns=columns_to_consider_train,
        keep='first'
    )
    
    # Remove duplicates based on all columns except 'id' for the test dataset
    columns_to_consider_test = [col for col in cleaned_test_df.columns if col != 'id']
    cleaned_test_df = remove_duplicates(
        data=cleaned_test_df,
        columns=columns_to_consider_test,
        keep='first'
    )
    
    # Define numerical columns again after Task 1 and Task 2
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP', 'Age at enrollment', 'Application order'
    ]
    
    # Ensure numerical columns exist after previous tasks
    numerical_cols = [col for col in numerical_cols if col in cleaned_train_df.columns]
    
    # Convert numerical columns to appropriate data types
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=numerical_cols,
        target_type='float'  # Assuming all are floats; adjust if some are integers
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=numerical_cols,
        target_type='float'
    )
    
    # Convert 'id' column to string
    cleaned_train_df = convert_data_types(
        data=cleaned_train_df,
        columns=['id'],
        target_type='str'
    )
    
    cleaned_test_df = convert_data_types(
        data=cleaned_test_df,
        columns=['id'],
        target_type='str'
    )
    
    # Verify data types (optional)
    # print(cleaned_train_df.dtypes)
    # print(cleaned_test_df.dtypes)
    
    # Save the final cleaned datasets
    cleaned_train_df.to_csv(CLEANED_DATA_DIR + 'cleaned_train.csv', index=False)
    cleaned_test_df.to_csv(CLEANED_DATA_DIR + 'cleaned_test.csv', index=False)
    
    pass
    


    
    import pandas as pd
    import numpy as np
    from scipy.stats import skew, kurtosis
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Define the data directories
    DATA_DIR = '/mnt/d/PythonProjects/AutoKaggleMaster/multi_agents/competition/classification_with_an_academic_success_dataset/'
    CLEANED_DATA_DIR = DATA_DIR
    IMAGE_DIR = os.path.join(DATA_DIR, 'deep_eda', 'images')
    
    # Create the image directory if it doesn't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Load the cleaned training dataset
    cleaned_train_path = os.path.join(CLEANED_DATA_DIR, 'cleaned_train.csv')
    train_df = pd.read_csv(cleaned_train_path)
    
    # Define numerical columns
    numerical_cols = [
        'Previous qualification (grade)', 'Admission grade',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP', 'Age at enrollment'
    ]
    
    # Ensure all numerical features are correctly typed
    train_df[numerical_cols] = train_df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
    # Initialize a dictionary to store descriptive statistics
    descriptive_stats = {}
    
    print("Comprehensive Univariate Analysis of Numerical Features:\n")
    
    for col in numerical_cols:
        if col in train_df.columns:
            col_data = train_df[col].dropna()
            stats = {
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Standard Deviation': col_data.std(),
                'Min': col_data.min(),
                '25%': col_data.quantile(0.25),
                '50%': col_data.quantile(0.50),
                '75%': col_data.quantile(0.75),
                'Max': col_data.max(),
                'Skewness': skew(col_data),
                'Kurtosis': kurtosis(col_data)
            }
            descriptive_stats[col] = stats
            # Print the descriptive statistics
            print(f"Feature: {col}")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")
            print("\n")
    
            # Generate Histogram
            plt.figure(figsize=(8, 4))
            sns.histplot(col_data, kde=True, bins=30, color='blue', edgecolor='black', stat="density")
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            histogram_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_histogram.png')
            plt.savefig(histogram_path, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram for '{col}' at {histogram_path}")
    
            # Generate Box Plot
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=col_data, color='orange')
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
            boxplot_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_boxplot.png')
            plt.savefig(boxplot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved box plot for '{col}' at {boxplot_path}\n")
    
    # Optionally, save the descriptive statistics to a CSV file
    descriptive_stats_df = pd.DataFrame(descriptive_stats).T
    descriptive_stats_df.to_csv(os.path.join(IMAGE_DIR, 'numerical_descriptive_stats.csv'))
    print("Descriptive statistics saved to 'numerical_descriptive_stats.csv'")
    
    
    # Define categorical columns
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced',
        'Educational special needs', 'Debtor', 'Tuition fees up to date',
        'Gender', 'Scholarship holder', 'International'
    ]
    
    print("Detailed Univariate Analysis of Categorical Features:\n")
    
    for col in categorical_cols:
        if col in train_df.columns:
            col_data = train_df[col].dropna()
            value_counts = col_data.value_counts()
            proportions = col_data.value_counts(normalize=True) * 100
            category_df = pd.DataFrame({
                'Count': value_counts,
                'Proportion (%)': proportions.round(2)
            })
            print(f"Feature: {col}")
            print(category_df)
            print("\n")
    
            # Save the value counts to a CSV file
            category_csv_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_value_counts.csv')
            category_df.to_csv(category_csv_path)
            print(f"Saved value counts for '{col}' at {category_csv_path}")
    
            # Identify categories with less than 5% proportion
            low_freq = category_df[category_df['Proportion (%)'] < 5]
            if not low_freq.empty:
                print(f"Categories in '{col}' with less than 5% proportion:")
                print(low_freq)
                print("\n")
    
            # Generate Bar Plot
            plt.figure(figsize=(10, 6))
            sns.countplot(data=train_df, x=col, order=value_counts.index, palette='viridis')
            plt.title(f'Count Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            barplot_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_countplot.png')
            plt.savefig(barplot_path)
            plt.close()
            print(f"Saved count plot for '{col}' at {barplot_path}\n")
    
    
    from scipy.stats import chi2_contingency
    from sklearn.preprocessing import LabelEncoder
    
    # Define target variable
    target = 'Target'
    if target not in train_df.columns:
        print(f"Target variable '{target}' not found in the dataset.")
    else:
        print("Bivariate Analysis Between Features and Target Variable:\n")
    
        # Initialize dictionaries to store association metrics
        numerical_associations = {}
        categorical_associations = {}
    
        # Bivariate Analysis for Numerical Features
        print("Numerical Features Analysis:\n")
        for col in numerical_cols:
            if col in train_df.columns:
                print(f"Feature: {col}")
                # Group data by target and calculate descriptive stats
                grouped = train_df.groupby(target)[col].describe()
                print(grouped)
                print("\n")
    
                # Calculate Pearson Correlation with Target
                # Encode target variable
                le = LabelEncoder()
                target_encoded = le.fit_transform(train_df[target])
                correlation = train_df[col].corr(pd.Series(target_encoded))
                numerical_associations[col] = correlation
                print(f"Pearson Correlation between '{col}' and '{target}': {correlation:.4f}\n")
    
                # Generate Box Plot
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=target, y=col, data=train_df, palette='Set2')
                plt.title(f'Box Plot of {col} by {target}')
                plt.xlabel(target)
                plt.ylabel(col)
                boxplot_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_vs_{target}_boxplot.png')
                plt.savefig(boxplot_path, bbox_inches='tight')
                plt.close()
                print(f"Saved box plot for '{col}' vs '{target}' at {boxplot_path}\n")
    
        # Bivariate Analysis for Categorical Features
        print("Categorical Features Analysis:\n")
        for col in categorical_cols:
            if col in train_df.columns:
                print(f"Feature: {col}")
                contingency_table = pd.crosstab(train_df[col], train_df[target])
                print(contingency_table)
                print("\n")
    
                # Perform Chi-Square Test
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                categorical_associations[col] = p
                print(f"Chi-Square Test for '{col}' and '{target}': p-value = {p:.4f}\n")
    
                # Generate Stacked Bar Plot
                contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
                plt.title(f'Stacked Bar Plot of {col} by {target}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title=target)
                plt.tight_layout()
                stacked_bar_path = os.path.join(IMAGE_DIR, f'{col.replace(" ", "_")}_vs_{target}_stacked_bar.png')
                plt.savefig(stacked_bar_path)
                plt.close()
                print(f"Saved stacked bar plot for '{col}' vs '{target}' at {stacked_bar_path}\n")
    
        # Summary of Associations
        print("Summary of Feature Associations with Target Variable:\n")
    
        # Numerical Features with significant correlations
        print("Numerical Features Correlation with Target:")
        significant_numerical = {k: v for k, v in numerical_associations.items() if abs(v) > 0.1}
        for feature, corr_value in significant_numerical.items():
            print(f"  {feature}: Pearson Correlation = {corr_value:.4f}")
        print("\n")
    
        # Categorical Features with significant associations (p-value < 0.05)
        print("Categorical Features Chi-Square Test p-values:")
        significant_categorical = {k: v for k, v in categorical_associations.items() if v < 0.05}
        for feature, p_value in significant_categorical.items():
            print(f"  {feature}: p-value = {p_value:.4f}")
        print("\n")
    
        # Save association metrics to CSV files
        numerical_assoc_df = pd.DataFrame.from_dict(numerical_associations, orient='index', columns=['Pearson Correlation'])
        numerical_assoc_df.to_csv(os.path.join(IMAGE_DIR, 'numerical_target_correlations.csv'))
        categorical_assoc_df = pd.DataFrame.from_dict(categorical_associations, orient='index', columns=['Chi2 p-value'])
        categorical_assoc_df.to_csv(os.path.join(IMAGE_DIR, 'categorical_target_chi2_pvalues.csv'))
        print("Association metrics saved to 'numerical_target_correlations.csv' and 'categorical_target_chi2_pvalues.csv'")
    
    
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("Correlation and Multivariate Analysis of Numerical Features:\n")
    
    # Compute the correlation matrix
    corr_matrix = train_df[numerical_cols].corr(method='pearson')
    
    # Select correlations with absolute value greater than 0.7
    high_corr = corr_matrix[(corr_matrix.abs() > 0.7) & (corr_matrix.abs() < 1.0)]
    
    # Display high correlations
    print("Pairs of Numerical Features with |Pearson Correlation| > 0.7:\n")
    high_corr_pairs = high_corr.stack().reset_index()
    high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Pearson Correlation']
    print(high_corr_pairs)
    print("\n")
    
    # Save high correlations to CSV
    high_corr_pairs.to_csv(os.path.join(IMAGE_DIR, 'high_correlations_numerical_features.csv'), index=False)
    print("High correlation pairs saved to 'high_correlations_numerical_features.csv'\n")
    
    # Calculate Variance Inflation Factor (VIF)
    # VIF is calculated for each feature by regressing it against all other features
    # First, drop any columns with missing values
    vif_data = train_df[numerical_cols].dropna()
    
    # Add a constant term for VIF calculation
    X = sm.add_constant(vif_data)
    
    vif = pd.DataFrame()
    vif['Feature'] = numerical_cols
    vif['VIF'] = [variance_inflation_factor(X.values, i+1) for i in range(len(numerical_cols))]  # +1 to skip the constant
    
    print("Variance Inflation Factor (VIF) for Numerical Features:\n")
    print(vif)
    print("\n")
    
    # Save VIF results to CSV
    vif.to_csv(os.path.join(IMAGE_DIR, 'vif_numerical_features.csv'), index=False)
    print("VIF results saved to 'vif_numerical_features.csv'\n")
    
    # Recommendations based on VIF
    high_vif = vif[vif['VIF'] > 5]
    if not high_vif.empty:
        print("Features with VIF > 5, indicating multicollinearity:\n")
        print(high_vif)
        print("\nConsider removing or combining these features to reduce multicollinearity.\n")
    else:
        print("No features with VIF > 5 detected. Multicollinearity is not a concern.\n")
    
    # Visualize the correlation matrix with a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmax=1, vmin=-1, linewidths=0.5, annot_kws={"size":8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    heatmap_path = os.path.join(IMAGE_DIR, 'numerical_features_correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap at {heatmap_path}\n")
    


if __name__ == "__main__":
    generated_code_function()
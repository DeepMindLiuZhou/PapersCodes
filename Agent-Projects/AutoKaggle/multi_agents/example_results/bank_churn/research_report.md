# COMPETITION RESEARCH REPORT

## 1. PRELIMINARY EDA
In the preliminary exploratory data analysis (EDA), we examined the dataset consisting of various features related to bank customer behavior, focusing on the target variable `Exited`. Key findings included:
- **Feature Overview**: Initially, the dataset comprised several numerical and categorical features, including `CreditScore`, `Age`, `Geography`, and `Gender`. Notably, no features were deleted or created during this phase.
- **Outlier Detection**: Outliers were identified in the `Age` and `CreditScore` features, indicating potential areas for further investigation.
- **Statistics**: Basic statistics indicated a wide range in features like `EstimatedSalary`, while `Age` showed outliers that could impact analyses.
- **Missing Values**: The dataset had 0% missing values, simplifying subsequent cleaning efforts.

### Initial Actions and Rationale
The main action from this phase was to visualize distributions and identify outliers, which laid the groundwork for targeted data cleaning and transformation in subsequent phases.

## 2. DATA CLEANING
During the data cleaning phase, significant steps were taken to enhance data quality:
- **Duplicates**: All duplicates were removed, ensuring a unique dataset for analysis.
- **Outlier Handling**: Outliers in `CreditScore`, `Age`, and other numerical features were clipped to mitigate their impact on modeling.
- **Data Type Consistency**: Features were standardized to their appropriate data types to ensure consistency.

### Significant Cleaning Actions
- **Outlier Clipping**: For example, `CreditScore` and `Age` were clipped based on the Interquartile Range (IQR) to maintain robust statistical analyses.
- **Data Type Adjustments**: Numeric features were ensured to be in float format, while binary features were converted to integers. This consistency is crucial for effective model training.

## 3. DEEP EDA
The deep EDA phase provided profound insights into feature relationships and distributions:
- **Statistical Analysis**: Distributions of numerical features were visualized, revealing insights such as potential transformations for `Balance` and `EstimatedSalary`.
- **Correlation Analysis**: A correlation matrix was created, helping to identify relationships between features and the target variable `Exited`, indicating the need for feature interactions.

### Findings Influence
Insights from this phase guided feature engineering decisions, particularly in creating new categorical features based on observed interactions, such as combining `Geography` and `Balance`.

## 4. FEATURE ENGINEERING
Feature engineering involved creating new features and transforming existing ones to optimize model performance:
- **New Features**: Created `AgeGroup`, `HasBalance`, `Region_Balance_Interaction`, and `Active_Card_User` to capture useful patterns.
- **Transformations**: Features like `CreditScore` were normalized, `EstimatedSalary` was log-transformed, and `Age` was standardized.

### Reasoning and Expected Impact
These transformations aimed to improve model interpretability and performance. For instance, creating `AgeGroup` helps segment the population, enhancing predictive capabilities.

## 5. MODEL BUILDING, VALIDATION, AND PREDICTION
Three models were trained: Logistic Regression, Random Forest, and XGBoost. The model performances were as follows:
1. **Logistic Regression**: AUC ROC Score: 0.8415
2. **Random Forest**: AUC ROC Score: 0.8621
3. **XGBoost**: AUC ROC Score: 0.8666 (Best Performing Model)

### Rationale Behind Model Choices
The choice of models was based on their ability to handle binary classification problems effectively. Hyperparameter tuning was suggested for further enhancement, and categorical variables were encoded appropriately to avoid pitfalls like the dummy variable trap.

## 6. CONCLUSION
The competition's approach involved thorough data exploration, cleaning, and transformation, which led to successful model training and validation. Key insights included:
- **Feature Relevance**: Features such as `Age`, `Balance`, and engineered features were critical for performance.
- **Future Recommendations**: Emphasizing hyperparameter tuning, error handling, and detailed feature importance analysis can enhance future iterations.

Overall, the decisions made throughout the process were pivotal in achieving a robust predictive model for customer churn.
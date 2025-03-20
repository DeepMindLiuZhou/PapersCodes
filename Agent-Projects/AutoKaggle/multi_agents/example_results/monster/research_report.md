# COMPETITION RESEARCH REPORT

## 1. PRELIMINARY EDA
The initial exploratory data analysis (EDA) focused on understanding the dataset's structure and identifying any immediate concerns. Key findings included:
- **Features Identified**: The dataset contained several features, including `bone_length`, `rotting_flesh`, `hair_length`, `has_soul`, `color`, and the target variable `type`.
- **Missing Values**: There were no missing values in the dataset.
- **Outliers**: Outliers were detected in numerical features using the Interquartile Range (IQR) method, although specific counts were not detailed.
- **Unique Values**: Categorical features such as `color` had consistent unique values across training and testing datasets.

**Actions Taken**: Recommendations for data cleaning included addressing outliers, verifying feature consistency, and preparing for feature engineering. This laid the groundwork for a thorough cleaning phase.

## 2. DATA CLEANING
During the data cleaning phase, the following actions were taken:
- **Outlier Handling**: Outliers in numerical features (`bone_length`, `rotting_flesh`, `hair_length`, `has_soul`) were clipped in the training dataset, while testing data outliers were capped to maintain size.
- **Standardization**: The `color` feature was standardized to lowercase to ensure consistency across datasets.
- **Duplicate Removal**: Duplicates were checked and removed, enhancing dataset integrity.

**Rationale**: These steps were crucial for ensuring data quality, which is essential for effective modeling. The cleaned datasets were then saved for subsequent analysis.

## 3. DEEP EDA
In-depth analysis provided rich insights into feature distributions and correlations:
- **Statistical Summaries**: Descriptive statistics revealed the need for scaling or transformation, particularly for features with high skewness.
- **Correlations**: Strong correlations were identified between numerical features and the target variable `type`, guiding feature selection for modeling.
- **Visual Insights**: Violin plots illustrated significant differences in feature distributions among different types, indicating their potential importance in classification tasks.

**Influence on Decisions**: The findings emphasized the need for careful feature selection and the creation of interaction terms, which would potentially enhance model performance.

## 4. FEATURE ENGINEERING
Feature engineering involved creating new features and modifying existing ones:
- **New Features Created**:
  - Interaction terms such as `hair_length_has_soul` and `bone_length_rotting_flesh`.
  - Polynomial features like `bone_length_squared` and `rotting_flesh_squared`.
  - Ratio features such as `bone_to_flesh_ratio` and `soul_to_hair_ratio`.
- **Encoding**: The `color` feature was transformed through One-Hot Encoding to capture its relationship with the target variable.

**Expected Impact**: These transformations aimed to improve the model's ability to capture complex relationships, enhancing predictive power.

## 5. MODEL BUILDING, VALIDATION, AND PREDICTION
The model building phase involved several crucial steps:
- **Model Selection**: Models were tested using cross-validation to mitigate overfitting due to the limited size of the training dataset.
- **Best Model**: XGBoost was selected as the best model based on performance metrics.
- **Prediction Generation**: Predictions were made for the test set, ensuring proper formatting for submission.

**Rationale**: The structured approach to training and validation ensured robust model performance, essential for effective predictions in a competitive setting.

## 6. CONCLUSION
The overall approach taken throughout the competition emphasized thorough data preparation, insightful exploratory analysis, and strategic feature engineering. Key insights included the importance of handling outliers, the impact of feature interactions, and careful model selection.

**Impactful Decisions**: The decision to clip outliers and create interaction features were particularly impactful, leading to enhanced model performance. Future recommendations include implementing more robust error handling, utilizing additional performance metrics, and exploring data augmentation techniques to further improve predictive capabilities.
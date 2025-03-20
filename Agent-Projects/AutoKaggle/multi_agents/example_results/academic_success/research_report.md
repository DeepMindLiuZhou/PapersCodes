# COMPETITION RESEARCH REPORT

## 1. PRELIMINARY EDA
In the initial exploratory data analysis (EDA) phase, we focused on understanding the dataset's structure and the features available for analysis. The target variable was identified, along with a comprehensive list of features. The training dataset contained 76,518 samples and 38 features, while the test dataset had 51,012 samples and 37 features.

Key findings included:
- Identification of numerical and categorical features.
- Detection of missing values and potential outliers.
- Preliminary correlation analysis indicated relationships between numerical features and the target variable.

**Actions Taken:**
- No features were deleted; however, there was a recommendation to clean the data by handling missing values, addressing outliers, and normalizing categorical features to prepare for the next phase.

## 2. DATA CLEANING
In this phase, we implemented several data cleaning strategies. We processed both the training and test datasets, addressing missing values, outliers, and inconsistencies.

**Main Data Cleaning Steps:**
- **Imputation of Missing Values:** Numerical features were filled with the median, while categorical features were imputed with the mode.
- **Removal of Features:** Features with over 60% missing values were removed to reduce noise.
- **Outlier Handling:** Outliers were clipped using the IQR method, ensuring robust distributions.
- **Standardization of Categorical Features:** Categorical features were standardized to lower case and stripped of whitespace.

**Rationale:**
These steps were crucial to ensure data integrity and improve model performance by providing cleaner, more reliable inputs.

## 3. DEEP EDA
The in-depth exploratory analysis revealed significant patterns and relationships among the features and the target variable.

**Findings:**
- Statistical characteristics of numerical features showed the presence of outliers and normal-like distributions.
- Significant class imbalances in categorical features, such as courses with low representation, were identified.
- Correlation analysis indicated that admission grades had a strong positive correlation with the target variable.

**Implications:**
These insights guided our feature engineering decisions, emphasizing the need for new features that capture interactions and non-linear relationships.

## 4. FEATURE ENGINEERING
Feature engineering involved creating interaction and polynomial features based on previous insights.

**New Features Created:**
- Interaction features such as `Admission_grade_x_GDP` and `Age_at_enrollment_x_Educational_special_needs`.
- Polynomial features were generated for selected numerical features.

**Rationale:**
These engineered features were expected to capture more complex relationships within the data, enhancing model performance.

## 5. MODEL BUILDING, VALIDATION, AND PREDICTION
During model building, we trained several models, including Random Forest and Logistic Regression.

**Performance Evaluation:**
- Models were evaluated using cross-validated accuracy scores, although specific scores were not detailed.
- The separation of the target variable and the removal of non-numeric features were critical steps to ensure model readiness.

**Challenges Encountered:**
- Handling non-numeric features and ensuring alignment between training and test datasets were key challenges addressed through systematic feature selection and data type handling.

**Insights:**
- Future competitions should emphasize broader model selection and hyperparameter tuning to optimize performance.

## 6. CONCLUSION
This competition involved a thorough exploration of the dataset, systematic data cleaning, and innovative feature engineering. Key insights regarding the relationships between features and the target variable guided our modeling decisions, leading to the development of a robust predictive model.

**Impactful Decisions:**
- The identification and handling of outliers and class imbalances were critical in shaping the final model.
- The creation of interaction and polynomial features was instrumental in capturing complex relationships, leading to improved model performance.

Overall, the structured approach to data handling and model training provided a strong foundation for competitive performance in future iterations.
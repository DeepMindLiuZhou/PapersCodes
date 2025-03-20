# COMPETITION RESEARCH REPORT

## 1. PRELIMINARY EDA
The initial exploratory data analysis (EDA) focused on understanding the dataset's structure and identifying potential issues. Key findings included:

- **Datasets Processed:** `train.csv` and `test.csv`.
- **Preliminary Statistics:** Summary statistics for numerical features like `Age`, `Height`, and `Weight`, along with frequency counts for categorical features like `Gender` and `FAVC`.
- **Missing Values:** Several features exhibited missing values, prompting recommendations for imputation strategies.
- **Outliers Identified:** Features such as `Age`, `Height`, and `Weight` showed significant outliers via boxplots.
- **Correlations:** Notable correlations among numerical features were observed, suggesting potential feature engineering opportunities.
  
**Actions Taken:**
- Address missing values through imputation.
- Handle outliers by capping or removal.
- Standardize categorical values for consistency.
- Investigate correlated features for potential combinations.

## 2. DATA CLEANING
The data cleaning phase involved refining the dataset to prepare it for analysis. Key actions included:

- **Missing Values Imputation:** Numerical features were filled with the mean, and categorical features were filled with the mode.
- **Outlier Handling:** Outliers in numerical features were capped using the IQR method.
- **Standardization:** Categorical features were converted to lowercase, and numerical features were cast to appropriate types.

**Significant Cleaning Actions:**
- **Imputation Methods:** Mean for numerical and mode for categorical features were chosen to maintain data integrity without introducing bias.
- **Outlier Capping:** This was necessary to mitigate the impact of extreme values on model performance.

## 3. DEEP EDA
In-depth analysis provided further insights into the dataset. Notable findings included:

- **Statistical Characteristics:** Features like `Age`, `Height`, and `Weight` displayed distinct characteristics influencing future feature engineering decisions.
- **Categorical Relationships:** Strong associations between variables such as `FAVC` and `NObeyesdad` indicated potential predictors for obesity.
- **Chi-Square Tests:** Significant dependencies were found for features like `FAVC` and `CAEC`, guiding model focus.

**Implications for Future Decisions:**
- The relationships observed among features underscored the importance of targeted feature engineering and selection to improve model accuracy.

## 4. FEATURE ENGINEERING
Feature engineering aimed to enrich the dataset for modeling:

- **New Features Created:**
  - **BMI** calculated as `Weight / (Height^2)`.
  - **Age_Category** based on age bins.
  - Categorical transformations for `FCVC` and `NCP`.

**Rationale for New Features:**
- **BMI** provides a standardized measure for obesity risk.
- **Categorical Features** enhance interpretability and improve model performance by simplifying complex relationships.

## 5. MODEL BUILDING, VALIDATION, AND PREDICTION
Model training involved several algorithms, with the **GradientBoostingClassifier** emerging as the best performer:

- **Models Trained:** RandomForest, GradientBoosting, and XGBoost.
- **Cross-Validation Accuracy:** GradientBoosting achieved optimal performance with an average accuracy of 0.900.

**Key Performance Metrics:**
- The confusion matrix indicated effective classification for `Normal_Weight` but misclassifications in obesity categories, suggesting room for improvement.

**Challenges Encountered:**
- No significant overfitting or underfitting was reported, likely due to the robust cross-validation approach.

## 6. CONCLUSION
The research process demonstrated a systematic approach from preliminary EDA through to model validation. Key insights included:

- The importance of lifestyle factors like `FAF`, `TUE`, and `BMI` in predicting obesity.
- Effective feature engineering enhanced model accuracy and interpretability.
- Future steps include exploring additional models and performance metrics to refine predictions further.

Overall, the methodology adopted throughout the competition allowed for a comprehensive understanding of the data and informed decision-making at every phase.
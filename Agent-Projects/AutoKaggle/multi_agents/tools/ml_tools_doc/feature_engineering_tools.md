## one_hot_encode

**Name:** one_hot_encode  
**Description:** Perform one-hot encoding on specified categorical columns. The resulting columns will follow the format 'original_column_value'
**Applicable Situations:** Encoding categorical variables with no ordinal relationship, especially useful for machine learning models that cannot handle categorical data directly (e.g., linear regression, neural networks). Best for categorical variables with relatively few unique categories.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing categorical columns to be encoded.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to encode.
- `handle_unknown`:
  - **Type:** `string`
  - **Description:** How to handle unknown categories during transform.
  - **Enum:** `error` | `ignore`
  - **Default:** `error`

**Required:** `data`, `columns`  
**Result:** DataFrame with one-hot encoded columns  
**Notes:**
- One-hot encoding creates a new binary column for each category in the original column.
- It can significantly increase the number of features, especially for columns with many unique categories.
- May lead to the 'curse of dimensionality' if used on high-cardinality categorical variables.
- Suitable for nominal categorical data where there's no inherent order among categories.
- The function will raise a warning if applied to non-categorical columns.
- Setting handle_unknown='ignore' will create an all-zero row for unknown categories during transform.
- Consider using other encoding methods for high-cardinality features to avoid dimensionality issues.
**Example:**
  - **Input:**
    - `data`: {'color': ['red', 'blue', 'green']}
    - `columns`: color
  - **Output:**
    - `data`: {'color': ['red', 'blue', 'green'],
              'color_blue': [0, 1, 0],
              'color_green': [0, 0, 1],
              'color_red': [1, 0, 0]
            }
    - `columns`: ['color', 'color_blue', 'color_green', 'color_red']

---
## label_encode

**Name:** label_encode  
**Description:** Perform label encoding on specified categorical columns. The resulting columns will follow the format 'original_column_encoded'.  
**Applicable Situations:** Encoding categorical variables with an ordinal relationship, or when the number of categories is large and one-hot encoding would lead to too many features. Useful for tree-based models that can handle categorical data.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing categorical columns to be encoded.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to encode.

**Required:** `data`, `columns`  
**Result:** DataFrame with label encoded columns  
**Notes:**
- Label encoding assigns a unique integer to each category based on alphabetical order.
- It preserves memory compared to one-hot encoding, especially for high-cardinality features.
- Suitable for ordinal categorical data where there's a clear order among categories.
- May introduce an ordinal relationship where none exists, which can be problematic for some models.
- The function will raise a warning if applied to non-categorical columns.
- Tree-based models can often handle label-encoded categorical variables well.
- Be cautious when using with linear models, as they may interpret the labels as having an ordinal relationship.
**Example:**
  - **Input:**
    - `data`: {'fruit': ['apple', 'banana', 'apple', 'cherry']}
    - `columns`: fruit
  - **Output:**
    - `data`: {'fruit': ['apple', 'banana', 'apple', 'cherry'],
              'fruit_encoded': [0, 1, 0, 2]
            }
    - `columns`: ['fruit', 'fruit_encoded']

---
## frequency_encode

**Name:** frequency_encode  
**Description:** Perform frequency encoding on specified categorical columns. The resulting columns will follow the format 'original_column_freq'.  
**Applicable Situations:** Encoding high-cardinality categorical variables, especially when the frequency of categories is informative. Useful for both tree-based and linear models.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing categorical columns to be encoded.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to encode.

**Required:** `data`, `columns`  
**Result:** DataFrame with frequency encoded columns  
**Notes:**
- Frequency encoding replaces each category with its relative frequency in the dataset.
- It can capture some information about the importance of each category.
- Useful for high-cardinality categorical variables where one-hot encoding would create too many features.
- Preserves information about the distribution of categories.
- May be particularly useful when the frequency of a category is informative for the target variable.
- The function will raise a warning if applied to non-categorical columns.
- Be aware that this method can potentially introduce a false sense of ordinality among categories.
**Example:**
  - **Input:**
    - `data`: {'city': ['New York', 'London', 'Paris', 'New York', 'London', 'New York']}
    - `columns`: city
  - **Output:**
    - `data`: {'city': ['New York', 'London', 'Paris', 'New York', 'London', 'New York'],
              'city_freq': [0.5, 0.33, 0.17, 0.5, 0.33, 0.5]
            }
    - `columns`: ['city', 'city_freq']

---
## target_encode

**Name:** target_encode  
**Description:** Perform frequency encoding on specified categorical columns. The resulting columns will follow the format 'original_column_freq'.  
**Applicable Situations:** Encoding categorical variables in supervised learning tasks, especially effective for high-cardinality features. Useful when there's a clear relationship between categories and the target variable.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing categorical columns to be encoded and the target variable.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to encode.
- `target`:
  - **Type:** `string`
  - **Description:** The name of the target column in the DataFrame.
- `min_samples_leaf`:
  - **Type:** `integer`
  - **Description:** Minimum samples to take category average into account.
  - **Default:** `1`
- `smoothing`:
  - **Type:** `number`
  - **Description:** Smoothing effect to balance categorical average vs prior.
  - **Default:** `1.0`

**Required:** `data`, `columns`
**Result:** DataFrame with target encoded columns  
**Notes:**
- Target encoding replaces a categorical value with the mean of the target variable for that value.
- It can capture complex relationships between categorical variables and the target.
- Particularly useful for high-cardinality categorical variables in supervised learning tasks.
- The smoothing parameter helps prevent overfitting, especially for categories with few samples.
- Be cautious of potential data leakage; consider using cross-validation techniques for encoding.
- The function will raise a warning if applied to non-categorical columns.
- This method can be sensitive to outliers in the target variable.
- Consider the impact of target encoding on model interpretability.
**Example:**
  - **Input:**
    - `data`: {'fruit': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'kiwi'], 'region': ['north', 'north', 'south', 'south', 'north', 'south', 'north', 'south', 'north', 'north', 'south'], 'price': [1, 0, 1, 0, 2, 3, 1, 0, 1, 2, 3] }
    - `columns`: ['fruit', 'region', 'price']
  - **Output:**
    - `data`:  {'fruit': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'kiwi'], 'region': ['north', 'north', 'south', 'south', 'north', 'south', 'north', 'south', 'north', 'north', 'south'], 'price': [1, 0, 1, 0, 2, 3, 1, 0, 1, 2, 3], 'fruit_target_enc': [1.437566, 0.912568, 1.437566, 0.796902, 0.912568, 1.437566, 0.796902, 0.912568, 1.437566, 0.796902, 1.750000], 'region_target_enc': [1.509699, 1.509699, 1.250000, 1.250000, 1.509699, 1.250000, 1.509699, 1.250000, 1.509699, 1.509699, 1.250000] }
    - `columns`: ['fruit', 'region', 'price', 'fruit_price_enc', 'region_price_enc']

---
## correlation_feature_selection

**Name:** correlation_feature_selection  
**Description:** Perform feature selection based on correlation analysis. This tool helps identify features that have a strong correlation with the target variable.  
**Applicable Situations:** feature selection, dimensionality reduction, identifying important features for predictive modeling

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset, including features and target.
- `target`:
  - **Type:** `string`
  - **Description:** The name of the target column in the DataFrame.
- `method`:
  - **Type:** `string`
  - **Description:** The correlation method to use.
  - **Enum:** `pearson` | `spearman` | `kendall`
  - **Default:** `pearson`
- `threshold`:
  - **Type:** `number`
  - **Description:** The correlation threshold for feature selection. Features with absolute correlation greater than this value will be selected.
  - **Default:** `0.5`

**Required:** `data`, `target`  
**Result:** DataFrame with selected features and their correlation with the target  
**Notes:**
- Pearson correlation assumes a linear relationship and is sensitive to outliers.
- Spearman correlation is rank-based and can capture monotonic relationships.
- Kendall correlation is another rank-based method, often used for small sample sizes.
- This method is most suitable for numerical features and targets.
- Be cautious with high correlations between features (multicollinearity).
- Consider the domain knowledge when interpreting the results and selecting features.
- This method does not account for interactions between features or non-linear relationships with the target.

---
## variance_feature_selection

**Name:** variance_feature_selection  
**Description:** Perform feature selection based on variance analysis. This tool helps identify and remove features with low variance, which often contribute little to model performance.  
**Applicable Situations:** feature selection, dimensionality reduction, removing constant or near-constant features

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset with features.
- `threshold`:
  - **Type:** `number`
  - **Description:** Features with a variance lower than this threshold will be removed.
  - **Default:** `0.0`
- `columns`:
  - **Type:** ``string` | `array` | `null``
  - **Description:** Column label or sequence of labels to consider. If None, use all columns.
  - **Default:** `None`

**Required:** `data`  
**Result:** DataFrame with selected features and their variances  
**Notes:**
- This method is most suitable for numerical features.
- A threshold of 0.0 will remove features that are constant across all samples.
- For binary features, a threshold of 0.8 * (1 - 0.8) = 0.16 would remove features that have the same value in more than 80% of the samples.
- Consider scaling your features before applying this method if they are on different scales.
- This method does not consider the relationship between features and the target variable.
- Be cautious when using this method with small datasets, as variance estimates may be unreliable.
- Features with high variance are not necessarily informative; consider combining this method with other feature selection techniques.

---
## scale_features

**Name:** scale_features  
**Description:** Scale numerical features in the specified columns of a DataFrame using various scaling methods.  
**Applicable Situations:** feature scaling for numerical data, data preprocessing for numerical features, preparing numerical data for machine learning models that are sensitive to the scale of input features (e.g., neural networks, SVM, K-means clustering)

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset with numerical features to be scaled.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or sequence of labels of numerical features to scale.
- `method`:
  - **Type:** `string`
  - **Description:** The scaling method to use.
  - **Enum:** `standard` | `minmax` | `robust`
  - **Default:** `standard`
- `copy`:
  - **Type:** `boolean`
  - **Description:** If False, try to avoid a copy and do inplace scaling instead.
  - **Default:** `True`

**Required:** `data`, `columns`  
**Result:** DataFrame with scaled features  
**Notes:**
- This function is designed for numerical features only. It should not be used on categorical data.
- StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
- MinMaxScaler: Scales features to a given range, usually between 0 and 1.
- RobustScaler: Scales features using statistics that are robust to outliers.
- Scaling is sensitive to the presence of outliers, especially for StandardScaler and MinMaxScaler.
- RobustScaler is a good choice when your data contains many outliers.
- Scaling should typically be done after splitting your data into training and test sets to avoid data leakage.
- For categorical data, consider using encoding techniques instead of scaling.

---
## perform_pca

**Name:** perform_pca  
**Description:** Perform Principal Component Analysis (PCA) on specified columns of a DataFrame. This tool is useful for dimensionality reduction, feature extraction, and data visualization.  
**Applicable Situations:** dimensionality reduction, feature extraction, data visualization, handling multicollinearity

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset with features.
- `n_components`:
  - **Type:** ``integer` | `number` | `string``
  - **Description:** Number of components to keep. If int, it represents the exact number of components. If float between 0 and 1, it represents the proportion of variance to be retained. If 'mle', Minka's MLE is used to guess the dimension.
  - **Default:** `0.95`
- `columns`:
  - **Type:** ``string` | `array` | `null``
  - **Description:** Column label or sequence of labels to consider. If None, use all columns.
  - **Default:** `None`
- `scale`:
  - **Type:** `boolean`
  - **Description:** Whether to scale the data before applying PCA. Recommended when features are not on the same scale.
  - **Default:** `True`

**Required:** `data`  
**Result:** DataFrame with PCA results  
**Notes:**
- PCA assumes linear relationships between features.
- It's sensitive to the scale of the features, so scaling is often recommended.
- PCA may not be suitable for categorical data or when preserving feature interpretability is crucial.
- The number of components to keep is a trade-off between dimensionality reduction and information retention.
- Consider visualizing the cumulative explained variance to choose an appropriate number of components.
- PCA can help address multicollinearity in regression problems.
- The resulting principal components are orthogonal (uncorrelated) to each other.
- The function now returns only the DataFrame with PCA results, without additional information about explained variance or the PCA model.
**Example:**
  - **Input:**
    - `data`: {'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 4, 5, 4, 5], 'feature3': [3, 6, 7, 8, 9]}
    - `n_components`: 2
  - **Output:**
    - `PC1`: [-2.12132, -0.707107, 0.0, 0.707107, 2.12132]
    - `PC2`: [-0.707107, 0.707107, 0.0, -0.707107, 0.707107]

---
## perform_rfe

**Name:** perform_rfe  
**Description:** Perform Recursive Feature Elimination (RFE) on specified columns of a DataFrame. This tool is useful for feature selection, especially when dealing with high-dimensional data.  
**Applicable Situations:** feature selection, dimensionality reduction, identifying important features for predictive modeling

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset with features.
- `target`:
  - **Type:** ``string` | `pd.Series``
  - **Description:** The target variable. If string, it should be the name of the target column in data.
- `n_features_to_select`:
  - **Type:** ``integer` | `number``
  - **Description:** Number of features to select. If int, it represents the exact number of features. If float between 0 and 1, it represents the proportion of features to select.
  - **Default:** `0.5`
- `step`:
  - **Type:** `integer`
  - **Description:** Number of features to remove at each iteration.
  - **Default:** `1`
- `estimator`:
  - **Type:** `string`
  - **Description:** The estimator to use for feature importance ranking.
  - **Enum:** `auto` | `logistic` | `rf` | `linear` | `rf_regressor`
  - **Default:** `auto`
- `columns`:
  - **Type:** ``string` | `array` | `null``
  - **Description:** Column label or sequence of labels to consider. If None, use all columns except the target (if target is a column name in data).
  - **Default:** `None`

**Required:** `data`, `target`  
**Result:** DataFrame with selected features  
**Notes:**
- RFE is computationally expensive, especially with a large number of features.
- The choice of estimator can significantly affect the results.
- The 'auto' estimator option will automatically choose based on the target variable type.
- RFE does not consider interactions between features.
- The step parameter can be increased to speed up the process, but may result in less optimal feature selection.
- Consider cross-validation for more robust feature selection.
- The selected features may not always be the optimal set for all models or tasks.
- RFE assumes that the importance of a feature is reflected in the magnitude of its coefficient or feature importance score.
- The function returns only the DataFrame with selected features, without additional information about feature rankings or the RFE model.

---
## create_polynomial_features

**Name:** create_polynomial_features  
**Description:** Create polynomial features from specified numerical columns of a DataFrame.  
**Applicable Situations:** Capturing non-linear relationships between features and the target variable. Useful for linear models to learn non-linear patterns, or for enhancing the feature space of any model when non-linear interactions are suspected.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing numerical columns for polynomial feature creation.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to use for creating polynomial features.
- `degree`:
  - **Type:** `integer`
  - **Description:** The degree of the polynomial features.
  - **Default:** `2`
- `interaction_only`:
  - **Type:** `boolean`
  - **Description:** If True, only interaction features are produced.
  - **Default:** `False`
- `include_bias`:
  - **Type:** `boolean`
  - **Description:** If True, include a bias column (all 1s).
  - **Default:** `False`

**Required:** `data`, `columns`  
**Result:** DataFrame with original and new polynomial features  
**Notes:**
- Only works with numerical features. Will raise an error if non-numeric columns are specified.
- Can significantly increase the number of features, potentially leading to overfitting or computational issues.
- Higher degrees capture more complex non-linear relationships but increase the risk of overfitting.
- Consider using regularization techniques when using polynomial features with linear models.
- The function will warn if the resulting DataFrame has over 1000 columns.
- Polynomial features can be particularly useful for regression problems or for capturing complex interactions in classification tasks.
- Be cautious of multicollinearity when using polynomial features, especially with high degrees.

---
## create_feature_combinations

**Name:** create_feature_combinations  
**Description:** Create feature combinations from specified numerical columns of a DataFrame.  
**Applicable Situations:** Capturing interactions between features that may be important for the target variable. Useful for both linear and non-linear models to learn from feature interactions.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing numerical columns for feature combination.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or list of column labels to use for creating feature combinations.
- `combination_type`:
  - **Type:** `string`
  - **Description:** Type of combination to create.
  - **Enum:** `multiplication` | `addition`
  - **Default:** `multiplication`
- `max_combination_size`:
  - **Type:** `integer`
  - **Description:** Maximum number of features to combine.
  - **Default:** `2`

**Required:** `data`, `columns`  
**Result:** DataFrame with original and new combined features  
**Notes:**
- Only works with numerical features. Will raise an error if non-numeric columns are specified.
- Can significantly increase the number of features, potentially leading to overfitting or computational issues.
- Multiplication combinations are useful for capturing non-linear interactions.
- Addition combinations can be useful for creating aggregate features.
- The function will warn if the resulting DataFrame has over 1000 columns.
- Consider the interpretability of the resulting features, especially with high-order combinations.
- Feature combinations can help in discovering complex patterns that individual features might not capture.
- Be mindful of the computational cost, especially with a large number of input features or high max_combination_size.

---
## extract_time_series_features

**Name:** extract_time_series_features  
**Description:** Extract various time series features from specified date and value columns of a DataFrame.  
**Applicable Situations:** Time series analysis, forecasting tasks, and any scenario where temporal patterns and trends are important. Useful for enhancing the feature space of time-based machine learning models.

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame containing time series data.
- `date_column`:
  - **Type:** `string`
  - **Description:** Name of the column containing datetime information.
- `value_column`:
  - **Type:** `string`
  - **Description:** Name of the column containing the time series values.
- `features`:
  - **Type:** `array`
  - **Description:** List of feature types to extract.
  - **Default:** `['basic', 'lags', 'rolling', 'expanding']`
- `lag_periods`:
  - **Type:** `array`
  - **Description:** Periods for lag features.
  - **Default:** `[1, 7, 30]`
- `window_sizes`:
  - **Type:** `array`
  - **Description:** Window sizes for rolling features.
  - **Default:** `[7, 30]`
- `expanding_functions`:
  - **Type:** `array`
  - **Description:** Functions to apply for expanding window features.
  - **Default:** `['mean', 'std', 'min', 'max']`

**Required:** `data`, `date_column`, `value_column`  
**Result:** DataFrame with original and new time series features  
**Notes:**
- Ensures the date column is in datetime format, attempting conversion if necessary.
- Sorts the data by date to ensure proper time-based feature extraction.
- Basic features include year, month, day, day of week, quarter, and weekend indicator.
- Lag features capture past values at specified intervals.
- Rolling features compute statistics over fixed-size windows.
- Expanding features compute cumulative statistics up to each point.
- Warns about potential data leakage if multiple entries per day are detected.
- Alerts users to large gaps in the time series data.
- Cautions if the dataset might be too short for reliable feature extraction.
- Consider the impact of missing data on feature quality, especially for lag and window-based features.
- Be mindful of the look-ahead bias when using these features in predictive models.
- Some features may not be suitable for all types of time series data; choose features based on domain knowledge and the specific problem.

---

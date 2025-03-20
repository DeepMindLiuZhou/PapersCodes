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


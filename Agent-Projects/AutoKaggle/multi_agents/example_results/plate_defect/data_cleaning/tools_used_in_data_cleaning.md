## fill_missing_values

**Name:** fill_missing_values  
**Description:** Fill missing values in specified columns of a DataFrame. This tool can handle both numerical and categorical features by using different filling methods.  
**Applicable Situations:** handle missing values in various types of features

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** The name(s) of the column(s) where missing values should be filled.
- `method`:
  - **Type:** `string`
  - **Description:** The method to use for filling missing values.
  - **Enum:** `auto` | `mean` | `median` | `mode` | `constant`
  - **Default:** `auto`
- `fill_value`:
  - **Type:** ``number` | `string` | `null``
  - **Description:** The value to use when method is 'constant'.
  - **Default:** `None`

**Required:** `data`, `columns`  
**Result:** Successfully fill missing values in the specified column(s) of data  
**Notes:**
- The 'auto' method uses mean for numeric columns and mode for non-numeric columns.
- Using 'mean' or 'median' on non-numeric columns will raise an error.
- The 'mode' method uses the most frequent value, which may not always be appropriate.
- Filling missing values can introduce bias, especially if the data is not missing completely at random.
- Consider the impact of filling missing values on your analysis and model performance.


## detect_and_handle_outliers_iqr

**Name:** detect_and_handle_outliers_iqr  
**Description:** Detect and handle outliers in specified columns using the Interquartile Range (IQR) method. This tool is useful for identifying and managing extreme values in numerical features without assuming a specific distribution of the data.  
**Applicable Situations:** detect and handle outliers in numerical features, especially when the data distribution is unknown, non-normal, or when the dataset is small or contains extreme outliers

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** The name(s) of the column(s) to check for outliers.
- `factor`:
  - **Type:** `number`
  - **Description:** The IQR factor to determine the outlier threshold. Typically 1.5 for outliers or 3 for extreme outliers.
  - **Default:** `1.5`
- `method`:
  - **Type:** `string`
  - **Description:** The method to handle outliers.
  - **Enum:** `clip` | `remove`
  - **Default:** `clip`
- `return_mask`:
  - **Type:** `boolean`
  - **Description:** If True, return a boolean mask indicating outliers instead of removing them.
  - **Default:** `False`

**Required:** `data`, `columns`  
**Result:** Successfully detect and handle outliers in the specified column(s) of data using the IQR method  
**Notes:**
- This method does not assume any specific data distribution.
- It is less sensitive to extreme outliers compared to the Z-score method.
- May be less precise for normally distributed data compared to the Z-score method.
- The choice of factor affects the range of what is considered an outlier.
- Using the 'remove' method may delete data entries, which is not recommended for test sets.


## detect_and_handle_outliers_zscore

**Name:** detect_and_handle_outliers_zscore  
**Description:** Detect and handle outliers in specified columns using the Z-score method. This tool is useful for identifying and managing extreme values in numerical features based on their distance from the mean in terms of standard deviations.  
**Applicable Situations:** detect and handle outliers in numerical features, especially when the data is approximately normally distributed and the sample size is large

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** The name(s) of the column(s) to check for outliers.
- `threshold`:
  - **Type:** `number`
  - **Description:** The Z-score threshold to identify outliers. Values with absolute Z-scores above this threshold are considered outliers. Typically 3.0 or 2.5.
  - **Default:** `3.0`
- `method`:
  - **Type:** `string`
  - **Description:** The method to handle outliers.
  - **Enum:** `clip` | `remove`
  - **Default:** `clip`

**Required:** `data`, `columns`  
**Result:** Successfully detect and handle outliers in the specified column(s) of data  
**Notes:**
- This method assumes the data is approximately normally distributed.
- It may be sensitive to extreme outliers as they can affect the mean and standard deviation.
- Not suitable for highly skewed distributions.
- The choice of threshold affects the sensitivity of outlier detection.


## convert_data_types

**Name:** convert_data_types  
**Description:** Convert the data type of specified columns in a DataFrame. This tool is useful for ensuring data consistency and preparing data for analysis or modeling.  
**Applicable Situations:** data type conversion, data preprocessing, ensuring data consistency across columns

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or sequence of labels to convert.
- `target_type`:
  - **Type:** `string`
  - **Description:** The target data type to convert to.
  - **Enum:** `int` | `float` | `str` | `bool` | `datetime`

**Required:** `data`, `columns`, `target_type`  
**Result:** Successfully convert the data type of specified column(s) in the DataFrame  
**Notes:**
- For 'int' and 'float' conversions, non-numeric values will be converted to NaN.
- The 'int' conversion uses the 'Int64' type, which supports NaN values.
- The 'datetime' conversion will set invalid date/time values to NaT (Not a Time).
- The 'bool' conversion may produce unexpected results for non-boolean data.
- Consider the impact of type conversion on your data analysis and model performance.
- Always verify the results after conversion to ensure data integrity.


## remove_duplicates

**Name:** remove_duplicates  
**Description:** Remove duplicate rows from a DataFrame. This tool provides flexible options for identifying and handling duplicate entries in a dataset.  
**Applicable Situations:** remove duplicate entries from a dataset, especially when dealing with data that may have been entered multiple times or when consolidating data from multiple sources

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array` | `null``
  - **Description:** Column label or sequence of labels to consider for identifying duplicates. If None, use all columns.
  - **Default:** `None`
- `keep`:
  - **Type:** `string`
  - **Description:** Determines which duplicates (if any) to keep.
  - **Enum:** `first` | `last` | `False`
  - **Default:** `first`
- `inplace`:
  - **Type:** `boolean`
  - **Description:** Whether to drop duplicates in place or return a copy.
  - **Default:** `False`

**Required:** `data`  
**Result:** Successfully remove duplicate rows from the DataFrame  
**Notes:**
- If 'columns' is None, all columns are used for identifying duplicates.
- The 'keep' parameter determines which duplicate rows are retained.
- Setting 'inplace' to True will modify the original DataFrame.
- Be cautious when removing duplicates, as it may affect the integrity of your dataset.
- Consider the impact of removing duplicates on your analysis and model performance.
- This method is useful for data cleaning, but make sure you understand why duplicates exist before removing them.


## convert_data_types

**Name:** convert_data_types  
**Description:** Convert the data type of specified columns in a DataFrame. This tool is useful for ensuring data consistency and preparing data for analysis or modeling.  
**Applicable Situations:** data type conversion, data preprocessing, ensuring data consistency across columns

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Column label or sequence of labels to convert.
- `target_type`:
  - **Type:** `string`
  - **Description:** The target data type to convert to.
  - **Enum:** `int` | `float` | `str` | `bool` | `datetime`

**Required:** `data`, `columns`, `target_type`  
**Result:** Successfully convert the data type of specified column(s) in the DataFrame  
**Notes:**
- For 'int' and 'float' conversions, non-numeric values will be converted to NaN.
- The 'int' conversion uses the 'Int64' type, which supports NaN values.
- The 'datetime' conversion will set invalid date/time values to NaT (Not a Time).
- The 'bool' conversion may produce unexpected results for non-boolean data.
- Consider the impact of type conversion on your data analysis and model performance.
- Always verify the results after conversion to ensure data integrity.


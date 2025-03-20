# How to Add Custom Machine Learning Tools

This guide explains how to integrate your custom machine learning tools into our system. The process involves two main steps:

1. Implementing the Function
2. Adding Documentation

## 1. Implementing the Function

Add your custom function to the `ml_tools.py` file. Here's an example using the `fill_missing_values` function:

### Key Requirements:

1. **Type Hints**: Include proper type annotations for all parameters and return values (e.g., `pd.DataFrame`).
2. **Error Handling**: Implement comprehensive error handling mechanisms to:
   - Help AutoKaggle effectively diagnose and correct code issues
   - Provide clear error messages with suggested solutions
   - You can start with a basic implementation and let the LLM enhance the error handling

### Example Implementation:

```Python
def fill_missing_values(data: pd.DataFrame, columns: Union[str, List[str]], method: str = 'auto', fill_value: Any = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): The name(s) of the column(s) to fill missing values.
        method (str, optional): The method to use for filling missing values. 
            Options: 'auto', 'mean', 'median', 'mode', 'constant'. Defaults to 'auto'.
        fill_value (Any, optional): The value to use when method is 'constant'. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with filled missing values.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if method == 'auto':
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == 'median':
            data[column].fillna(data[column].median(), inplace=True)
        elif method == 'mode':
            data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'constant':
            data[column].fillna(fill_value, inplace=True)
        else:
            raise ValueError("Invalid method. Choose from 'auto', 'mean', 'median', 'mode', or 'constant'.")

    return data
```

## 2. Adding Documentation

Your tool requires documentation in two formats:

### 2.1 JSON Schema Documentation

Add an entry to `function_to_schema.json`. You can use LLM to automatically generate this documentation based on your function.


```json
"fill_missing_values": {
    "name": "fill_missing_values",
    "description": "Fill missing values in specified columns of a DataFrame. This tool can handle both numerical and categorical features by using different filling methods.",
    "applicable_situations": "handle missing values in various types of features",
    "parameters": {
        "data": {
            "type": "pd.DataFrame",
            "description": "A pandas DataFrame object representing the dataset."
        },
        "columns": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
            },
            "description": "The name(s) of the column(s) where missing values should be filled."
        },
        "method": {
            "type": "string",
            "description": "The method to use for filling missing values.",
            "enum": ["auto", "mean", "median", "mode", "constant"],
            "default": "auto"
        },
        "fill_value": {
            "type": ["number", "string", "null"],
            "description": "The value to use when method is 'constant'.",
            "default": null
        }
    },
    "required": ["data", "columns"],
    "result": "Successfully fill missing values in the specified column(s) of data",
    "additionalProperties": false,
    "notes": [
        "The 'auto' method uses mean for numeric columns and mode for non-numeric columns.",
        "Using 'mean' or 'median' on non-numeric columns will raise an error.",
        "The 'mode' method uses the most frequent value, which may not always be appropriate.",
        "Filling missing values can introduce bias, especially if the data is not missing completely at random.",
        "Consider the impact of filling missing values on your analysis and model performance."
    ]
},
```

### 2.2 Markdown Documentation

Add corresponding documentation in the appropriate file under `tools/ml_tools_doc/` (e.g., `data_cleaning_tools.md` for data cleaning functions). This documentation is used by the `Planner` for RAG-based tool selection and delegation to the `Developer`.

```markdown
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
```

## NOTE

1. Ensure consistency between implementation and documentation
2. Provide comprehensive examples in the documentation
3. Keep error messages clear and actionable
4. Update both `JSON` schema and `Markdown` documentation when modifying functions


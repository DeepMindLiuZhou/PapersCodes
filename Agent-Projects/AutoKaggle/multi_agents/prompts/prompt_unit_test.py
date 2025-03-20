PROMPT_TEST_SUBMISSION_VALIDITY = '''
# CONTEXT #
You are a data scientist, you are given two csv files: submission.csv and sample_submission.csv.
The first 20 lines of sample_submission.csv is as follows:
{sample_head}
The first 20 lines of submission.csv is as follows:
{submission_head}

#############
# TASK #
You need to check if the submission.csv is valid.
Here are some rules you can refer to:
1. For Id-type columns, the values in submission.csv should be the same as the values in sample_submission.csv.
    - Normally, The first column in the csv files is the Id-type column.
2. For target variable column:
    - If target variable is numerical, the values in submission.csv should be within the range of the values in sample_submission.csv.
        - The range is [mean/5, mean*5], mean is the average value of the target variable in sample_submission.csv.
        - You should calculate the mean value in two files, submission.csv and sample_submission.csv, and then check if the target variable values in submission.csv are within the range.
    - Normally, the target variable column is the last column in the csv files.

#############
# RESPONSE #
Let's work this out in a step by step way. Finally, you should give me the result of this test.
You should follow the format:
```json
{{
    "result": "Valid" or "Invalid",
    "reason": "reason for the result"
}}
```
Here are two examples of the response:
```json
{{  
    "result": "Invalid",  
    "reason": "1. SalePrice (Target variable) values in submission.csv are outside the expected range, seems like log transformation is applied to SalePrice in previous phases, you should reverse the transformation before making submission. \n2. Ids must match."  
}}
```
```json
{{  
    "result": "Valid",  
    "reason": "All Ids match and SalePrice values are within the expected range."  
}}
```
'''

PROMPT_TEST_CLEANED_TRAIN_ID_COLUMNS = '''

'''

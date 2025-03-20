import os
import pandas as pd
import json
import chromadb
import sys
import re
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from memory import Memory, transfer_text_to_json
from memory import Memory
from llm import OpenaiEmbeddings, LLM
from state import State
from utils import load_config
from prompts.prompt_unit_test import *

class TestTool:
    def __init__(
        self, 
        memory: Memory = None,
        model: str = 'gpt-4o-mini',
        type: str = 'api'
    ):
        self.llm = LLM(model, type)
        self.memory = memory
        # self.summary_ducument = summary_ducument
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def execute_tests(self, state: State):
        not_pass_tests = []
        test_function_names = state.phase_to_unit_tests[state.phase]
        for func_name in test_function_names:
            if hasattr(self, func_name): # if the function exists
                func = getattr(self, func_name)
                result = func(state) # return execution result, test number, test information
                if not result[0]: # if the test failed
                    not_pass_tests.append(result)
                    logger.info(f"Test '{func_name}' failed: {result[2]}")
                    if func_name == 'test_document_exist': # if the file does not exist, return directly without further unit test
                        return not_pass_tests
                else:
                    logger.info(f"Test '{func_name}' succeeded") # assert result
            else:
                logger.info(f"Function '{func_name}' not found in TestTool class")
                result = True, 0, f"Function '{func_name}' not found in TestTool class"
        return not_pass_tests

    def test_example(self, state: State):
        return True, 1, "This is an example of unit test detection without focusing on the example"
    
    
    def test_document_exist(self, state: State):
        '''
        Check if the required CSV documents exist in the data_dir.
        '''
        # Check in the state.competition_dir if the documents exist
        # Read all the files in the directory
        files = os.listdir(state.competition_dir)
        
        if state.phase == "Model Building, Validation, and Prediction":
            # Check for the existence of submission.csv
            required_files = ["submission.csv"]
            missing_files = [file for file in required_files if file not in files]
            if not missing_files:
                return True, 2, "submission.csv exists"
            else:
                return False, 2, f"Missing files: {', '.join(missing_files)}, it should be saved in {state.competition_dir}/"

        
        elif state.phase == "Data Cleaning":
            # Check for the existence of cleaned_train and cleaned_test
            required_files = ["cleaned_train.csv", "cleaned_test.csv"]
            missing_files = [file for file in required_files if file not in files]
            
            if not missing_files:
                return True, 2, "cleaned_train.csv and cleaned_test.csv data exist"
            else:
                return False, 2, f"Missing files: {', '.join(missing_files)}, it should be saved in {state.competition_dir}/"
        
        elif state.phase == "Feature Engineering":
            # Check for the existence of processed_train and processed_test
            required_files = ["processed_train.csv", "processed_test.csv"]
            missing_files = [file for file in required_files if file not in files]
            
            if not missing_files:
                return True, 2, "processed_train.csv and processed_test.csv data exist"
            else:
                return False, 2, f"Missing files: {', '.join(missing_files)}, it should be saved in {state.competition_dir}/"
        
        else:
            return True, 2, "Don't need to check the document in this phase"
    
    def test_no_duplicate_cleaned_train(self, state: State):
        '''
        Check if there are any duplicate rows in the csv
        '''
        df = pd.read_csv(f"{state.competition_dir}/cleaned_train.csv")
        duplicates = df.duplicated().sum()
        # the details of the duplicated rows
        duplicated_rows = df[df.duplicated(keep=False)]

        if duplicates == 0:
            return True, 3, "No duplicate rows in cleaned_train.csv"
        else:
            return False, 3, f"There are {duplicates} duplicate rows in the cleaned_train.csv. Rows with duplicated values are: {duplicated_rows.index}"

    def test_no_duplicate_cleaned_test(self, state: State):
        '''
        Check if there are any duplicate rows in the csv
        '''
        df = pd.read_csv(f"{state.competition_dir}/cleaned_test.csv")
        duplicates = df.duplicated().sum()
        # id of the duplicated rows
        duplicates_rows = df[df.duplicated(keep=False)]

        if duplicates == 0:
            return True, 4, "No duplicate rows in cleaned_test.csv"
        else:
            return False, 4, f"There are {duplicates} duplicate rows in the cleaned_test.csv, Rows with duplicated values are: {duplicates_rows.index}"

    def test_no_duplicate_submission(self, state: State):
        '''
        Check if there are any duplicate rows in the csv
        '''
         
        files = os.listdir(state.competition_dir)
        for file in files:
            if file == "submission.csv" :
                # the sample_submission.csv file is also checked, which is not necessary
                df = pd.read_csv(f"{state.competition_dir}/{file}")
                duplicates = df.duplicated().sum()
                duplicates_rows = df[df.duplicated(keep=False)]

                if duplicates == 0:
                    return True, 5, "No duplicate rows in submission.csv"
                else:
                    return False, 5, f"There are {duplicates} duplicate rows in the submission.csv, Rows with duplicated values are: {duplicates_rows.index}"

    def test_readable_cleaned_train(self, state: State):
        path = f"{state.competition_dir}/cleaned_train.csv"
        with open(path, 'r') as file:
            if file.readable():
                return True, 6, "cleaned_train.csv is readable, please continue to the next step of the process"
            else:
                return False, 6, "cleaned_train.csv could not be read, please try to reprocess it"

    def test_readable_cleaned_test(self, state: State):
        path = f"{state.competition_dir}/cleaned_test.csv"
        with open(path, 'r') as file:
            if file.readable():
                return True, 7, "cleaned_test.csv is readable, please continue to the next step of the process"
            else:
                return False, 7, "cleaned_test.csv could not be read, please try to reprocess it"

    def test_readable_submission(self, state: State):
        files = os.listdir(state.competition_dir)
        for file in files:
            if f"submission.csv" in file:
                path = f"{state.competition_dir}/{file}"
                with open(path, 'r') as file:
                    if file.readable():
                        return True, 8, "submission.csv is readable, please continue to the next step of the process"
                    else:
                        return False, 8, "submission.csv is not readable, please try to reprocess it"

    def test_cleaned_train_no_missing_values(self, state: State):
        path = f"{state.competition_dir}/cleaned_train.csv"
        df = pd.read_csv(path)
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0]
        
        if missing_columns.empty:
            return True, 9, "The cleaned_train.csv file has no missing values, please continue to the next step of the process"
        else:
            missing_details = []
            for col, count in missing_columns.items():
                percentage = (count / len(df)) * 100
                missing_details.append(f"{col}: {count} ({percentage:.2f}%)")
            
            return False, 9, f"There are missing values in the cleaned_train.csv file. Detailed missing value information:\n" + "\n".join(missing_details) + "\nDo NOT fill the missing values with another NaN-type value, such as 'None', 'NaN', or 'nan'."

    def test_cleaned_test_no_missing_values(self, state: State):
        path = f"{state.competition_dir}/cleaned_test.csv"
        df = pd.read_csv(path)
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0]
        
        if missing_columns.empty:
            return True, 10, "The cleaned_test.csv file has no missing values, please continue to the next step of the process"
        else:
            missing_details = []
            for col, count in missing_columns.items():
                percentage = (count / len(df)) * 100
                missing_details.append(f"{col}: {count} ({percentage:.2f}%)")
            
            return False, 10, f"There are missing values in the cleaned_test.csv file. Detailed missing value information:\n" + "\n".join(missing_details) + "\nNOTE that apply the same methods as applied in cleaned_train.csv to deal with missing values."

    def test_cleaned_train_no_duplicated_features(self, state: State):
        # but I don't think this is a good test, because the "df = pd.read_csv(path)" directly deletes the duplicated columns
        path = f"{state.competition_dir}/cleaned_train.csv"
        df = pd.read_csv(path)
        duplicated_features = df.columns[df.columns.duplicated()]
        if duplicated_features.empty:
            return True, 11, "The cleaned_train.csv file has no repeated features, please continue to the next step of the process"
        else:
            return False, 11, f"The cleaned_train.csv file has repeated features: {', '.join(duplicated_features)}"
        
    def test_cleaned_test_no_duplicated_features(self, state: State):
        # but I don't think this is a good test, because the "df = pd.read_csv(path)" directly deletes the duplicated columns
        path = f"{state.competition_dir}/cleaned_test.csv"
        df = pd.read_csv(path)
        duplicated_features = df.columns[df.columns.duplicated()]
        if duplicated_features.empty:
            return True, 12, "The cleaned_test.csv file has no repeated features, please continue to the next step of the process"
        else:
            return False, 12, f"The cleaned_test.csv file has repeated features: {', '.join(duplicated_features)}"

    def test_processed_train_no_duplicated_features(self, state: State):
        # but I don't think this is a good test, because the "df = pd.read_csv(path)" directly deletes the duplicated columns
        path = f"{state.competition_dir}/processed_train.csv"
        df = pd.read_csv(path)
        duplicated_features = df.columns[df.columns.duplicated()]
        if duplicated_features.empty:
            return True, 13, "The processed_train.csv file has no repeated features, please continue to the next step of the process"
        else:
            return False, 13, f"The processed_train.csv file has repeated features: {', '.join(duplicated_features)}"
        
    def test_processed_test_no_duplicated_features(self, state: State):
        # but I don't think this is a good test, because the "df = pd.read_csv(path)" directly deletes the duplicated columns
        path = f"{state.competition_dir}/processed_test.csv"
        df = pd.read_csv(path)
        duplicated_features = df.columns[df.columns.duplicated()]
        if duplicated_features.empty:
            return True, 14, "The processed_test.csv file has no repeated features, please continue to the next step of the process"
        else:
            return False, 14, f"The processed_test.csv file has repeated features: {', '.join(duplicated_features)}"
        

    def test_processed_train_feature_number(self, state: State):
        def get_categorical_nunique_formatted(dataframe):
            categorical_columns = dataframe.select_dtypes(include=['object', 'category', 'bool']).columns
            nunique_counts = dataframe[categorical_columns].nunique().sort_values(ascending=False)
            formatted_output = "\n".join([f"{feature}    number of unique values: {count}" for feature, count in nunique_counts.items()])
            return formatted_output

        path = f"{state.competition_dir}/processed_train.csv"
        df = pd.read_csv(path)
        path_to_origin_train = f"{state.competition_dir}/cleaned_train.csv"
        df_origin = pd.read_csv(path_to_origin_train)
        result = "Valid"
        if (len(df.columns) <= 3 * len(df_origin.columns) or len(df.columns) <= 50) and result == "Valid":
            return True, 15, f"The feature engineering phase is well performed."
        else:
            false_info = "There are too many features after handling features in the feature engineering phase."
            false_info += f'''processed_train.csv is the processed data of cleaned_train.csv after feature engineering.
During the feature engineering phase, improper feature handling has resulted in an excessive number of features. 
One possible reason is that during the feature processing, certain categorical features (such as brand, transmission, etc.) have too many categories, leading to a large number of features being generated after one-hot encoding.

Here is the information about the categorical features of cleaned_train.csv and their unique value counts:
Categorical features of cleaned_train.csv and their unique value counts:
{get_categorical_nunique_formatted(df_origin)}

Here is the information about the features of cleaned_train.csv:
{df_origin.columns}

Here is the information about the features of processed_train.csv:
{df.columns}
'''
            return False, 15, false_info

    def test_processed_test_feature_number(self, state: State):
        def get_categorical_nunique_formatted(dataframe):
            categorical_columns = dataframe.select_dtypes(include=['object', 'category', 'bool']).columns
            nunique_counts = dataframe[categorical_columns].nunique().sort_values(ascending=False)
            formatted_output = "\n".join([f"{feature}    number of unique values: {count}" for feature, count in nunique_counts.items()])
            return formatted_output

        path = f"{state.competition_dir}/processed_test.csv"
        df = pd.read_csv(path)
        path_to_origin_train = f"{state.competition_dir}/cleaned_test.csv"
        df_origin = pd.read_csv(path_to_origin_train)
        result = "Valid"
        if (len(df.columns) <= 3 * len(df_origin.columns) or len(df.columns) <= 50) and result == "Valid":
            return True, 16, f"The feature engineering phase is well performed."
        else:
            false_info = "There are too many features after handling features in the feature engineering phase."
            false_info += f'''processed_test.csv is the processed data of cleaned_test.csv after feature engineering.
During the feature engineering phase, improper feature handling has resulted in an excessive number of features. 
One possible reason is that during the feature processing, certain categorical features (such as brand, transmission, etc.) have too many categories, leading to a large number of features being generated after one-hot encoding.

Here is the information about the categorical features of cleaned_test.csv and their unique value counts:
Categorical features of cleaned_test.csv and their unique value counts:
{get_categorical_nunique_formatted(df_origin)}

Here is the information about the features of cleaned_test.csv:
{df_origin.columns}

Here is the information about the features of processed_test.csv:
{df.columns}
'''
            return False, 16, false_info

    def test_cleaned_train_no_missing_target(self, state: State):
        # read train.csv and test.csv
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_cleaned_train = f"{state.competition_dir}/cleaned_train.csv"
        
        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        # check if the target column is in the cleaned_train.csv
        df = pd.read_csv(path_cleaned_train)
        if all(col in df.columns for col in target_columns):
            return True, 17, "The target columns are in the cleaned_train.csv file, please continue to the next step of the process"
        else:
            return False, 17, f"The target columns {target_columns} are not in the cleaned_train.csv file, please reprocess it"
        
    def test_cleaned_test_no_target_column(self, state: State):
        # read train.csv and test.csv
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_cleaned_test = f"{state.competition_dir}/cleaned_test.csv"
        
        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        df = pd.read_csv(path_cleaned_test)
        target_columns_in_cleaned_test = [col for col in target_columns if col in df.columns]

        if len(target_columns_in_cleaned_test) > 0:
            return False, 18, f"The target columns {target_columns_in_cleaned_test} are in the cleaned_test.csv file, please reprocess it"
        else:
            return True, 18, "The target columns are not in the cleaned_test.csv file, please continue to the next step of the process"
        
    def test_processed_train_no_missing_target(self, state: State):
        # read train.csv and test.csv
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_processed_train = f"{state.competition_dir}/processed_train.csv"
        
        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        # check if the target column is in the processed_train.csv
        df = pd.read_csv(path_processed_train)
        if all(col in df.columns for col in target_columns):
            return True, 19, "The target columns are in the processed_train.csv file, please continue to the next step of the process"
        else:
            return False, 19, f"The target columns {target_columns} are not in the processed_train.csv file, please reprocess it"

    def test_processed_test_no_target_column(self, state: State):
        # read train.csv and test.csv
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_processed_test = f"{state.competition_dir}/processed_test.csv"
        
        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        df = pd.read_csv(path_processed_test)
        target_columns_in_processed_test = [col for col in target_columns if col in df.columns]

        if len(target_columns_in_processed_test) > 0:
            return False, 20, f"The target columns {target_columns_in_processed_test} are in the processed_test.csv file, please reprocess it"
        else:
            return True, 20, "The target columns are not in the processed_test.csv file, please continue to the next step of the process"
        
    def test_cleaned_difference_train_test_columns(self, state: State):
        # test if the columns in cleaned_train.csv only has one more column than cleaned_test.csv, which is the target column
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_cleaned_train = f"{state.competition_dir}/cleaned_train.csv"
        path_cleaned_test = f"{state.competition_dir}/cleaned_test.csv"
        path_sample_submission = f"{state.competition_dir}/sample_submission.csv"

        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        df_train = pd.read_csv(path_cleaned_train)
        df_test = pd.read_csv(path_cleaned_test)
        sample_submission = pd.read_csv(path_sample_submission)
        target_length = len(sample_submission.columns) - 1

        # Find the differences in columns
        train_only_columns = set(df_train.columns) - set(df_test.columns)
        test_only_columns = set(df_test.columns) - set(df_train.columns)
        
        if len(df_train.columns) == len(df_test.columns) + target_length and all(col in df_train.columns for col in target_columns):
            return True, 21, f"The cleaned_train.csv file has {target_length} more columns than cleaned_test.csv, which are the target columns {target_columns}, please continue to the next step of the process"
        else:
            error_message = f"The cleaned_train.csv file has different columns from cleaned_test.csv, please find the difference between the two files and find out the reason. cleaned_train.csv should only have {target_length} columns than cleaned_test.csv, which are the target columns {target_columns}.\n"
            # error_message += f"Features in cleaned_train.csv: {df_train.columns}.\n"
            # error_message += f"Features in cleaned_test.csv: {df_test.columns}.\n"
            error_message += f"Columns only in cleaned_train.csv: {train_only_columns}\n"
            error_message += f"Columns only in cleaned_test.csv: {test_only_columns}"
            return False, 21, error_message
    
    def test_processed_difference_train_test_columns(self, state: State):
        path_train = f"{state.competition_dir}/train.csv"
        path_test = f"{state.competition_dir}/test.csv"
        path_processed_train = f"{state.competition_dir}/processed_train.csv"
        path_processed_test = f"{state.competition_dir}/processed_test.csv"
        path_sample_submission = f"{state.competition_dir}/sample_submission.csv"

        train_columns = pd.read_csv(path_train).columns
        test_columns = pd.read_csv(path_test).columns
        target_columns = [col for col in train_columns if col not in test_columns]

        df_train = pd.read_csv(path_processed_train)
        df_test = pd.read_csv(path_processed_test)
        sample_submission = pd.read_csv(path_sample_submission)
        target_length = len(sample_submission.columns) - 1

        # Find the differences in columns
        train_only_columns = set(df_train.columns) - set(df_test.columns)
        test_only_columns = set(df_test.columns) - set(df_train.columns)

        if len(df_train.columns) == len(df_test.columns) + target_length and all(col in df_train.columns for col in target_columns):
            return True, 22, f"The processed_train.csv file has {target_length} more columns than processed_test.csv, which are the target columns {target_columns}, please continue to the next step of the process"
        else:
            error_message = f"The processed_train.csv file has different columns from processed_test.csv, please find the difference between the two files and find out the reason. processed_train.csv should only have {target_length} more columns than processed_test.csv, which are the target columns {target_columns}.\n"
            # error_message += f"Features in processed_train.csv: {df_train.columns}.\n"
            # error_message += f"Features in processed_test.csv: {df_test.columns}.\n"
            error_message += f"Columns only in processed_train.csv: {train_only_columns}\n"
            error_message += f"Columns only in processed_test.csv: {test_only_columns}"
            return False, 22, error_message
        
    def test_submission_no_missing_values(self, state: State):
        files = os.listdir(state.competition_dir)
        for file in files:
            # submission file may have different names
            if file == "submission.csv" :
                path = f"{state.competition_dir}/{file}"
                df = pd.read_csv(path)
                missing_columns = df.columns[df.isnull().any()].tolist()
                if df.isnull().sum().sum() == 0:
                    return True, 23, "The submission.csv file has no missing values, please continue to the next step of the process"
                else:
                    return False, 23, f"There are missing values in the submission.csv file. The columns with missing values are: {', '.join(missing_columns)}"

    def test_processed_train_no_missing_values(self, state: State):
        path = f"{state.competition_dir}/processed_train.csv"
        df = pd.read_csv(path)
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0]

        if missing_columns.empty:
            return True, 24, "The processed_train.csv file has no missing values, please continue to the next step of the process"
        else:
            missing_details = []
            for col, count in missing_columns.items():
                percentage = (count / len(df)) * 100
                missing_details.append(f"{col}: {count} ({percentage:.2f}%)")
            
            return False, 24, f"There are missing values in the processed_train.csv file. Detailed missing value information:\n" + "\n".join(missing_details) + "\nDo NOT fill the missing values with another NaN-type value, such as 'None', 'NaN', or 'nan'."

    def test_processed_test_no_missing_values(self, state: State):
        path = f"{state.competition_dir}/processed_test.csv"
        df = pd.read_csv(path)
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0]

        if missing_columns.empty:
            return True, 25, "The processed_test.csv file has no missing values, please continue to the next step of the process"
        else:
            missing_details = []
            for col, count in missing_columns.items():
                percentage = (count / len(df)) * 100
                missing_details.append(f"{col}: {count} ({percentage:.2f}%)")
            
            return False, 25, f"There are missing values in the processed_test.csv file. Detailed missing value information:\n" + "\n".join(missing_details) + "\nDo NOT fill the missing values with another NaN-type value, such as 'None', 'NaN', or 'nan'."

    def test_image_num(self, state: State):
        image_count = 0
        if "Preliminary Exploratory Data Analysis" in state.phase:
            path = f"{state.competition_dir}/pre_eda/images"
        elif "In-depth Exploratory Data Analysis" in state.phase:
            path = f"{state.competition_dir}/deep_eda/images"
        else:
            return True, 24, "No need to check the number of images at this stage, please continue to the next step of the process"
            # 遍历指定目录
        for entry in os.scandir(path):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_count += 1
        if image_count > 10:
            return False, 26, f"The number of images in the path is {image_count}, greater than 10, please re-process to reduce the number of images to 10 or less."
        else:
            return True, 26, "Number of images is less than or equal to 10, unit test passed, please continue to the next step of the process"
    
    def test_file_num_submission(self, state: State):
        path = f"{state.competition_dir}/sample_submission.csv"
        df = pd.read_csv(path)
        files = os.listdir(state.competition_dir)
        for file in files:
            # submission file may have different names
            if file == "submission.csv" :
                path1 = f"{state.competition_dir}/{file}"
                df1 = pd.read_csv(path1)
                if len(df) == len(df1):
                    return True, 27, "submission.csv and sample_submission.csv files have the same number of rows, unit test passed"
                else:
                    # also report missing rows number
                    row_inx_sample = set(df.index)
                    row_inx_submission = set(df1.index)
                    missing_rows = row_inx_sample - row_inx_submission
                    return False, 27, f"submission.csv and sample_submission.csv files have different number of rows. submission.csv has {len(df1)} rows, while sample_submission.csv has {len(df)} rows. Missing rows are: {missing_rows}."
    
    def test_column_names_submission(self, state: State):
        path = f"{state.competition_dir}/sample_submission.csv"
        df = pd.read_csv(path)
        files = os.listdir(state.competition_dir)
        for file in files:
            # submission file may have different names
            if file == "submission.csv" :
                path1 = f"{state.competition_dir}/{file}"
                df1 = pd.read_csv(path1)
                # 比较两个 DataFrame 的列名集合是否相同
                if list(df.columns) == list(df1.columns):
                    return True, 28, "submission.csv and sample_submission.csv files have the same column names, unit test passed"
                else:
                    return False, 28, f"submission.csv and sample_submission.csv files have different column names or different column order. submission.csv has columns: {set(df1.columns)}, while sample_submission.csv has columns: {set(df.columns)}."

    
    def test_submission_validity(self, state: State):
        # 检查submission.csv和sample_submission.csv的第一个列是否相同
        # 检查submission.csv的数值是否在sample_submission.csv的数值范围内
        # 要保证有submission.csv生成
        path_sample = f"{state.competition_dir}/sample_submission.csv"
        path_submission = f"{state.competition_dir}/submission.csv"
        
        df_sample = pd.read_csv(path_sample)
        df_submission = pd.read_csv(path_submission)
        
        # Replace the first column of submission.csv with the first column from sample_submission.csv
        first_column_name = df_sample.columns[0]
        df_submission[first_column_name] = df_sample[first_column_name]
        
        # Save the modified submission.csv
        df_submission.to_csv(path_submission, index=False)

        unique_values = df_submission.iloc[:, 1].unique()
        if set(unique_values) == {0, 1} or set(unique_values) == {False, True} or set(unique_values) == {0.0, 1.0}:
            result = "Valid"
            return True, 29, "submission.csv is valid."

        # If the data type of the second column is numeric
        if pd.api.types.is_numeric_dtype(df_sample.iloc[:, 1]) and not pd.api.types.is_bool_dtype(df_submission.iloc[:, 1]):
            # Calculate mean of first 100 values in the second column
            # Check if submission mean is within the range of 1/10 to 10 times the sample mean
            sample_mean = df_sample.iloc[:100, 1].mean()
            submission_mean = df_submission.iloc[:100, 1].mean()
            lower_bound = sample_mean / 10
            upper_bound = sample_mean * 10
            if lower_bound <= submission_mean <= upper_bound:
                result = "Valid"
                reason = "The mean of the first 100 values in the submission file is within the expected range."
            else:
                result = "Invalid"
                reason = f"The mean of the first 100 values in the submission file ({submission_mean}) is outside the expected range ({lower_bound} to {upper_bound})."
        elif pd.api.types.is_numeric_dtype(df_sample.iloc[:, 1]) != pd.api.types.is_numeric_dtype(df_submission.iloc[:, 1]):
            result = "Invalid"
            sample_dtype = df_sample.iloc[:, 1].dtype
            submission_dtype = df_submission.iloc[:, 1].dtype
            sample_values = df_sample.iloc[:10, 1].tolist()
            submission_values = df_submission.iloc[:10, 1].tolist()
            reason = f"The data types of the second column in sample_submission.csv ({sample_dtype}) and submission.csv ({submission_dtype}) do not match."
            reason += f"\n\nFirst 10 values in sample_submission.csv ({sample_dtype}):\n{sample_values}"
            reason += f"\n\nFirst 10 values in submission.csv ({submission_dtype}):\n{submission_values}"
        else: 
            result = "Valid"
        # compare the first column values of two DataFrames
        if result == "Valid":
            return True, 29, "submission.csv is valid."
        else:
            false_info = f"submission.csv is not valid. {reason}"
            false_info += f'''
This is the first 10 lines of sample_submission.csv:
{df_sample.head(10)}
This is the first 10 lines of submission.csv:
{df_submission.head(10)}
For Id-type column, submission.csv should have exactly the same values as sample_submission.csv. I suggest you load Id-type column directly from `{state.competition_dir}/test.csv`.
If you use some transformation on the features in submission.csv, please make sure you have reversed the transformation before submitting the file.
Here is an example that specific transformation applied on features (ID, SalePrice) in submisson.csv is **not reversed**, which is wrong:
<example>
- submission.csv:
Id,SalePrice
1.733237550296372,-0.7385090666351347
1.7356102231920547,-0.2723912737214865
...
- sample_submission.csv:
Id,SalePrice
1461,169277.0524984
1462,187758.393988768
</example>
'''
            return False, 29, false_info
    
    
    def test_file_size(self, state: State):
        max_size_mb = 100
        path = f"{state.competition_dir}/train.csv"
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        path = f"{state.competition_dir}/test.csv"
        file_size_mb += os.path.getsize(path) / (1024 * 1024)
        path = f"{state.competition_dir}/sample_submission.csv"
        file_size_mb += os.path.getsize(path) / (1024 * 1024)
        if file_size_mb < max_size_mb:
            return True, 30, "File size less than 100M, unit test passed"
        else:
            return False, 30, f"The three files are too large, the maximum allowed size is {max_size_mb}MB, the current size is {file_size_mb:.2f}MB."

    def test_cleaned_train_id_column(self, state: State):
        path = f"{state.competition_dir}/cleaned_train.csv"
        df = pd.read_csv(path)
        
        # Check if any column name (case-insensitive) matches 'id'
        # here can add an id columns set later and use any() function to check
        id_columns = [col for col in df.columns if (col.lower() == 'id' or col.lower() == 'passengerid')]
        
        if id_columns:
            return True, 31, f"The cleaned_train.csv file contains an ID column: {id_columns[0]}"
        else:
            return False, 31, f"The cleaned_train.csv file does not contain an ID column. The columns in cleaned_train.csv are {df.columns}. Please ensure that the ID column is preserved during the cleaning process."
    
    def test_cleaned_test_id_column(self, state: State):
        path = f"{state.competition_dir}/cleaned_test.csv"
        df = pd.read_csv(path)
        
        # Check if any column name (case-insensitive) matches 'id'
        id_columns = [col for col in df.columns if (col.lower() == 'id' or col.lower() == 'passengerid')]
        
        if id_columns:
            return True, 32, f"The cleaned_test.csv file contains an ID column: {id_columns[0]}"
        else:
            return False, 32, f"The cleaned_test.csv file does not contain an ID column. The columns in cleaned_test.csv are {df.columns}. Please ensure that the ID column is preserved during the cleaning process."

    def test_processed_train_id_column(self, state: State):
        path = f"{state.competition_dir}/processed_train.csv"
        df = pd.read_csv(path)
        
        # Check if any column name (case-insensitive) matches 'id'
        id_columns = [col for col in df.columns if (col.lower() == 'id' or col.lower() == 'passengerid')]
        
        if id_columns:
            return True, 33, f"The processed_train.csv file contains an ID column: {id_columns[0]}"
        else:
            return False, 33, f"The processed_train.csv file does not contain an ID column. The columns in processed_train.csv are {df.columns}. Please ensure that the ID column is preserved during the cleaning process."

    def test_processed_test_id_column(self, state: State):
        path = f"{state.competition_dir}/processed_test.csv"
        df = pd.read_csv(path)
        
        # Check if any column name (case-insensitive) matches 'id'
        id_columns = [col for col in df.columns if (col.lower() == 'id' or col.lower() == 'passengerid')]
        
        if id_columns:
            return True, 34, f"The processed_test.csv file contains an ID column: {id_columns[0]}"
        else:
            return False, 34, f"The processed_test.csv file does not contain an ID column. The columns in processed_test.csv are {df.columns}. Please ensure that the ID column is preserved during the cleaning process."
    
    def test_cleaned_train_no_missing_rows(self, state: State):
        original_path = f"{state.competition_dir}/train.csv"
        cleaned_path = f"{state.competition_dir}/cleaned_train.csv"
        
        original_df = pd.read_csv(original_path)
        cleaned_df = pd.read_csv(cleaned_path)
        
        if len(original_df) == len(cleaned_df):
            return True, 35, f"The cleaned_train.csv file has the correct number of rows: {len(cleaned_df)}"
        else:
            return False, 35, f"The cleaned_train.csv file has {len(cleaned_df)} rows, but the original train.csv has {len(original_df)} rows. Please check your data cleaning process for any unintended row removals."

    def test_cleaned_test_no_missing_rows(self, state: State):
        original_path = f"{state.competition_dir}/test.csv"
        cleaned_path = f"{state.competition_dir}/cleaned_test.csv"
        
        original_df = pd.read_csv(original_path)
        cleaned_df = pd.read_csv(cleaned_path)
        
        if len(original_df) == len(cleaned_df):
            return True, 36, f"The cleaned_test.csv file has the correct number of rows: {len(cleaned_df)}"
        else:
            return False, 36, f"The cleaned_test.csv file has {len(cleaned_df)} rows, but the original test.csv has {len(original_df)} rows. Please check your data cleaning process for any unintended row removals."

    def test_processed_train_no_missing_rows(self, state: State):
        original_path = f"{state.competition_dir}/train.csv"
        processed_path = f"{state.competition_dir}/processed_train.csv"
        
        original_df = pd.read_csv(original_path)
        processed_df = pd.read_csv(processed_path)
        
        if len(original_df) == len(processed_df):
            return True, 37, f"The processed_train.csv file has the correct number of rows: {len(processed_df)}"
        else:
            return False, 37, f"The processed_train.csv file has {len(processed_df)} rows, but the original train.csv has {len(original_df)} rows. Please check your feature engineering process for any unintended row removals or additions."

    def test_processed_test_no_missing_rows(self, state: State):
        original_path = f"{state.competition_dir}/test.csv"
        processed_path = f"{state.competition_dir}/processed_test.csv"
        
        original_df = pd.read_csv(original_path)
        processed_df = pd.read_csv(processed_path)
        
        if len(original_df) == len(processed_df):
            return True, 38, f"The processed_test.csv file has the correct number of rows: {len(processed_df)}"
        else:
            return False, 38, f"The processed_test.csv file has {len(processed_df)} rows, but the original test.csv has {len(original_df)} rows. Please check your feature engineering process for any unintended row removals or additions."

    def test_submission_no_missing_rows(self, state: State):
        sample_path = f"{state.competition_dir}/sample_submission.csv"
        submission_path = f"{state.competition_dir}/submission.csv"
        
        sample_df = pd.read_csv(sample_path)
        submission_df = pd.read_csv(submission_path)
        
        if len(sample_df) == len(submission_df):
            return True, 39, f"The submission.csv file has the correct number of rows: {len(submission_df)}"
        else:
            return False, 39, f"The submission.csv file has {len(submission_df)} rows, but the sample_submission.csv has {len(sample_df)} rows. Please ensure that your submission file includes predictions for all test samples."

if __name__ == '__main__':
    test_tool = TestTool(memory=None, model='gpt-4o', type='api')
    test_tool._execute_tests()

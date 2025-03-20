PROMPT_READER_TASK = '''
Please conduct a comprehensive analysis of the competition, focusing on the following aspects:
1. Competition Overview: Understand the background and context of the topic.
2. Files: Analyze each provided file, detailing its purpose and how it should be used in the competition.
3. Problem Definition: Clarify the problem's definition and requirements.
4. Data Information: Gather detailed information about the data, including its structure and contents.
    4.1 Data type:
        4.1.1. ID type: features that are unique identifiers for each data point, which will NOT be used in the model training.
        4.1.2. Numerical type: features that are numerical values.
        4.1.3. Categorical type: features that are categorical values.
        4.1.4 Datetime type: features that are datetime values.
    4.2 Detailed data description
5. Target Variable: Identify the target variable that needs to be predicted or optimized, which is provided in the training set but not in the test set.
6. Evaluation Metrics: Determine the evaluation metrics that will be used to assess the submissions.
7. Submission Format: Understand the required format for the final submission.
8. Other Key Aspects: Highlight any other important aspects that could influence the approach to the competition.
Ensure that the analysis is thorough, with a strong emphasis on :
1. Understanding the purpose and usage of each file provided.
2. Figuring out the target variable and evaluation metrics.
3. Classification of the features.
'''


PROMPT_READER = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: Background Understand.

#############
# TASK #
{task} 

#############
# RESPONSE #
Let's work this out in a step by step way.

#############
# START ANALYSIS #
If you understand, please request the overview of this data science competition, and data preview from me.
'''

PROMPT_READER_ROUND2 = '''
# TASK #
Please extract essential information from your answer and reorganize into a specified MARKDOWN format. 
You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. 
You must ensure that the essential information is complete and accurate.

#############
# RESPONSE: MARKDOWN FORMAT #
Here is the MARKDOWN format you should follow:
```markdown
# Competition Information
## 1. Competition Overview
[Brief description of the competition]

## 2. Files
[List and description of provided files]

## 3. Problem Definition
[Clear statement of the problem to be solved]

## 4. Data Information
### 4.1 Data type
#### 4.1.1 ID type
[List of ID features]
#### 4.1.2 Numerical type
[List of numerical features]
#### 4.1.3 Categorical type
[List of categorical features]
#### 4.1.4 Datetime type
[List of datetime features]

### 4.2 Detailed data description
[Detailed description of data]

## 5. Target Variable
[Description of the target variable]

## 6. Evaluation Metrics
[Explanation of evaluation criteria]

## 7. Submission Format
[Details on required submission format]

## 8. Other Key Aspects
[Additional important information]
```

#############
# START REORGANIZING #
'''


PROMPT_READER_WITH_EXPERIENCE_ROUND0 = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: Background Understand.

#############
# TASK #
{task} In the past, you have attempted this task multiple times. However, due to errors in your answers or insufficient quality, you have not succeeded. I will provide you with your previous attempts' experiences and a professional reviewer's suggestions for improvement (PREVIOUS EXPERIENCE WITH SUGGESTION). Based on these, please formulate a new, concise high-level plan to mitigate similar failures and successfully complete the task.
You must follow these subtasks:
1. Analyze the previous experience and suggestions. Think about what went wrong and how you can improve.
2. Develop a new solution based on the previous experience and suggestions.

#############
# PREVIOUS EXPERIENCE WITH SUGGESTION #
{experience_with_suggestion}

#############
# RESPONSE #
Subtask 1: Analyze the previous experience and suggestions. Think about what went wrong and how you can improve.
Let's work **Subtask 1** out in a step by step way.

#############
# START ANALYSIS #
If you understand, please request the Overview of this data science competition from me.
'''

PROMPT_READER_WITH_EXPERIENCE_ROUND2 = '''
# RESPONSE: MARKDOWN FORMAT #
Subtask2: Develop a new solution based on the previous experience and suggestions.
Here is the MARKDOWN format you should follow:
```markdown
# Competition Information
## 1. Competition Overview
[Brief description of the competition]

## 2. Files
[List and description of provided files]

## 3. Problem Definition
[Clear statement of the problem to be solved]

## 4. Data Information
### 4.1 Data type
#### 4.1.1 ID type
[List of ID features]
#### 4.1.2 Numerical type
[List of numerical features]
#### 4.1.3 Categorical type
[List of categorical features]
#### 4.1.4 Datetime type
[List of datetime features]

### 4.2 Detailed data description
[Detailed description of data]

## 5. Target Variable
[Description of the target variable]

## 6. Evaluation Metrics
[Explanation of evaluation criteria]

## 7. Submission Format
[Details on required submission format]

## 8. Other Key Aspects
[Additional important information]
```
'''
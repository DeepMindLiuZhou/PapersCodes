PREFIX_IN_CODE_FILE = '''import sys
import os
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents', 'multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.ml_tools import *

def generated_code_function():
    import numpy as np
    import pandas as pd'''


PROMPT_DEVELOPER_TASK = '''
Develop an efficient solution based on the Planner's provided plan:
1. Implement specific tasks and methods outlined in the plan
2. Ensure code is clear, concise, and well-documented
3. Utilize available tools by calling them with correct parameters
4. Consider data types, project requirements, and resource constraints
5. Write code that is easily understandable by others

Remember to balance efficiency with readability and maintainability.
'''


PROMPT_AVAILABLE_TOOLS = '''
# AVAILABLE TOOLS #
## TOOL LIST ##
You have access to the following tools:
{tool_names}
## USAGE INSTRUCTIONS ##
1. These tools are pre-defined and pre-imported in the system. You do NOT need import them or implement them again.
2. Use these tools by calling them with the correct parameters.
3. Example: To drop specific columns, use the `remove_columns_with_missing_data` tool with appropriate parameters.
## ADDITIONAL RESOURCES ##
You can also use functions from public libraries such as:
- Pandas
- NumPy
- Scikit-learn
- etc.
## DETAILED TOOL DESCRIPTIONS ##
{tools}
'''


PROMPT_DEVELOPER_CONSTRAINTS = '''
# CONSTRAINTS #

## DATA HANDLING ##
1. Data Loading:
   - Always load data files from the `{competition_path}/` directory.
   - Use specific files for each phase:
     - Data Cleaning: `train.csv` and `test.csv`
     - Feature Engineering: `cleaned_train.csv` and `cleaned_test.csv`
     - Model Building, Validation, and Prediction: `processed_train.csv` and `processed_test.csv`

2. Data Saving:
   - Save image files in the `{restore_path}/images/` directory.
   - Save data files in the `{competition_path}/` directory.
   - Use clear, meaningful names for image files that reflect their content.
   - Do NOT use special characters like `/` or spaces in file names.
   - Save specific files for each phase:
     - Data Cleaning: `cleaned_train.csv` and `cleaned_test.csv`
     - Feature Engineering: `processed_train.csv` and `processed_test.csv`

3. Data Processing:
   - Always work on a copy of the DataFrame, not the original.
   - Ensure correct data types for all columns before any operations.
   - Apply consistent modifications (e.g., scaling, encoding) to both training and test sets.
   - Take care with target-dependent operations on the test set, which lacks the target variable.
   - Do NOT modify Id-type columns.

## CODING PRACTICES ##
1. General Rules:
   - Use `print()` for outputting values.
   - Avoid writing `assert` statements.

2. Visualization:
   - Use `plt.close()` after saving each image.
   - Limit EDA visualizations to 10 or fewer, focusing on the most insightful.
   - Optimize for large datasets (e.g., `annot=False` in seaborn heatmaps).

3. Efficiency:
   - Prioritize runtime efficiency, especially for:
     - Data visualization
     - Large dataset handling
     - Complex algorithms

## EXAMPLES ##
- Before calculating a correlation matrix, ensure all data is numerical. Handle non-numerical data appropriately.
- Verify consistent data types across columns before merging or joining operations.

Remember: Always consider resource constraints and prioritize efficiency in your code.
'''


PROMPT_DEVELOPER = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: {phase_name}.

#############
# INFORMATION #
{background_info}

{state_info}

#############
# PLAN #
{plan}

#############
# TASK #
{task}

#############
# RESPONSE: BLOCK (CODE & EXPLANATION) #
TASK 1:
THOUGHT PROCESS:
[Explain your approach and reasoning]
CODE:
```python
[code]
```
EXPLANATION:
[Brief explanation of the code and its purpose]

TASK 2:
[Repeat the above structure for each task/subtask]

...

#############
# START CODING #
Before you begin, please request the following information from me:
1. Code from previous phases
2. All features of the data
3. Available tools

Once you have this information, provide your complete response with code and explanations for all tasks in a single message.
'''


PROMPT_DEVELOPER_WITH_EXPERIENCE_ROUND0_0 = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase:{phase_name}.

#############
# INFORMATION #
{background_info}

{state_info}

#############
# PLAN #
{plan}

#############
# TASK #
{task} 
You have attempted this task multiple times but have not succeeded due to errors or insufficient quality in your answers.

I will provide you with:
1. Your previous attempts' experiences
2. A professional reviewer's suggestions for improvement

This information will be in the "PREVIOUS EXPERIENCE WITH SUGGESTION" section below.

You must complete the following subtasks:
1. Analyze the previous experience and suggestions
   - Think about what went wrong
   - Consider how you can improve
2. Develop a new solution based on your analysis

#############
# PREVIOUS EXPERIENCE WITH SUGGESTION #
{experience_with_suggestion}

#############
# RESPONSE #
Focus ONLY on Subtask 1: Analyze the previous experience and suggestions
Approach Subtask 1 step by step in your response

#############
# START ANALYSIS #
Before you begin, please request the following information from me:
1. Code from previous phases
2. All features of the data
3. Available tools

Once you have this information, begin your analysis of the previous experience and suggestions.
'''

PROMPT_DEVELOPER_WITH_EXPERIENCE_ROUND0_2 = '''
# RESPONSE: BLOCK (CODE & EXPLANATION) #
Subtask 2: Develop a new solution based on the previous experience and suggestions.
TASK 1:
THOUGHT PROCESS:
[Explain your approach and reasoning]
CODE:
```python
[code]
```
EXPLANATION:
[Brief explanation of the code and its purpose]

TASK 2:
[Repeat the above structure for each task/subtask]

...

#############
# START CODING #
'''




PROMPT_DEVELOPER_DEBUG_LOCATE = '''
# CONTEXT #
I'm getting an error executing the code you generated.

#############
# TASK #
Locate and identify the most relevant code snippet causing the error (5 to 10 lines in length).

## Input Provided ##
1. Previous code
2. Code containing the error
3. Error messages
4. Output messages of the code
5. Tools used in this phase (with detailed descriptions)

## Instructions ##
1. Analyze the provided information to identify the error source.
2. Focus on the code that causes the error, not just error reporting statements.
3. If an assert statement or similar construct merely reports an error:
   - Identify the underlying code causing the assert to fail.
   - Only output the assert statement if you believe it's redundant or incorrect.
   - Apply this rule to raise statements or other error-reporting constructs that don't perform calculations, create graphs, or modify data.
4. Do NOT attempt to correct the error in this step.

Note: Ensure the final code snippet in your response is the most relevant error-causing code.

#############
# PREVIOUS CODE #
{previous_code}

#############
# CODE CONTAINS ERROR #
{wrong_code}

#############
# ERROR MESSAGES #
{error_messages}

#############
# OUTPUT MESSAGES #
{output_messages}

#############
# RESPONSE: MOST RELEVANT CODE SNIPPET CAUSES ERROR #
Let's work this out in a step by step way. 

#############
# START LOCATION ERROR #
Please request the following information from me:
1. Tool Descriptions
After you get the tool descriptions, you can start to locate the error.
'''

PROMPT_DEVELOPER_DEBUG_ASK_FOR_HELP = '''
# IMPORTANT NOTE #
This is your {i}-th attempt to fix the error. Remember, you can ONLY try 4 times in total.

Please carefully review all error messages collected from your previous attempts.

Criteria for judgment:
1. If the error messages from the last two attempts are identical, or
2. If more than two out of the last three error messages contain the same keywords or error types

Then, this indicates you are not making progress and should request help to avoid wasting time and resources.

Requesting help:
If the above criteria are met, please output the following message:
<MESSAGE> <HELP> I need help. </HELP> </MESSAGE>

If you don't need help, DO NOT output the above message.

#############
# ALL ERROR MESSAGES #
{all_error_messages}

#############
# RESPONSE #
Please respond according to the instructions above.
'''

PROMPT_DEVELOPER_DEBUG_FIX = '''
# CONTEXT #
I have an error code snippet with error messages. 

#############
# TASK #
Please correct the error code snippet according to the error messages, output messages of code and tools' descriptions. You must follow these steps:
1. Analyze why the error code snippet causes the error according to the error messages and output messages of code.
2. Think about how to correct the error code snippet.
3. Correct the error code snippet.
NOTE that if the error occurs when trying to import the provided tool, remember you do NOT import tool, they are pre-defined and pre-imported in the system.
NOTE that the **last** code snippet in your response should be the **code snippet after correction** that I ask you to output.

#############
# ERROR CODE SNIPPET #
{most_relevant_code_snippet}

#############
# ERROR MESSAGES #
{error_messages}

#############
# OUTPUT MESSAGES #
{output_messages}

#############
# TOOL DESCRIPTIONS #
{tools}

#############
# RESPONSE: CODE SNIPPET AFTER CORRECTION #
Let's work this out in a step by step way. (Output all steps in ONE response.)
'''

PROMPT_DEVELOPER_DEBUG_MERGE = '''
# CONTEXT #
When running the code you generated, I encountered some errors. I have analyzed and located the erroneous code snippet and have corrected it to produce the correct code snippet.

#############
# TASK #
- CODE CONTAINS ERROR: The original code you generated contains an error.
- ERROR CODE SNIPPET: The code snippet from your original code that causes the error, as identified through analysis.
- CODE SNIPPET AFTER CORRECTION: The correct code snippet obtained after fixing the ERROR CODE SNIPPET.
Please replace the ERROR CODE SNIPPET in CODE CONTAINS ERROR with the CODE SNIPPET AFTER CORRECTION to produce the fully corrected code.

#############
# CODE CONTAINS ERROR #
{wrong_code}

#############
# ERROR CODE SNIPPET #
{most_relevant_code_snippet}

#############
# CODE SNIPPET AFTER CORRECTION #
{code_snippet_after_correction}

#############
# RESPONSE: ALL CORRECT CODE #
'''





PROMPT_DEVELOPER_TEST_LOCATE = '''
# CONTEXT #
Your code has some tests that don't pass.

#############
# TASK #
For EACH test that does not pass, please analyze the code with problem, figure out which code snippet causes the test not pass, and output the problematic code snippet (5 to 10 lines in length). 
I will provide you with the previous code, code with problem, output messages of code and not pass tests' information.
NOTE that in your each analysis for each test, the **last** code snippet in your response should be the **problematic code snippet** that I ask you to output.

#############
# PREVIOUS CODE #
{previous_code}

#############
# CODE WITH PROBLEM #
{code_with_problem}

#############
# OUTPUT MESSAGES #
{output_messages}

#############
# NOT PASS TEST CASES #
{not_pass_information}

#############
# RESPONSE #
Let's work this out in a step by step way.
'''

PROMPT_DEVELOPER_TEST_REORGANIZE_LOCATE_ANSWER = '''
# TASK #
Please reorganize the code snippets that you have identified as problematic in the previous step. 

#############
# RESPONSE: CODE SNIPPETS WITH PROBLEM #
You should ONLY output the each code snippet with problem, without any other content.
Here is the template you can use:
## CODE SNIPPET 1 WITH PROBLEM ##
```python
[code snippet 1 with problem]
```
## CODE SNIPPET 2 WITH PROBLEM ##
```python
[code snippet 2 with problem]
```
...
'''

PROMPT_DEVELOPER_TEST_FIX = '''
# CONTEXT #
Your code has a couple tests that don't pass.

#############
# TASK #
Please correct the some code snippets with problem according to output messages of code and the not pass tests' information.
You must follow these steps:
1. Analyze why the code snippets with problem cause the test not pass according to output messages of code and the not pass tests' information.
2. Think about how to correct the code snippets with problem.
3. Correct the code snippets with problem.

#############
# CODE SNIPPETS WITH PROBLEM #
{code_snippets_with_problem}

#############
# OUTPUT MESSAGES #
{output_messages}

#############
# NOT PASS TEST CASES #
{not_pass_information}

#############
# RESPONSE #
Let's work this out in a step by step way. (Output all steps in ONE response.)
'''

PROMPT_DEVELOPER_TEST_REORGANIZE_FIX_ANSWER = '''
# TASK #
Please reorganize the code snippets that you have corrected in the previous step.

#############
# RESPONSE: CODE SNIPPETS AFTER CORRECTION #
You should ONLY output the each code snippet after correction, without any other content.
NOTE that you should output the code snippets after correction in the order of the code snippets with problems, they have to correspond to each other.
Here is the template you can use:
## CODE SNIPPET 1 AFTER CORRECTION ##
```python
[code snippet 1 after correction]
```
## CODE SNIPPET 2 AFTER CORRECTION ##
```python
[code snippet 2 after correction]
```
...
'''

PROMPT_DEVELOPER_TEST_MERGE = '''
# CONTEXT #
Your code has a couple tests that don't pass. I have analyzed and located the code snippets with problem and have corrected them to produce the correct code snippets.

#############
# TASK #
- CODE WITH PROBLEM: The original code you generated which failed some tests.
- CODE SNIPPETS WITH PROBLEM: Precise code snippets from your original code that causes problem, as identified through analysis.
- CODE SNIPPETS AFTER CORRECTION: The correct code snippets obtained after fixing the CODE SNIPPETS WITH PROBLEM.
Please replace the CODE SNIPPETS WITH PROBLEM in CODE WITH PROBLEM with the CODE SNIPPETS AFTER CORRECTION to produce the fully corrected code.

#############
# CODE WITH PROBLEM #
{code_with_problem}

#############
# CODE SNIPPETS WITH PROBLEM #
{code_snippets_with_problem}

#############
# CODE SNIPPETS AFTER CORRECTION #
{code_snippets_after_correction}

#############
# RESPONSE: ALL CORRECT CODE #
```python
[all_correct_code]
```
'''
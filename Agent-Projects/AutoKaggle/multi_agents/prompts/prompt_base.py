AGENT_ROLE_TEMPLATE = '''You are an excellent {agent_role}.\n'''

PHASES_IN_CONTEXT_PREFIX = '''
I am working on a data science competition called "# {competition_name}". 
I plan to divide the task into the following phases and complete them in order:
'''

PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION = '''
## EXPERIENCE {index}##
<EXPERIENCE>
{experience}
</EXPERIENCE>
<SUGGESTION>
{suggestion}
</SUGGESTION>
<SCORE>
{score}
</SCORE>
'''

PROMPT_FEATURE_INFO = '''# FEATURE INFO
## TARGET VARIABLE
{target_variable}
## FEATURES BEFORE THIS PHASE
{features_before}
## FEATURES AFTER THIS PHASE
{features_after}
'''

PROMPT_EXTRACT_TOOLS = '''
Please extract the all tools involved in the following document.
{document}
All available tools are as follows:
{all_tool_names}

Each tool name MUST be in the available tool names.
Your response should be in the following format:
```json
{{
    "tool_names": [
        "<tool_name 1>",
        "<tool_name 2>",
        "<tool_name 3>",
        ...
    ]
}}
```
'''


PROMPT_REORGANIZE_EXTRACT_TOOLS = '''
# TASK #
Try to reorganize the following information into a JSON format.

# INFORMATION #
{information}

# RESPONSE: JSON FORMAT #
```json
{{
    "tool_names": [
        "<tool_name 1>",
        "<tool_name 2>",
        "<tool_name 3>",
        ...
    ]
}}
```
'''

PROMPT_REORGANIZE_JSON = '''
# TASK #
Please reorganize the following information into a JSON format.
{information}

# RESPONSE: JSON FORMAT #
```json
[json_format]
```
'''


PROMPT_DATA_PREVIEW = '''
# TASK #
Please carefully review the following data and provide a summary of its basic information. Use the specified MARKDOWN format for your summary.
Instructions:
1. Analyze the provided data thoroughly.
2. Summarize the key information.
3. Format your response using the MARKDOWN template below.

#############
# DATA #
{data}

#############
# RESPONSE: MARKDOWN FORMAT #
```markdown
# Data Information
## Data Type
### ID type
[List features that are unique identifiers for each data point, which will NOT be used in model training.]

### Numerical type
[List features that are numerical values.]

### Categorical type
[List features that are categorical values.]

### Datetime type
[List features that are datetime values.]

## Detailed data description
[Provide a comprehensive description of the data, including any notable patterns, distributions, or characteristics.]

## Target Variable
[Provide the target variable and its description.]

# Submission format (if applicable)
[Provide the format of the submission file, including the required columns and their types.]
```

#############
# START ANALYSIS #
Let's work out this task in a step by step way.
'''
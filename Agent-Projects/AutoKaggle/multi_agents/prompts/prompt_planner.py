PROMPT_PLANNER_TASK = '''
Please design plan that is clear and specific to each FEATURE for the current development phase: {phase_name}. 
The developer will execute tasks based on your plan. 
I will provide you with INFORMATION, RESOURCE CONSTRAINTS, and previous reports and plans.
You can use the following reasoning pattern to design the plan:
1. Break down the task into smaller steps.
2. For each step, ask yourself and answer:
    - "What is the objective of this step?"
    - "What are the essential actions to achieve the objective?"
    - "What features are involved in each action?"
    - "Which tool can be used for each action? What are the parameters of the tool?"
    - "What are the expected output of each action? What is the impact of the action on the data?"
    - "What are the constraints of this step?"
'''


PROMPT_PLANNER_TOOLS = '''
# AVAILABLE TOOLS #
## TOOL LIST ##
You have access to the following tools:
{tool_names}
## ADDITIONAL RESOURCES ##
You can also use functions from public libraries such as:
- Pandas
- NumPy
- Scikit-learn
- etc.
## DETAILED TOOL DESCRIPTIONS ##
{tools}
'''

PROMPT_PLANNER = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: {phase_name}.

#############
# INFORMATION #
{background_info}

{state_info}

#############
# NOTE #
## PLANNING GUIDELINES ##
1. Limit the plan to a MAXIMUM of FOUR tasks.
2. Provide clear methods and constraints for each task.
3. Focus on critical steps specific to the current phase.
4. Prioritize methods and values mentioned in USER RULES.
5. Offer detailed plans without writing actual code.
6. ONLY focus on tasks relevant to this phase, avoiding those belonging to other phases. For example, during the in-depth EDA phase:
    - you CAN perform detailed univariate analysis on KEY features.
    - you CAN NOT modify any feature or modify data.

## DATA OUTPUT PREFERENCES ##
1. Prioritize TEXT format (print) for statistical information.
2. Print a description before outputting statistics.
3. Generate images only if text description is inadequate.

## METHODOLOGY REQUIREMENTS ##
1. Provide highly detailed methods, especially for data cleaning.
2. Specify actions for each feature without omissions.

## RESOURCE MANAGEMENT ##
1. Consider runtime and efficiency, particularly for:
   - Data visualization
   - Large dataset handling
   - Complex algorithms
2. Limit generated images to a MAXIMUM of 10 for EDA.
3. Focus on critical visualizations with valuable insights.

## OPTIMIZATION EXAMPLE ##
When using seaborn or matplotlib for large datasets:
- Turn off unnecessary details (e.g., annot=False in heatmaps)
- Prioritize efficiency in plot generation

#############
# TASK #
{task}

#############
# RESPONSE #
Let's work this out in a step by step way.

#############
# START PLANNING #
Before you begin, please request the following documents from me, which contain important information that will guide your planning:
1. Report and plan from the previous phase
2. Available tools in this phase
3. Sample data for analysis
'''


PROMPT_PLNNAER_REORGANIZE_IN_MARKDOWN = '''
# TASK #
Please extract essential information from your answer and reorganize into a specified MARKDOWN format. 
You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. 
You must ensure that the essential information is complete and accurate.

#############
# RESPONSE: MARKDOWN FORMAT #
Here is the MARKDOWN format you should follow:
```markdown
## PLAN
### STEP 1
Task: [The specific task to be performed]
Tools, involved features and correct parameters: [The tools, involved features and correct parameters to be used]
Expected output or Impact on data: [The expected output of the action or the impact of the action on the data]
Constraints: [Any constraints or considerations to keep in mind]

### STEP 2
Task: [The specific task to be performed]
Tools, involved features and correct parameters: [The tools, involved features and correct parameters to be used]
Expected output or Impact on data: [The expected output of the action or the impact of the action on the data]
Constraints: [Any constraints or considerations to keep in mind]

...
```

#############
# START REORGANIZING #
'''


PROMPT_PLNNAER_REORGANIZE_IN_JSON = '''
# TASK #
Please extract essential information from your answer and reorganize into a specified JSON format. 
You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. 
You must ensure that the essential information is complete and accurate.

#############
# RESPONSE: JSON FORMAT #
Here is the JSON format you should follow:
```json
{{
    "final_answer": list=[
        {{
            "task": str="The specific task to be performed",
            "tools, involved features and correct parameters": list=["The tools, involved features and correct parameters to be used"],
            "expected output or impact on data": list=["The expected output of the action or the impact of the action on the data"],
            "constraints": list=["Any constraints or considerations to keep in mind"]
        }}
    ]
}}
```

#############
# START REORGANIZING #
'''
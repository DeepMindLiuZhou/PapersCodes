PROMPT_REVIEWER_ROUND0 = '''
# CONTEXT #
{phases_in_context}
Each phase involves collaboration between multiple agents. You are currently evaluating the performance of agents in Phase: {phase_name}.

#############
# TASK #
Your task is to assess the performance of several agents in completing Phase: {phase_name}. 
I will provide descriptions of each agent, the tasks they performed, and the outcomes of those tasks. 
Please assign a score from 1 to 5 for each agent, with 1 indicating very poor performance and 5 indicating excellent performance. 
Additionally, provide specific suggestions for improving each agent's performance, if applicable. 
If an agent's performance is satisfactory, no suggestions are necessary.

#############
# RESPONSE: JSON FORMAT #
Let's work this out in a step by step way.

#############
# START EVALUATION #
If you are ready, please request from me the role, description, input, task and execution result of the agent to be evaluated.
'''

PROMPT_REVIEWER_ROUND1_EACH_AGENT = '''
#############
# AGENT {role} TO BE EVALUATED #
<DESCRIPTION>
{description}
</DESCRIPTION>
<TASK>
{task}
</TASK>
<INPUT>
{input}
</INPUT>
<EXECUTION RESULT>
{result}
</EXECUTION RESULT>

'''

PROMPT_REVIEWER_ROUND2 = '''
# TASK #
Please extract essential information from your last answer and reorganize into a specified JSON format. You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. You must ensure that the essential information is complete and accurate.

#############
# RESPONSE: JSON FORMAT #
Here is the JSON format you should follow:
```json
{{
    "final_answer": {{
	    "final_suggestion": {{
            str="agent name": str="Specific suggestions for improving the agent's performance"
        }},
        "final_score": {{
            str="agent name": int="The final score you assign to the evaluated agent, only one score in range 1-5"
        }}
    }}
}}
```
Here is an example you can refer to:
```json
{{
    "final_suggestion": {{
        "agent developer": "1. Proactive Error Handling: Implement proactive checks for missing values before encoding to prevent issues from arising. 2. Documentation: Add more comments explaining the rationale behind specific choices (e.g., why mode imputation was chosen for ordinal features) to enhance understanding for future collaborators.",
        "agent planner": "1. Task Limitation: Consider combining related tasks or prioritizing the most impactful ones to streamline the process. 2. Resource and Time Constraints: Include a brief mention of resource and time constraints to ensure feasibility within the given context."
    }},
    "final_score": {{
        "agent developer": 4,
        "agent planner": 4
    }}
}}
```

#############
# START REORGANIZING #
'''
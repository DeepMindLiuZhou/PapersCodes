from typing import Dict, Any
import json
import re
import logging
import sys 
import os
from builtins import input as builtin_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_base import Agent
from utils import read_file, PREFIX_MULTI_AGENTS
from llm import LLM
from state import State
from prompts.prompt_base import *
from prompts.prompt_planner import *

class Planner(Agent):
    def __init__(self, model: str, type: str):  
        super().__init__(
            role="planner",
            description="You are good at planning tasks and creating roadmaps.",
            model=model,
            type=type
        )

    def _get_previous_plan_and_report(self, state: State):
        previous_plan = ""
        previous_phases = state.get_previous_phase(type="plan")
        for previous_phase in previous_phases:
            previous_dir_name = state.phase_to_directory[previous_phase]
            previous_plan += f"## {previous_phase.upper()} ##\n"
            path_to_previous_plan = f'{state.competition_dir}/{previous_dir_name}/plan.json'
            if os.path.exists(path_to_previous_plan):
                with open(path_to_previous_plan, 'r') as f:
                    previous_plan += f.read()
                    previous_plan += '\n'
            else:
                previous_plan = "There is no plan in this phase.\n"
        path_to_previous_report = f'{state.competition_dir}/{previous_dir_name}/report.txt'
        if os.path.exists(path_to_previous_report):
            previous_report = read_file(path_to_previous_report)
        else:
            previous_report = "There is no report in the previous phase.\n"
        return previous_plan, previous_report

    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        # implement the planning function
        history = []
        data_preview = self._data_preview(state, num_lines=11)
        background_info = f"Data preview:\n{data_preview}"
        state.set_background_info(background_info)
        state_info = state.get_state_info()

        if len(state.memory) == 1: # if there is no memory before, it means it is the first execution
            if self.model == 'gpt-4o':
                history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
            elif self.model == 'o1-mini':
                history.append({"role": "user", "content": f"{role_prompt}{self.description}"})
            # Round 0
            task = PROMPT_PLANNER_TASK.format(phase_name=state.phase)
            user_rules = state.generate_rules()
            input = PROMPT_PLANNER.format(phases_in_context=state.context, phase_name=state.phase, state_info=state_info, 
                                          user_rules=user_rules, background_info=background_info, task=task)
            _, history = self.llm.generate(input, history, max_completion_tokens=4096)

            # Round 1
            input = f"# PREVIOUS PLAN #\n{self._get_previous_plan_and_report(state)[0]}\n#############\n# PREVIOUS REPORT #\n{self._get_previous_plan_and_report(state)[1]}\n"
            input += self._read_data(state, num_lines=1)
            tools, tool_names = self._get_tools(state)
            if len(tool_names) > 0:
                input += PROMPT_PLANNER_TOOLS.format(tools=tools, tool_names=tool_names)
            else:
                input += "# AVAILABLE TOOLS #\nThere is no pre-defined tools in this phase. You can use the functions from public libraries such as Pandas, NumPy, Scikit-learn, etc.\n"
            raw_plan_reply, history = self.llm.generate(input, history, max_completion_tokens=4096)
            with open(f'{state.restore_dir}/raw_plan_reply.txt', 'w') as f:
                f.write(raw_plan_reply)

            # Round 2
            input = PROMPT_PLNNAER_REORGANIZE_IN_MARKDOWN
            organized_markdown_plan, history = self.llm.generate(input, history, max_completion_tokens=4096)
            markdown_plan = self._parse_markdown(organized_markdown_plan)
            with open(f'{state.restore_dir}/markdown_plan.txt', 'w') as f:
                f.write(markdown_plan)

            # Round 3
            input = PROMPT_PLNNAER_REORGANIZE_IN_JSON
            raw_json_plan, history = self.llm.generate(input, history, max_completion_tokens=4096)
            try:
                json_plan = self._parse_json(raw_json_plan)['final_answer']
            except Exception as e:
                logger.info(f"Error parsing JSON: {e}")
                json_plan = self._parse_json(raw_json_plan)
            with open(f'{state.restore_dir}/json_plan.json', 'w') as f:
                json.dump(json_plan, f, indent=4)

        else:
            last_planner_score = state.memory[-2].get("reviewer", {}).get("score", {}).get("agent planner", 0)
            if last_planner_score >= 3: # if the score of the last planner is greater than or equal to 3, it means the planner's plan is acceptable
                return {"planner": state.memory[-2]["planner"]}
            else:
                return {"planner": state.memory[-2]["planner"]}

        # save history
        with open(f'{state.restore_dir}/{self.role}_history.json', 'w') as f:
            json.dump(history, f, indent=4)

        input_used_in_review = f"   <background_info>\n{background_info}\n    </background_info>"

        print(f"State {state.phase} - Agent {self.role} finishes working.")

        with open(f'{PREFIX_MULTI_AGENTS}/config.json', 'r') as f:
            config = json.load(f)
        user_interaction = config['user_interaction']['plan']
        if user_interaction == "True":
            # user interaction here
            print("A plan has been generated and saved to 'markdown_plan.txt'.")
            print("You can now review and modify the plan if needed.")
            print("If you want to get some suggestions for modifying the plan, please type 'suggestion'.")
            user_input = builtin_input("Press Enter to continue with the current plan, or type 'edit' to modify: ")
            user_input = user_input.strip().lower()

            if user_input == 'edit':
                print(f"\nPlease edit the file: {state.restore_dir}/markdown_plan.txt")
                print("Save your changes and press Enter when you're done.")
                builtin_input("Press Enter to continue...")
                # Re-read the potentially modified plan
                with open(f'{state.restore_dir}/markdown_plan.txt', 'r') as f:
                    markdown_plan = f.read()
            elif user_input == 'suggest':
                print(f"\nPlease refer to note section for each function in the file `ml_tools_doc/{state.restore_dir}_tools.md`.")
            elif user_input:
                print("Invalid input. Continuing with the current plan.")
            else:
                print("Continuing with the current plan.")

        # save plan and result
        plan = markdown_plan
        result = markdown_plan

        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": task,
                "input": input_used_in_review,
                "plan": plan,
                "result": result
            }
        }

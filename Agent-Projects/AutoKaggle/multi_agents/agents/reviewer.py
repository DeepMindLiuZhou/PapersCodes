from typing import Dict, Any, List
import json
import re
import logging
import sys 
import os
import pdb

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
from prompts.prompt_reviewer import *

class Reviewer(Agent):
    def __init__(self, model: str, type: str):  
        super().__init__(
            role="reviewer",
            description="You are skilled at assessing the performance of one or more agents in completing a given task. Provide detailed scores for their performance and offer constructive suggestions to optimize their results.",
            model=model,
            type=type
        )

    def _merge_dicts(self, dicts: List[Dict[str, Any]], state: State) -> Dict[str, Any]:
        merged_dict = {"final_suggestion": {}, "final_score": {}}

        # define the keys to be unified
        if state.phase == 'Understand Background':
            key_mapping = {
                "reader": "agent reader"
            }
        else:
            key_mapping = {
                "planner": "agent planner",
                "developer": "agent developer"
            }
        
        try:
            for d in dicts:
                for key in d["final_suggestion"]:
                    normalized_key = key.lower()
                    for k, v in key_mapping.items():
                        if k in normalized_key:
                            normalized_key = v
                            break
                    merged_dict["final_suggestion"][normalized_key] = d["final_suggestion"][key]
                for key in d["final_score"]:
                    normalized_key = key.lower()
                    for k, v in key_mapping.items():
                        if k in normalized_key:
                            normalized_key = v
                            break
                    merged_dict["final_score"][normalized_key] = d["final_score"][key]
        except Exception as e:
            logging.error(f"Error: {e}")
            pdb.set_trace()
        
        return merged_dict

    def _generate_prompt_for_agents(self, state: State) -> List[str]:
        prompt_for_agents = []
        evaluated_agents = list(state.memory[-1].keys()) # get all agents in the previous state
        print(f"Evaluating agents: {evaluated_agents}")
        for each_agent_memory in state.memory[-1].values(): # get the current state's memory
            role = each_agent_memory["role"]
            description = each_agent_memory["description"]
            task = each_agent_memory["task"]
            input = each_agent_memory["input"]
            result = each_agent_memory["result"]
            prompt_for_agent = PROMPT_REVIEWER_ROUND1_EACH_AGENT.format(role=role.upper(), description=description, task=task, input=input, result=result)
            prompt_for_agents.append(prompt_for_agent)
        return prompt_for_agents
    
    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        # implement the evaluation function
        # the second round input: the role_description, task, input, result of each agent in the previous state
        prompt_for_agents = self._generate_prompt_for_agents(state)
        history = []
        all_raw_reply = []
        history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        round = 0
        while round <= 3 * len(prompt_for_agents) - 1:
            if round % 3 == 0:
                input = PROMPT_REVIEWER_ROUND0.format(phases_in_context=state.context, phase_name=state.phase)
            elif round % 3 == 1:
                input = prompt_for_agents[round//3 - 1]
            elif round % 3 == 2:
                input = PROMPT_REVIEWER_ROUND2
            raw_reply, history = self.llm.generate(input, history, max_completion_tokens=4096)
            if round % 3 == 2:
                all_raw_reply.append(raw_reply)
            round += 1

        all_reply = []
        # pdb.set_trace()
        for each_raw_reply in all_raw_reply:
            reply = self._parse_json(each_raw_reply)
            try:
                all_reply.append(reply['final_answer'])
            except KeyError:
                # pdb.set_trace()
                all_reply.append(reply)

        # save history
        with open(f'{state.restore_dir}/{self.role}_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        with open(f'{state.restore_dir}/{self.role}_reply.txt', 'w') as f:
            f.write("\n\n\n".join(all_raw_reply))

        review = self._merge_dicts(all_reply, state)
        final_score = review['final_score']
        final_suggestion = review['final_suggestion']
        # developer code execution failed, score is 0
        if state.memory[-1].get("developer", {}).get("status", True) == False:
            final_score["agent developer"] = 0
            review["final_suggestion"]["agent developer"] = "The code execution failed. Please check the error message and write code again."
        with open(f'{state.restore_dir}/review.json', 'w') as f:
            json.dump(review, f, indent=4)

        print(f"State {state.phase} - Agent {self.role} finishes working.")
        return {
            self.role: {
                "history": history, 
                "score": final_score, 
                "suggestion": final_suggestion, 
                "result": review
            }
        }


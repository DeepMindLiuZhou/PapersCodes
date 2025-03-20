from typing import Dict, Any
import json
import re
import logging
import sys 
import os

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
from prompts.prompt_summarizer import *
from tools import *

class Summarizer(Agent):
    def __init__(self, model: str, type: str):  
        super().__init__(
            role="summarizer",
            description="You are good at asking key questions and answer the questions from given information.",
            model=model,
            type=type
        )

    def _generate_prompt_round1(self, state: State) -> str:
        prompt_round1 = ""
        current_memory = state.memory[-1]
        for role, memory in current_memory.items():
            trajectory = json.dumps(memory.get("history", []), indent=4)
            prompt_round1 += f"\n#############\n# TRAJECTORY OF AGENT {role.upper()} #\n{trajectory}"

        return prompt_round1
    
    def _get_insight_from_visualization(self, state: State) -> str:
        images_dir = f"{state.restore_dir}/images"
        if not os.path.exists(images_dir):
            return "There is no image in this phase."
        else:
            images = os.listdir(images_dir)
        count_of_image = 0
        for image in images:
            if image.endswith('png'):
                count_of_image += 1
        if len(images) == 0 or count_of_image == 0:
            return "There is no image in this phase."
        images_str = "\n".join(images)
        num_of_chosen_images = min(5, len(images))
        chosen_images = []
        input = PROMPT_SUMMARIZER_IMAGE_CHOOSE.format(phases_in_context=state.context, phase_name=state.phase, num=num_of_chosen_images+3, images=images_str)
        raw_reply, _ = self.llm.generate(input, [], max_completion_tokens=4096)
        with open(f'{state.restore_dir}/chosen_images_reply.txt', 'w') as f:
            f.write(raw_reply)
        try:
            raw_chosen_images = self._parse_json(raw_reply)['images']
            for image in raw_chosen_images:
                if image in images:
                    chosen_images.append(image)
                    if len(chosen_images) == num_of_chosen_images:
                        break
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")
            for image in images:
                if image in raw_reply:
                    chosen_images.append(image)
                    if len(chosen_images) == num_of_chosen_images:
                        break

        image_to_text_tool = ImageToTextTool(model='gpt-4o', type='api')
        images_to_descriptions = image_to_text_tool.image_to_text(state, chosen_images)
        insight_from_visualization = ""
        for image, description in images_to_descriptions.items():
            insight_from_visualization += f"## IMAGE: {image} ##\n{description}\n"
        with open(f'{state.restore_dir}/insight_from_visualization.txt', 'w') as f:
            f.write(insight_from_visualization)

        return insight_from_visualization

    def _generate_research_report(self, state: State) -> str:
        previous_dirs = ['pre_eda', 'data_cleaning', 'deep_eda', 'feature_engineering', 'model_build_predict']
        previous_report = ""
        for dir in previous_dirs:
            if os.path.exists(f'{state.competition_dir}/{dir}/report.txt'):
                with open(f'{state.competition_dir}/{dir}/report.txt', 'r') as f:
                    report = f.read()
                    previous_report += f"## {dir.replace('_', ' ').upper()} ##\n{report}\n"

        _, research_report_history = self.llm.generate(PROMPT_SUMMARIZER_RESEARCH_REPORT, [], max_completion_tokens=4096)
        raw_research_report, research_report_history = self.llm.generate(previous_report, research_report_history, max_completion_tokens=4096)
        try:
            research_report = self._parse_markdown(raw_research_report)
        except Exception as e:
            research_report = raw_research_report
        return research_report

    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        # implement the summarizing function, read the current state's memory and generate report
        if state.memory[-1].get("developer", {}).get("status", True) == False:
            print(f"State {state.phase} - Agent {self.role} gives up summarizing because the code execution failed.")
            return {self.role: {"history": [], "report": ""}}

        history = []
        history.append({"role": "system", "content": f"{role_prompt} {self.description}"})

        # read background_info and plan
        background_info = state.background_info
        state_info = state.get_state_info()
        with open(f'{state.restore_dir}/markdown_plan.txt', 'r') as f:
            plan = f.read()

        # Design questions
        design_questions_history = []
        next_phase_name = state.get_next_phase()
        input = PROMPT_SUMMARIZER_DESIGN_QUESITONS.format(phases_in_context=state.context, phase_name=state.phase, next_phase_name=next_phase_name)
        _, design_questions_history = self.llm.generate(input, design_questions_history, max_completion_tokens=4096)

        input = f"# INFO #\n{background_info}\n{state_info}\n#############\n# PLAN #\n{plan}"
        design_questions_reply, design_questions_history = self.llm.generate(input, design_questions_history, max_completion_tokens=4096)
        with open(f'{state.restore_dir}/design_questions_reply.txt', 'w') as f:
            f.write(design_questions_reply)

        input = PROMPT_SUMMARIZER_REORGAINZE_QUESTIONS
        reorganize_questions_reply, design_questions_history = self.llm.generate(input, design_questions_history, max_completion_tokens=4096)
        questions = self._parse_markdown(reorganize_questions_reply)
        with open(f'{state.restore_dir}/questions.txt', 'w') as f:
            f.write(questions)
        history.append(design_questions_history)

        # Answer questions
        with open(f'{state.restore_dir}/single_phase_code.txt', 'r') as f:
            code = f.read()
        with open(f'{state.restore_dir}/{state.dir_name}_output.txt', 'r') as f:
            output = f.read()
            if len(output) > 1000: # if the output is too long, truncate it
                output = output[:1000]
        with open(f'{state.restore_dir}/review.json', 'r') as f:
            review = json.load(f)

        answer_questions_history = []
        input = PROMPT_SUMMARIZER_ANSWER_QUESTIONS.format(phases_in_context=state.context, phase_name=state.phase, questions=questions)
        _, answer_questions_history = self.llm.generate(input, answer_questions_history, max_completion_tokens=4096)
        
        insight_from_visualization = self._get_insight_from_visualization(state)
        input = PROMPT_INFORMATION_FOR_ANSWER.format(background_info=background_info, state_info=state_info, plan=plan, code=code, output=output, insight_from_visualization=insight_from_visualization, review=review)
        answer_questions_reply, answer_questions_history = self.llm.generate(input, answer_questions_history, max_completion_tokens=4096)
        with open(f'{state.restore_dir}/answer_questions_reply.txt', 'w') as f:
            f.write(answer_questions_reply)

        input = PROMPT_SUMMARIZER_REORGANIZE_ANSWERS
        reorganize_answers_reply, answer_questions_history = self.llm.generate(input, answer_questions_history, max_completion_tokens=4096)
        report = self._parse_markdown(reorganize_answers_reply)
        feature_info = self._get_feature_info(state)
        report = feature_info + report
        with open(f'{state.restore_dir}/report.txt', 'w') as f:
            f.write(report)
        history.append(answer_questions_history)

        if state.phase == 'Model Building, Validation, and Prediction':
            research_report = self._generate_research_report(state)
            with open(f'{state.competition_dir}/research_report.md', 'w') as f:
                f.write(research_report)

        # save history
        with open(f'{state.restore_dir}/{self.role}_history.json', 'w') as f:
            json.dump(history, f, indent=4)

        print(f"State {state.phase} - Agent {self.role} finishes working.")
        return {self.role: {"history": history, "report": report}}
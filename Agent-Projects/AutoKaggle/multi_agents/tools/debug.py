import os
import pandas as pd
import json
import sys
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import Memory
from llm import OpenaiEmbeddings, LLM
from state import State
from utils import load_config
from prompts.prompt_developer import *

class DebugTool:
    def __init__(
        self,
        model: str = 'gpt-4o',
        type: str = 'api'       
    ):
        self.llm = LLM(model, type)

    def debug_code_with_error(self, state: State, all_error_messages: list, output_messages: str, previous_code: str, wrong_code: str, error_messages: str, tools: str, tool_names: list) -> str:
        debug_times = len(all_error_messages)
        logger.info(f"Debug times: {debug_times}")

        single_round_debug_history = []
        # locate error
        input = PROMPT_DEVELOPER_DEBUG_LOCATE.format(
            previous_code=previous_code,
            wrong_code=wrong_code,
            error_messages=error_messages,
            output_messages=output_messages
        )
        _, locate_history = self.llm.generate(input, [], max_completion_tokens=4096)
        input = f"# TOOL DESCRIPTIONS #\n{tools}"
        locate_reply, locate_history = self.llm.generate(input, locate_history, max_completion_tokens=4096)
        single_round_debug_history.append(locate_history)
        with open(f'{state.restore_dir}/debug_locate_error.txt', 'w') as f:
            f.write(locate_reply)

        if debug_times >= 3:
            all_error_info = ""
            for i, error_message in enumerate(all_error_messages):
                all_error_info += f"This is the {i}-th error message:\n{error_message}\n ------------\n"
            input = PROMPT_DEVELOPER_DEBUG_ASK_FOR_HELP.format(i=debug_times, all_error_messages=all_error_info)
            help_reply, help_history = self.llm.generate(input, [], max_completion_tokens=4096)
            single_round_debug_history.append(help_history)
            with open(f'{state.restore_dir}/debug_ask_for_help.txt', 'w') as f:
                f.write(help_reply)
            if any(keyword in help_reply for keyword in ["<HELP>", "</HELP>", "I need help", "need help"]):
                return "HELP", single_round_debug_history

        # extract code
        pattern = r"```python(.*?)```"
        error_code_matches = re.findall(pattern, locate_reply, re.DOTALL)
        try:
            most_relevant_code_snippet = error_code_matches[-1]
        except:
            most_relevant_code_snippet = "Can't find the most relevant code snippet."

        # fix bug
        input = PROMPT_DEVELOPER_DEBUG_FIX.format(
            most_relevant_code_snippet=most_relevant_code_snippet,
            error_messages=error_messages,
            output_messages=output_messages,
            tools=tools
        )
        fix_reply, fix_bug_history = self.llm.generate(input, [], max_completion_tokens=4096)
        single_round_debug_history.append(fix_bug_history)
        with open(f'{state.restore_dir}/debug_fix_bug.txt', 'w') as f:
            f.write(fix_reply)

        # extract code
        correct_code_matches = re.findall(pattern, fix_reply, re.DOTALL)
        code_snippet_after_correction = correct_code_matches[-1]

        # merge code
        input = PROMPT_DEVELOPER_DEBUG_MERGE.format(
            wrong_code=wrong_code,
            most_relevant_code_snippet=most_relevant_code_snippet,
            code_snippet_after_correction=code_snippet_after_correction
        )
        merge_reply, merge_code_history = self.llm.generate(input, [], max_completion_tokens=4096)
        single_round_debug_history.append(merge_code_history)
        with open(f'{state.restore_dir}/debug_merge_code.txt', 'w') as f:
            f.write(merge_reply)

        with open(f'{state.restore_dir}/single_round_debug_history.json', 'w') as f:
            json.dump(single_round_debug_history, f, indent=4)

        return merge_reply, single_round_debug_history

    def debug_code_with_no_pass_test(self, state: State, output_messages: str, previous_code: str, code_with_problem: str, not_pass_information: str) -> str:
        single_round_test_history = []
        # locate error
        input = PROMPT_DEVELOPER_TEST_LOCATE.format(
            previous_code=previous_code,
            code_with_problem=code_with_problem,
            not_pass_information=not_pass_information,
            output_messages=output_messages
        )
        raw_reply, test_locate_history = self.llm.generate(input, [], max_completion_tokens=4096)
        input = PROMPT_DEVELOPER_TEST_REORGANIZE_LOCATE_ANSWER
        code_snippets_with_problem, test_locate_history = self.llm.generate(input, test_locate_history, max_completion_tokens=4096)
        single_round_test_history.append(test_locate_history)
        with open(f'{state.restore_dir}/test_locate_problem.txt', 'w') as f:
            f.write(code_snippets_with_problem)

        # fix bug
        input = PROMPT_DEVELOPER_TEST_FIX.format(
            code_snippets_with_problem=code_snippets_with_problem,
            output_messages=output_messages,
            not_pass_information=not_pass_information
        )
        raw_reply, test_fix_history = self.llm.generate(input, [], max_completion_tokens=4096)
        with open(f'{state.restore_dir}/thought_to_test_fix_problem.txt', 'w') as f:
            f.write(raw_reply)
        single_round_test_history.append(test_fix_history)
        input = PROMPT_DEVELOPER_TEST_REORGANIZE_FIX_ANSWER
        code_snippets_after_correction, test_fix_history = self.llm.generate(input, test_fix_history, max_completion_tokens=4096)
        with open(f'{state.restore_dir}/test_fix_problem.txt', 'w') as f:
            f.write(code_snippets_after_correction)


        # merge code
        input = PROMPT_DEVELOPER_TEST_MERGE.format(
            code_with_problem=code_with_problem,
            code_snippets_with_problem=code_snippets_with_problem,
            code_snippets_after_correction=code_snippets_after_correction
        )
        raw_reply, merge_code_history = self.llm.generate(input, [], max_completion_tokens=4096)
        single_round_test_history.append(merge_code_history)
        with open(f'{state.restore_dir}/test_merge_code.txt', 'w') as f:
            f.write(raw_reply)

        with open(f'{state.restore_dir}/single_round_test_history.json', 'w') as f:
            json.dump(single_round_test_history, f, indent=4)

        return raw_reply, single_round_test_history
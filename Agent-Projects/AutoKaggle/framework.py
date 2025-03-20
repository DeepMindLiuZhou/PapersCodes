import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_agents.state import State
from multi_agents.sop import SOP
from utils import PREFIX_MULTI_AGENTS
import argparse
import logging

import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SOP for a competition.')
    parser.add_argument('--competition', type=str, default='titanic', help='Competition name')
    parser.add_argument('--model', type=str, default='gpt_4o', help='Model name')
    args = parser.parse_args()
    competition = args.competition
    model = args.model

    sop = SOP(competition, model)
    start_state = State(phase="Understand Background", competition=competition)
    start_message = ""
    new_state = start_state

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f"{PREFIX_MULTI_AGENTS}/competition/{competition}/{competition}.log")
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Start SOP for competition: {competition}")
    while True:
        current_state = new_state
        exec_state_info, new_state = sop.step(state=current_state)
        if exec_state_info == 'Fail':
            logging.error("Failed to update state.")
            exit()
        if exec_state_info == 'Complete':
            logging.info(f"Competition {competition} SOP is completed.")
            break  

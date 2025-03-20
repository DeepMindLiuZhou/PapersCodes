import os
from api_handler import APIHandler, APISettings
import base64
import subprocess
import re
import shutil
import json

DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX_STRONG_BASELINE = f'{DIR}/strong_baseline/competition'
PREFIX_WEAK_BASELINE = f'{DIR}/weak_baseline/competition'
PREFIX_MULTI_AGENTS = f'{DIR}/multi_agents'
SEPERATOR_TEMPLATE = '-----------------------------------{step_name}-----------------------------------'

def load_config(file_path: str):
    assert file_path.endswith('json'), "The configuration file should be in JSON format."
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def read_file(file_path: str):
    """
    Read the content of a file and return it as a string.
    """
    if file_path.endswith('txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    if file_path.endswith('csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
def multi_chat(api_handler: APIHandler, prompt, history=None, max_completion_tokens=4096):
    """
    Multi-round chat with the assistant.
    """
    if history is None:
        history = []

    messages = history + [{'role': 'user', 'content': prompt}]

    settings = APISettings(max_completion_tokens=max_completion_tokens)
    reply = api_handler.get_output(messages=messages, settings=settings)
    history.append({'role': 'user', 'content': prompt})
    history.append({'role': 'assistant', 'content': reply})
    
    return reply, history

def read_image(prompt, image_path):
    """
    Read the image and return the response.
    """
    # encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Getting the base64 string
    base64_image = encode_image(image_path)
    api_handler = APIHandler('gpt-4o')
    messages=[
        {"role": "system", "content": "You are a professional data analyst."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }
    ]
    settings = APISettings(max_completion_tokens=4096)
    reply = api_handler.get_output(messages=messages, settings=settings, response_type='image')
    return reply
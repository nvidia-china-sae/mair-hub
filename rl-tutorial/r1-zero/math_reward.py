# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import re
import random
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
from sympy import zoo, nan

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def calculate_accuracy_reward(completion, solution):
    """Reward function that checks if the completion is the same as the ground truth."""
    if completion.strip() == "":
        return -1.0

    last_boxed_str = last_boxed_only_string(completion)
    if last_boxed_str is None:
        return -1.0

    # remove \boxed
    if last_boxed_str[7:-1].strip() == solution.strip():
        return 1.0

    def _is_valid(parsed_result):
        if parsed_result is None:
            return False
        if len(parsed_result) == 0:
            return False
        if parsed_result[0] is zoo or parsed_result[0] is nan or parsed_result[1] is zoo or parsed_result[1] is nan:
            return False
        return True

    gold_parsed = parse(f"\\boxed{{{solution}}}", extraction_mode="first_match", extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        # equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
                # StringExtractionConfig()
            ],)
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            last_boxed_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        # equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
                # StringExtractionConfig()
            ],
            extraction_mode="first_match",
        )
        if len(answer_parsed) == 2 and len(gold_parsed) == 2 and _is_valid(answer_parsed) and _is_valid(gold_parsed):
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            if verify(answer_parsed, gold_parsed):
                return 1.0
            else:
                return 0.0
        else:
            return 0.0
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        return 1.0


# case = '<think>123</think><think>123</think><answer>456</answer>'
def is_format_correct(completion):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # pattern = r"^<think>.*?</think>"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    # tags = ["<think>", "</think>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # check if <think>...</think> is empty
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

def calculate_format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    return 1.0 if is_format_correct(completion) else -1.0


def extract_qwen_output(prompt):
    model_output = prompt.split("Assistant: <think>")[-1]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    model_output = model_output.strip()
    return model_output


def extract_answer_part(response):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return ""

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    response = extract_qwen_output(solution_str)
    # # extract <answer>...</answer>
    # final_answers = [extract_answer_part(response) for response in responses]
    final_answer = extract_answer_part(response)

    do_print = False
    if random.randint(0, 512) == 1:  
        do_print = True

    if not is_format_correct("<think>" + response):
        if do_print:
            print(f"[Invalid Format] Response Case: {response}")
        return 0.0
        
    # format_reward = calculate_format_reward(response)
    try:
        accuracy_reward = calculate_accuracy_reward(final_answer, ground_truth)
    except Exception as e:
        print(f"Error calculating accuracy reward: {e}")
        return 0.0
    
    if do_print:
        print(f"Response Case: {response}")
        print(f"[Reward: {accuracy_reward}] Answer Case: {final_answer} <====> GT: {ground_truth}")

    if accuracy_reward == 1.0:
        return 1.0
    else:
        return 0.0

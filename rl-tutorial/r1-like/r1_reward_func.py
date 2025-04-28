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
from math_verify import LatexExtractionConfig, StringExtractionConfig, ExprExtractionConfig, parse, verify
from transformers import AutoTokenizer

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


def calculate_accuracy_reward(completions, solution, do_print=False):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    for content, sol in zip(contents, solution):
        if content.strip() == "":
            rewards.append(0.0)
            continue

        last_boxed_str = last_boxed_only_string(content)
        if last_boxed_str is None:
            rewards.append(0.0)
            continue

        # remove \boxed
        if last_boxed_str[7:-1].strip() == sol.strip():
            rewards.append(1.0)
            continue

        gold_parsed = parse(f"\\boxed{{{sol}}}", extraction_mode="first_match", extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
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
                            equations=True,
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

            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            if verify(answer_parsed, gold_parsed):
                reward = 1.0
            else:
                reward = 0.0
            if do_print:
                print(f"[answer_parsed]: {answer_parsed}  <===> [gold_parsed]: {gold_parsed}  <===> [Score]: {reward}")
                print(f"[content]: {content} <===> [sol]: {sol}")

        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 0.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

# case = '<think>123</think><think>123</think><answer>456</answer>'
def extract_qwen_output(prompt):
    return prompt.split("<｜Assistant｜>")[-1].strip()


def calculate_length_reward(responses):
    rewards = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for response in responses:
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        response_length = len(response_tokens)
        # 当长度在5000时reward为-0.2，在10000时reward为0.2，线性变化
        if response_length <= 5000:
            rewards.append(-0.3)
        elif response_length >= 10240:
            rewards.append(0.3) 
        else:
            reward = -0.3 + (response_length - 5000) * (0.6 / 5240)
            rewards.append(reward)
    return rewards

def reward_func(queries, prompts, labels):
    answers = labels

    responses = [extract_qwen_output(query) for query in queries]

    # 将数据保存为JSON格式
    import json
    import os
    from datetime import datetime

    # 创建保存数据的目录
    save_dir = "/apps/reward_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成时间戳作为文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"reward_data_{timestamp}.json")

    # 构建要保存的数据结构
    data = []
    for query, prompt, label in zip(queries, prompts, labels):
        data.append({
            "query": query,
            "prompt": prompt,
            "label": label
        })

    # 保存为JSON文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    do_print = False
    if random.randint(0, 5) == 1:  
        do_print = True
        
    if do_print:
        print(f"Response Case: {responses[0]}")
        # print(f"Answer Case: {final_answers[0]}")

    accuracy_rewards = calculate_accuracy_reward(responses, answers, do_print=do_print)

    length_rewards = calculate_length_reward(responses)

    # 将 accuracy_rewards 和 length_rewards 相加得到最终的 rewards
    final_rewards = []
    for acc, length in zip(accuracy_rewards, length_rewards):
        if acc == 1.0:
            final_rewards.append(acc + length)
        else:
            final_rewards.append(0.0)
    
    return torch.tensor(final_rewards)
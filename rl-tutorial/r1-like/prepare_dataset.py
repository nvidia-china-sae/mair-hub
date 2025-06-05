# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from datasets import load_dataset
import json
from transformers import AutoTokenizer


user_prompt = """Please reason step by step, and put your final answer within \\boxed{}."""


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        

def download_and_save_dataset(dataset, split="train", output_file="dataset.json"):

    data_list = []
    for item in dataset:
        question = ""
        answer = ""

        for i in item:
            if i["from"] =="human" or i["from"] == "Human":
                question = i["value"]
            elif i["from"] == "assistant" or i["from"] == "Assistant":
                answer = i["ground_truth"]["value"]

        # Use Qwen's dialogue template to format the Q&A pair
        # Construct dialogue format
        if question != "" and answer != "":
            messages = [
                {"role": "user", "content": question + " " + user_prompt},
            ]
        else:
            continue
        
        # Apply Qwen's dialogue template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print("formatted_prompt: ", formatted_prompt)
        # Add the formatted dialogue to the data list
        data_list.append({
            "input": formatted_prompt,
            "ground_truth_answer": answer,
            "problem": question,
        })
    # Save the data as a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":

    dataset_name = "./orz_math_57k_collected.json"
    with open(dataset_name, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # print(data_list[0:5])
    download_and_save_dataset(data_list, split="train", output_file="orz-math-57k-distill.json")

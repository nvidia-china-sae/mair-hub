# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# 
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


import os
import datasets

import argparse

def make_map_fn(split, data_source, system):
    def process_fn(example, idx):
        problem = example.pop('problem')
        answer = example.pop('answer')
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                "content": problem + "  " + system
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer,
                'question': problem,
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')

    args = parser.parse_args()
    data_source = 'HuggingFaceH4/aime_2024'
    ds = datasets.load_dataset(data_source, trust_remote_code=True)
    system = ("You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
               "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>")

    dataset = ds['train'].map(function=make_map_fn('train', data_source, system), with_indices=True)
    dataset.to_parquet(os.path.join(args.local_dir, 'aime24.parquet'))
    print(f"{data_source} dataset has been saved to {args.local_dir} with {len(dataset)} samples")

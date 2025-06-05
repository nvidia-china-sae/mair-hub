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
import ast

def make_map_fn(split, data_source, system):

    def process_fn(example, idx):
        question = example['prompt'][0]['content']
        example['prompt'][0]['content'] += "  " + system
        ground_truth = example['reward_model']['ground_truth']
        ground_truth = ast.literal_eval(ground_truth)
        ground_truth = ground_truth[0]
        example['reward_model']['ground_truth'] = ground_truth
        example['extra_info'] = {
                'split': split,
                'index': idx,
                'answer': ground_truth,
                "question": question
                }

        return example

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')

    args = parser.parse_args()
    data_source = 'Skywork/Skywork-OR1-RL-Data'  
    ds = datasets.load_dataset(data_source)

    system = ("You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
               "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>")
             
    dataset = ds['math'].map(function=make_map_fn('math', data_source, system), with_indices=True, num_proc=8)
    dataset.to_parquet(os.path.join(args.local_dir, 'skywork_or1.parquet'))
    print(f"{data_source} dataset has been saved to {args.local_dir} with {len(dataset)} samples")

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
from PIL import Image

from datasets import Sequence
from datasets import Image as ImageData

from .utils import ImageProcessor, valid_images

def make_map_fn(split, processor, data_source, system):
    def process_fn(example, idx):
        question = example.pop('question')
        answer = example.pop('answer')
        image = example.pop('image')
        if image.mode in ['P', 'L']:
            image = image.convert('RGBA')
        image = processor(image)
        assert isinstance(image, Image.Image)

        data = {
            "data_source": data_source,
            "prompt": [
            {
                "role": "user",
                "content": "<image>\n" + question + "  " + system
            }],
            "images": [image],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "question": question,
                'answer': answer,
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'FanqingM/MMK12'
    ds = datasets.load_dataset(data_source)

    system = ("You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
               "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>")
               
    processor = ImageProcessor(max_pixels=768 * 768, min_pixels=56 * 56)
    
    dataset = ds['train'].map(function=make_map_fn('train', processor, data_source, system), with_indices=True, num_proc=8)
    # filter out the data
    print("before filter", len(dataset))
    dataset = dataset.filter(lambda x: x['reward_model']['ground_truth'] is not None and x['prompt'][0]['content'].count("<image") == 1, num_proc=8)
    print("after filter", len(dataset))
    # cast images to datasets.Image
    valid_images(dataset)
    dataset = dataset.cast_column('images', Sequence(feature=ImageData(decode=True)))
    
    dataset.to_parquet(os.path.join(args.local_dir, 'mmk12.parquet'))
    print(f"{data_source} dataset has been saved to {args.local_dir} with {len(dataset)} samples")

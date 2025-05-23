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

from datasets import Sequence
from datasets import Image as ImageData

from .utils import ImageProcessor, valid_images

def make_map_fn(split, processor, common_prefix, category_definitions, data_source, system):
    def process_fn(example, idx):
        category = example.pop('category')
        answer_option = example.pop('answer_option')
        final_answer_range = example.pop('final_answer_range')
        category_definition = category_definitions[category]
        prompt = "<image>\n" + common_prefix.format(category=category, category_definition=category_definition, answer_range=answer_option)
        answer = example.pop('final_answer')
        image = example.pop('image')
        image = processor(image)
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt + "  " + system
                }
            ],
            "images": [image],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer,
                "question": prompt,
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'him1411/polymath'
    ds = datasets.load_dataset(data_source)

    system = ("You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
               "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>")
    category_definitions = {
        "mathematical_reasoning" : "this question purely requires calculations of a mathematical nature. This includes solving a straightforward equation.",
        "pattern_recognition" : "this requires the understanding of a one-to-one relationship or pattern and replicating that pattern. For example, given the relationship between a and b, determining the equivalent of b to c. Questions involving substituting characters and operations in a pre-defined pattern fall into this category.",
        "sequence_completion" : "given a sequence of numbers or figures, this question involves finding the sequentially next element in a series.",
        "figure_completion" : "You are given a figure with an arrangement of numbers or characters such that their relationship to one another based on their position in the figure is consistent. Th goal is to complete the figure and identify the element missing from a marked position.",
        "odd_one_out" : "given a set of elements, identify the element that is not like the others.",
        "spatial_reasoning" : "questions involving reasoning observationally and visualizing the question in order to arrive at the answer.",
        "perspective_shift" : "Questions where a figure is given and you are instructed to morph it according to the instructions (flip, mirror image, rotate, etc)",
        "numerical_reasoning" : "questions involving counting the number of elements mentioned. The elements may be part of a single figure or conform to a specified pattern, but solving these questions requires counting.",
        "relative_reasoning" : "the question contains distinct data points, and solving the questions requires understanding the relationships between all data points and extrapolating relationships that are not explicitly mentioned. Questions involving venn diagrams, family relations, or relative positions given a reference point fall into this category.",
        "logical_reasoning" : "Questions involving simple logical reasoning such as entailment and contradiction."
    
    }
    common_prefix = "You are given a question to solve below:\n\nThis question requires skills and reasoning related to {category}. Definition: {category_definition}.\nThis question has a list of options : {answer_range}\n. "

               
    processor = ImageProcessor(max_pixels=768 * 768, min_pixels=56 * 56)
    
    dataset = ds['test'].map(function=make_map_fn('test', processor, common_prefix, category_definitions, data_source, system), with_indices=True, num_proc=8)
    # filter out the data
    print("before filter", len(dataset))
    dataset = dataset.filter(lambda x: x['reward_model']['ground_truth'] is not None and x['prompt'][0]['content'].count("<image") == 1, num_proc=8)
    print("after filter", len(dataset))
    # cast images to datasets.Image
    dataset = dataset.cast_column('images', Sequence(feature=ImageData()))
    valid_images(dataset)
    
    dataset.to_parquet(os.path.join(args.local_dir, 'polymath.parquet'))
    print(f"{data_source} dataset has been saved to {args.local_dir} with {len(dataset)} samples")

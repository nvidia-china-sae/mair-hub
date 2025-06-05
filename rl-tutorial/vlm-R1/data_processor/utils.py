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


import math
from PIL import Image
from dataclasses import dataclass

from datasets import Sequence
from datasets import Image as ImageData

@dataclass
class ImageProcessor:
    max_pixels: int
    min_pixels: int
    min_side: int = 28  # 添加最小边长参数，默认为28

    def __call__(self, image: Image.Image):
        flag = False
        # 处理图像总像素过多的情况
        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
            flag = True

        # 处理图像总像素过少的情况
        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
            flag = True
            
        # 确保宽度和高度都不小于最小边长
        if image.width < self.min_side or image.height < self.min_side:
            # 保持宽高比例调整
            if image.width <= image.height:
                # 宽度是较小的边
                new_width = self.min_side
                new_height = int(image.height * (self.min_side / image.width))
            else:
                # 高度是较小的边
                new_height = self.min_side
                new_width = int(image.width * (self.min_side / image.height))
            image = image.resize((new_width, new_height))
            flag = True
            
        if not flag:
            image = image.resize((image.width, image.height))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

def valid_images(dataset):
    dataset = dataset.cast_column('images', Sequence(feature=ImageData(decode=False)))
    byte_count = 0
    path_count = 0
    for example in dataset: 
        if example['images'][0]['bytes'] is None:
            byte_count += 1
            assert example['images'][0]['path'] is not None
        if example['images'][0]['path'] is None:
            path_count += 1
            assert example['images'][0]['bytes'] is not None
    assert byte_count == 0 and path_count == len(dataset), f"byte_count: {byte_count}, path_count: {path_count}"

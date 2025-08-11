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
import json
from tqdm import tqdm
import re

def load_jsonl(file_path: str):
    """Load data from jsonl file."""
    data = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            if item["language"] == "zh":
                # check if there is any english in the text
                if item["duration"] < 30:
                    count += 1
                    item["text"] = item["text"].lower()
                    if re.search(r'[a-z]', item["text"]):
                        print(item["text"])
                        continue
                    else:
                        data.append(item)
            if count > 80000:
                break
    print(f"Total data: {len(data)}")
    return data

if __name__ == "__main__":
    jsonl_file = "data/emilia_zh.jsonl"
    data = load_jsonl(jsonl_file)
    with open(f"./data/{jsonl_file.split('/')[-1].split('.')[0]}-zh-filtered.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")